from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.gcoanet.model import GCOANet, graph_regularization_loss


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_adj(edges: pd.DataFrame, g_cols: list[str], x_cols: list[str], x_key: str) -> torch.Tensor:
    g_idx = {g: i for i, g in enumerate(g_cols)}
    x_idx = {x: i for i, x in enumerate(x_cols)}
    A = np.zeros((len(g_cols), len(x_cols)), dtype=np.float32)

    for _, r in edges.iterrows():
        g = r["gene"]
        x = r[x_key]
        if g in g_idx and x in x_idx:
            A[g_idx[g], x_idx[x]] = 1.0

    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    A = A / row_sum
    return torch.tensor(A, dtype=torch.float32)


def train_one_fold(
    Xg_tr: np.ndarray,
    Xc_tr: np.ndarray,
    Xm_tr: np.ndarray,
    y_tr: np.ndarray,
    Xg_te: np.ndarray,
    Xc_te: np.ndarray,
    Xm_te: np.ndarray,
    n_classes: int,
    A_gc: torch.Tensor,
    A_gm: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    wd: float = 1e-4,
    lambda_gr: float = 0.001,
    hidden_dim: int = 64,
    num_layers: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCOANet(
        Xg_tr.shape[1], Xc_tr.shape[1], Xm_tr.shape[1],
        n_classes=n_classes, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1
    ).to(device)
    model.set_priors(A_gc.to(device), A_gm.to(device))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()

    xg = torch.tensor(Xg_tr, dtype=torch.float32, device=device)
    xc = torch.tensor(Xc_tr, dtype=torch.float32, device=device)
    xm = torch.tensor(Xm_tr, dtype=torch.float32, device=device)
    y = torch.tensor(y_tr, dtype=torch.long, device=device)

    best_state = None
    best_loss = float("inf")

    for _ in range(epochs):
        model.train()
        out = model(xg, xc, xm)
        loss = ce(out["logits"], y) + lambda_gr * graph_regularization_loss(model, out["h_g"], out["h_c"], out["h_m"])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        lv = float(loss.detach().cpu())
        if lv < best_loss:
            best_loss = lv
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        tg = torch.tensor(Xg_te, dtype=torch.float32, device=device)
        tc = torch.tensor(Xc_te, dtype=torch.float32, device=device)
        tm = torch.tensor(Xm_te, dtype=torch.float32, device=device)
        logits = model(tg, tc, tm)["logits"]
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)

    return pred, prob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sample")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lambda-gr", type=float, default=0.001)
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    data_dir = (root / args.data_dir).resolve()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(data_dir / "labels.csv", index_col=0)
    mrna = pd.read_csv(data_dir / "mrna.csv", index_col=0)
    methyl = pd.read_csv(data_dir / "methylation.csv", index_col=0)
    mirna = pd.read_csv(data_dir / "mirna.csv", index_col=0)
    edges_cg = pd.read_csv(data_dir / "edges_cpg_gene.csv")
    edges_mg = pd.read_csv(data_dir / "edges_mirna_gene.csv")

    idx = labels.index.intersection(mrna.index).intersection(methyl.index).intersection(mirna.index)
    labels = labels.loc[idx]
    mrna = mrna.loc[idx]
    methyl = methyl.loc[idx]
    mirna = mirna.loc[idx]

    A_gc = build_adj(edges_cg, list(mrna.columns), list(methyl.columns), x_key="cpg_feature")
    A_gm = build_adj(edges_mg, list(mrna.columns), list(mirna.columns), x_key="mirna_feature")

    le = LabelEncoder()
    y = le.fit_transform(labels["subtype"].values)

    Xg = mrna.values.astype(np.float32)
    Xc = methyl.values.astype(np.float32)
    Xm = mirna.values.astype(np.float32)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_prob = np.zeros((len(y), len(le.classes_)), dtype=np.float32)

    for fold, (tr, te) in enumerate(skf.split(Xg, y), start=1):
        sg, sc, sm = StandardScaler(), StandardScaler(), StandardScaler()
        Xg_tr, Xg_te = sg.fit_transform(Xg[tr]), sg.transform(Xg[te])
        Xc_tr, Xc_te = sc.fit_transform(Xc[tr]), sc.transform(Xc[te])
        Xm_tr, Xm_te = sm.fit_transform(Xm[tr]), sm.transform(Xm[te])

        pred, prob = train_one_fold(
            Xg_tr, Xc_tr, Xm_tr, y[tr], Xg_te, Xc_te, Xm_te,
            n_classes=len(le.classes_), A_gc=A_gc, A_gm=A_gm,
            epochs=args.epochs, lambda_gr=args.lambda_gr,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        )
        oof_pred[te] = pred
        oof_prob[te] = prob
        print(f"fold {fold} acc={accuracy_score(y[te], pred):.3f} macro_f1={f1_score(y[te], pred, average='macro'):.3f}")

    metrics = {
        "accuracy": float(accuracy_score(y, oof_pred)),
        "macro_f1": float(f1_score(y, oof_pred, average="macro")),
        "weighted_f1": float(f1_score(y, oof_pred, average="weighted")),
        "auroc_macro_ovr": float(roc_auc_score(y, oof_prob, multi_class="ovr", average="macro")),
        "n_samples": int(len(y)),
        "n_classes": int(len(le.classes_)),
        "epochs": int(args.epochs),
        "n_splits": int(args.n_splits),
        "lambda_gr": float(args.lambda_gr),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame(oof_prob, index=labels.index, columns=[f"prob_{c}" for c in le.classes_])
    pred_df["y_true"] = labels["subtype"].values
    pred_df["y_pred"] = le.inverse_transform(oof_pred)
    pred_df.to_csv(out_dir / "predictions.csv")

    print("done")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

