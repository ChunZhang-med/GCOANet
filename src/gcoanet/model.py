from __future__ import annotations

import torch
import torch.nn as nn


class NodeAttentionReadout(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [B, N, H]
        a = torch.softmax(self.score(torch.tanh(self.proj(h))).squeeze(-1), dim=1)  # [B, N]
        z = (h * a.unsqueeze(-1)).sum(dim=1)  # [B, H]
        return z, a


class HeteroRelationLayer(nn.Module):
    """One relation-aware heterogeneous message-passing layer."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w_g_self = nn.Linear(hidden_dim, hidden_dim)
        self.w_c_self = nn.Linear(hidden_dim, hidden_dim)
        self.w_m_self = nn.Linear(hidden_dim, hidden_dim)

        self.w_c2g = nn.Linear(hidden_dim, hidden_dim)
        self.w_m2g = nn.Linear(hidden_dim, hidden_dim)
        self.w_g2c = nn.Linear(hidden_dim, hidden_dim)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h_g: torch.Tensor,
        h_c: torch.Tensor,
        h_m: torch.Tensor,
        A_gc: torch.Tensor,
        A_gm: torch.Tensor,
        A_cg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # A_gc: [G,C], A_gm: [G,M], A_cg: [C,G]
        m_c2g = torch.einsum("gc,bch->bgh", A_gc, h_c) if A_gc.numel() else 0.0
        m_m2g = torch.einsum("gm,bmh->bgh", A_gm, h_m) if A_gm.numel() else 0.0
        m_g2c = torch.einsum("cg,bgh->bch", A_cg, h_g) if A_cg.numel() else 0.0

        h_g_new = self.act(self.w_g_self(h_g) + self.w_c2g(m_c2g) + self.w_m2g(m_m2g))
        h_c_new = self.act(self.w_c_self(h_c) + self.w_g2c(m_g2c))
        h_m_new = self.act(self.w_m_self(h_m))

        return self.drop(h_g_new), self.drop(h_c_new), self.drop(h_m_new)


class GCOANet(nn.Module):
    """
    Graph-regularized Cross-Omics Attention Network (core model).

    Inputs:
    - x_g: [B, G] gene expression
    - x_c: [B, C] promoter CpG methylation
    - x_m: [B, M] miRNA expression

    Priors:
    - A_gc: [G, C] normalized adjacency for gene <- CpG
    - A_gm: [G, M] normalized adjacency for gene <- miRNA
    """

    def __init__(
        self,
        n_gene: int,
        n_cpg: int,
        n_mirna: int,
        n_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # scalar-to-embedding initialization
        self.g_scalar = nn.Linear(1, hidden_dim)
        self.c_scalar = nn.Linear(1, hidden_dim)
        self.m_scalar = nn.Linear(1, hidden_dim)

        self.layers = nn.ModuleList([HeteroRelationLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)])

        # attentive readout per modality
        self.g_readout = NodeAttentionReadout(hidden_dim)
        self.c_readout = NodeAttentionReadout(hidden_dim)
        self.m_readout = NodeAttentionReadout(hidden_dim)

        # modality-level attention fusion
        self.modality_score = nn.Linear(hidden_dim, 1)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

        self.register_buffer("A_gc", torch.empty(0), persistent=False)
        self.register_buffer("A_gm", torch.empty(0), persistent=False)
        self.register_buffer("A_cg", torch.empty(0), persistent=False)

        # cached edge lists for graph-regularization
        self.register_buffer("edge_g_gc", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_c_gc", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_w_gc", torch.empty(0), persistent=False)

        self.register_buffer("edge_g_gm", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_m_gm", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_w_gm", torch.empty(0), persistent=False)

    def set_priors(self, A_gc: torch.Tensor, A_gm: torch.Tensor) -> None:
        self.A_gc = A_gc
        self.A_gm = A_gm

        # reverse normalization for g->c message passing
        if A_gc.numel() > 0:
            A_cg = A_gc.transpose(0, 1).contiguous()  # [C,G]
            denom = A_cg.sum(dim=1, keepdim=True)
            denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            self.A_cg = A_cg / denom
        else:
            self.A_cg = torch.empty(0, device=A_gc.device)

        # cache sparse edge lists for exact graph regularization
        if A_gc.numel() > 0:
            gi, ci = torch.nonzero(A_gc > 0, as_tuple=True)
            self.edge_g_gc = gi.long()
            self.edge_c_gc = ci.long()
            self.edge_w_gc = A_gc[gi, ci].float()
        else:
            self.edge_g_gc = torch.empty(0, dtype=torch.long, device=A_gc.device)
            self.edge_c_gc = torch.empty(0, dtype=torch.long, device=A_gc.device)
            self.edge_w_gc = torch.empty(0, device=A_gc.device)

        if A_gm.numel() > 0:
            gi, mi = torch.nonzero(A_gm > 0, as_tuple=True)
            self.edge_g_gm = gi.long()
            self.edge_m_gm = mi.long()
            self.edge_w_gm = A_gm[gi, mi].float()
        else:
            self.edge_g_gm = torch.empty(0, dtype=torch.long, device=A_gm.device)
            self.edge_m_gm = torch.empty(0, dtype=torch.long, device=A_gm.device)
            self.edge_w_gm = torch.empty(0, device=A_gm.device)

    def _scalar_embed(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        return torch.relu(layer(x.unsqueeze(-1)))

    def forward(self, x_g: torch.Tensor, x_c: torch.Tensor, x_m: torch.Tensor) -> dict[str, torch.Tensor]:
        h_g = self._scalar_embed(x_g, self.g_scalar)  # [B,G,H]
        h_c = self._scalar_embed(x_c, self.c_scalar)  # [B,C,H]
        h_m = self._scalar_embed(x_m, self.m_scalar)  # [B,M,H]

        for layer in self.layers:
            h_g, h_c, h_m = layer(h_g, h_c, h_m, self.A_gc, self.A_gm, self.A_cg)

        z_g, a_g = self.g_readout(h_g)
        z_c, a_c = self.c_readout(h_c)
        z_m, a_m = self.m_readout(h_m)

        Z = torch.stack([z_g, z_c, z_m], dim=1)  # [B,3,H]
        beta = torch.softmax(self.modality_score(torch.tanh(Z)).squeeze(-1), dim=1)  # [B,3]
        z = (Z * beta.unsqueeze(-1)).sum(dim=1)
        logits = self.cls(z)

        return {
            "logits": logits,
            "h_g": h_g,
            "h_c": h_c,
            "h_m": h_m,
            "attn_gene": a_g,
            "attn_cpg": a_c,
            "attn_mirna": a_m,
            "attn_modality": beta,
        }


def graph_regularization_loss(model: GCOANet, h_g: torch.Tensor, h_c: torch.Tensor, h_m: torch.Tensor) -> torch.Tensor:
    """Edge-wise smoothness over biologically grounded cross-omics relations."""
    device = h_g.device
    loss = torch.tensor(0.0, device=device)

    if model.edge_g_gc.numel() > 0:
        g = h_g[:, model.edge_g_gc, :]  # [B,E,H]
        c = h_c[:, model.edge_c_gc, :]  # [B,E,H]
        w = model.edge_w_gc.view(1, -1)
        l_gc = ((g - c) ** 2).sum(dim=-1) * w
        loss = loss + l_gc.mean()

    if model.edge_g_gm.numel() > 0:
        g = h_g[:, model.edge_g_gm, :]
        m = h_m[:, model.edge_m_gm, :]
        w = model.edge_w_gm.view(1, -1)
        l_gm = ((g - m) ** 2).sum(dim=-1) * w
        loss = loss + l_gm.mean()

    return loss
