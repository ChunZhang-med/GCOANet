# GCOA-Net

This repository provides the **core model implementation** from our paper:
- Graph-regularized cross-omics attention network
- Relation-aware heterogeneous message passing (gene/CpG/miRNA)
- Two-level attention readout (node-level + modality-level)
- Joint objective with graph regularization

## Structure
- `data/sample/`
  - Analysis-ready sample dataset for method demonstration
  - `labels.csv`, `mrna.csv`, `methylation.csv`, `mirna.csv`
  - `edges_cpg_gene.csv`, `edges_mirna_gene.csv`
- `src/gcoanet/model.py`
  - `GCOANet` (core model)
  - `graph_regularization_loss`
- `scripts/train_example.py`
  - Minimal runnable training/evaluation example

## Quick Start
```bash
pip install -r requirements.txt
python scripts/train_example.py --data-dir data/sample --output-dir outputs
```

## Outputs
- `outputs/metrics.json`
- `outputs/predictions.csv`

## Note
The bundled sample dataset is for code validation and method demonstration.
