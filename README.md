# CARD Refactored

This version keeps the original CARD training workflow while removing unused code paths and replacing AUPRC calculation with `sklearn.metrics.precision_recall_curve` + `sklearn.metrics.auc`.

## Environment

```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn --torch-backend=cu128
```

## Usage

```bash
python main.py --dataset cora --gama 0.6 --beta 0.9
```

The final evaluation prints both ROC-AUC and AUPRC:

```text
the auc is  ...
the auprc is  ...
```

Datasets are loaded from `~/datasets/GAD/mat` by default. Runtime artifacts are generated automatically when missing:

- `edgelist/{dataset}.edgelist`
- `diff/diff_A_{dataset}.npy`

Use `--data_root`, `--artifact_root`, or `--force_preprocess` to override this behavior.
