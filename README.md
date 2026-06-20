# CARD Refactored

This version keeps CARD training focused on **one dataset per run**.  The training code is split into clear stages:

1. argument validation and seed setup;
2. dataset/artifact preparation;
3. model construction;
4. training;
5. evaluation;
6. optional multi-trial CSV summary.

The code no longer writes or reloads a `best.pkl` checkpoint.  Evaluation uses the model in memory after training/early stopping, so no model file is produced.

## Environment

```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn optuna pandas --torch-backend=cu128
```

## Train one dataset

```bash
python main.py --dataset cora --gama 0.6 --beta 0.9
```

Repeated trials still run on the same dataset:

```bash
python main.py --dataset cora --gama 0.6 --beta 0.9 --trials 10
```

The final evaluation prints both ROC-AUC and AUPRC:

```text
the auc is  ...
the auprc is  ...
```

A CSV summary is appended to `results/card_multitrial.csv` by default.  Disable it with:

```bash
python main.py --dataset cora --gama 0.6 --beta 0.9 --no_save_csv
```

## Tune one dataset

```bash
python tune.py --dataset cora --n_trials 50 --device cuda:0
```

The tuning script accepts only `--dataset` instead of the previous multi-dataset `--datasets`.  It calls `main.py` once per sampled parameter set and seed, records all trials in `optuna_results/card_<dataset>.csv`, and saves a small best-parameter JSON summary.  It does not save model weights.

Use multiple seeds on the same dataset if needed:

```bash
python tune.py --dataset cora --seeds 2,3,4 --n_trials 50
```

## Data and runtime artifacts

Datasets are loaded from `~/datasets/GAD/mat` by default. Runtime graph artifacts are generated automatically when missing:

- `edgelist/{dataset}.edgelist`
- `diff/diff_A_{dataset}.npy`

Use `--data_root`, `--artifact_root`, or `--force_preprocess` to override this behavior.
