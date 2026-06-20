"""Optuna tuning for CARD on one dataset at a time.

The tuner launches ``main.py`` as a subprocess.  It does not save model
checkpoints; it records trial metrics/parameters to CSV and writes a small
best-parameter summary for convenience.

Typical usage:
    pip install optuna
    python tune.py --dataset cora --n_trials 50 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import optuna


AUC_RE = re.compile(r"the\s+auc\s+is\s+([0-9eE+\-.]+)", re.IGNORECASE)
AUPRC_RE = re.compile(r"the\s+auprc\s+is\s+([0-9eE+\-.]+)", re.IGNORECASE)


@dataclass(frozen=True)
class RunResult:
    dataset: str
    seed: int
    auc: float
    auprc: float
    elapsed_sec: float
    command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune CARD hyper-parameters on one dataset.")
    parser.add_argument("--dataset", type=str, default="cora", help="Single dataset name, e.g. cora.")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URI, e.g. sqlite:///optuna_results/card.db. Omit for in-memory study.",
    )
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument(
        "--objective",
        choices=["auc", "auprc", "mean", "harmonic"],
        default="auc",
        help="Metric optimized by Optuna.",
    )
    parser.add_argument("--main", type=str, default="main.py", help="Path to CARD main.py.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch main.py.")
    parser.add_argument("--device", type=str, default=None, help="cuda:0, cuda:1, or cpu.")
    parser.add_argument("--data_root", type=str, default="~/datasets/GAD/mat")
    parser.add_argument("--artifact_root", type=str, default=".")
    parser.add_argument("--num_epoch", type=int, default=None, help="Override main.py --num_epoch.")
    parser.add_argument(
        "--auc_test_rounds",
        type=int,
        default=20,
        help="Evaluation rounds during tuning. Use a smaller value for speed.",
    )
    parser.add_argument("--seeds", type=str, default="2", help="Comma-separated seeds, e.g. 2,3,4.")
    parser.add_argument("--timeout", type=int, default=None, help="Subprocess timeout in seconds for each seed.")
    parser.add_argument("--output_dir", type=str, default="optuna_results", help="Directory for tuning CSV and summaries.")
    parser.add_argument("--force_preprocess", action="store_true", help="Forward --force_preprocess to main.py.")
    parser.add_argument("--no_earlystop", action="store_true", help="Forward --no-earlystop to main.py.")
    parser.add_argument(
        "--save_main_csv",
        action="store_true",
        help="Let each launched main.py append to its own result CSV. Disabled by default to avoid duplicate logs.",
    )
    parser.add_argument(
        "--extra_args",
        type=str,
        default="",
        help='Extra arguments passed verbatim to main.py, e.g. "--gdc_alpha 0.01".',
    )
    parser.add_argument("--dry_run", action="store_true", help="Print one sampled command without running Optuna.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_trials <= 0:
        raise ValueError("--n_trials must be positive.")
    if any(sep in args.dataset for sep in [",", " ", "\t"]):
        raise ValueError("--dataset only accepts one dataset name. Run tune.py again for another dataset.")


def parse_seed_list(seed_text: str) -> List[int]:
    seeds = [int(item.strip()) for item in seed_text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty.")
    return seeds


def sample_params(trial: optuna.Trial) -> Dict[str, object]:
    """Search space matched to the current CARD ``main.py`` CLI."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 128]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 300]),
        "subgraph_size": trial.suggest_categorical("subgraph_size", [3, 4, 5, 6, 8]),
        "negsamp_ratio": trial.suggest_categorical("negsamp_ratio", [1, 2, 3]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.6),
        "readout": trial.suggest_categorical("readout", ["avg", "max", "min"]),
        "gama": trial.suggest_float("gama", 0.0, 1.0),
        "beta": trial.suggest_float("beta", 0.0, 1.0),
    }


def params_to_cli(params: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for key, value in params.items():
        args.extend([f"--{key}", str(value)])
    return args


def build_command(base_args: argparse.Namespace, params: Dict[str, object], seed: int) -> List[str]:
    command = [
        base_args.python,
        base_args.main,
        "--dataset", base_args.dataset,
        "--seed", str(seed),
        "--data_root", base_args.data_root,
        "--artifact_root", base_args.artifact_root,
        "--auc_test_rounds", str(base_args.auc_test_rounds),
    ]
    if base_args.num_epoch is not None:
        command.extend(["--num_epoch", str(base_args.num_epoch)])
    if base_args.device is not None:
        command.extend(["--device", base_args.device])
    if base_args.force_preprocess:
        command.append("--force_preprocess")
    if base_args.no_earlystop:
        command.append("--no-earlystop")
    if not base_args.save_main_csv:
        command.append("--no_save_csv")
    command.extend(params_to_cli(params))
    if base_args.extra_args.strip():
        command.extend(shlex.split(base_args.extra_args))
    return command


def parse_metrics(output: str) -> Tuple[float, float]:
    auc_match = AUC_RE.search(output)
    auprc_match = AUPRC_RE.search(output)
    if auc_match is None or auprc_match is None:
        tail = "\n".join(output.splitlines()[-80:])
        raise ValueError(
            "Could not parse `the auc is ...` and `the auprc is ...` from main.py output.\n"
            f"Last output lines:\n{tail}"
        )
    auc = float(auc_match.group(1))
    auprc = float(auprc_match.group(1))
    if not (math.isfinite(auc) and math.isfinite(auprc)):
        raise ValueError(f"Non-finite metrics parsed: auc={auc}, auprc={auprc}")
    return auc, auprc


def run_one(base_args: argparse.Namespace, params: Dict[str, object], seed: int) -> RunResult:
    command = build_command(base_args, params, seed)
    command_text = " ".join(shlex.quote(x) for x in command)

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=base_args.timeout,
        check=False,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    elapsed = time.perf_counter() - start

    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"main.py failed with return code {completed.returncode}.\n"
            f"Command: {command_text}\n"
            f"Last output lines:\n{tail}"
        )

    auc, auprc = parse_metrics(completed.stdout)
    return RunResult(
        dataset=base_args.dataset,
        seed=seed,
        auc=auc,
        auprc=auprc,
        elapsed_sec=elapsed,
        command=command_text,
    )


def score_from_results(results: Sequence[RunResult], objective: str) -> float:
    auc = sum(r.auc for r in results) / len(results)
    auprc = sum(r.auprc for r in results) / len(results)
    if objective == "auc":
        return auc
    if objective == "auprc":
        return auprc
    if objective == "mean":
        return 0.5 * (auc + auprc)
    if objective == "harmonic":
        denom = auc + auprc
        return 0.0 if denom <= 0 else 2.0 * auc * auprc / denom
    raise ValueError(f"Unsupported objective: {objective}")


def append_rows(csv_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def make_objective(base_args: argparse.Namespace, output_dir: Path):
    seeds = parse_seed_list(base_args.seeds)
    trial_csv = output_dir / f"{base_args.study_name}.csv"

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        started_at = datetime.now().isoformat(timespec="seconds")
        results: List[RunResult] = []

        try:
            for seed in seeds:
                results.append(run_one(base_args, params, seed))
        except Exception as exc:  # Store useful failure info for Optuna.
            trial.set_user_attr("failed_params", params)
            trial.set_user_attr("error", repr(exc))
            append_rows(
                trial_csv,
                [
                    {
                        "datetime": started_at,
                        "trial": trial.number,
                        "state": "failed_or_pruned",
                        "objective": "",
                        "dataset": base_args.dataset,
                        "seed": "",
                        "auc": "",
                        "auprc": "",
                        "elapsed_sec": "",
                        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                        "command": "",
                        "error": repr(exc),
                    }
                ],
            )
            raise optuna.TrialPruned(str(exc)) from exc

        score = score_from_results(results, base_args.objective)
        mean_auc = sum(r.auc for r in results) / len(results)
        mean_auprc = sum(r.auprc for r in results) / len(results)
        trial.set_user_attr("mean_auc", mean_auc)
        trial.set_user_attr("mean_auprc", mean_auprc)
        trial.set_user_attr("params", params)

        append_rows(
            trial_csv,
            [
                {
                    "datetime": started_at,
                    "trial": trial.number,
                    "state": "complete",
                    "objective": score,
                    "dataset": result.dataset,
                    "seed": result.seed,
                    "auc": result.auc,
                    "auprc": result.auprc,
                    "elapsed_sec": round(result.elapsed_sec, 3),
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "command": result.command,
                    "error": "",
                }
                for result in results
            ],
        )
        print(
            f"[trial {trial.number}] objective={score:.6f}, "
            f"mean_auc={mean_auc:.6f}, mean_auprc={mean_auprc:.6f}, params={params}"
        )
        return score

    return objective


def save_best_params(study: optuna.Study, output_dir: Path, args: argparse.Namespace) -> None:
    summary = {
        "study_name": study.study_name,
        "dataset": args.dataset,
        "objective": args.objective,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    best_path = output_dir / f"{args.study_name}_best_params.json"
    best_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    cmd = "python main.py" +  "".join(f" --{k} {v}" for k, v in study.best_params.items())
    best_cmd = output_dir / f"{args.study_name}_cmd.sh"
    with open(best_cmd, "w") as f:
        f.write(cmd)
    print(f"Saved best parameter summary to: {best_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.study_name is None:
        args.study_name = f"card_{args.dataset}"

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
    )

    if args.dry_run:
        trial = study.ask()
        params = sample_params(trial)
        command = build_command(args, params, parse_seed_list(args.seeds)[0])
        print("Dry-run sampled params:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("Command:")
        print(" ".join(shlex.quote(x) for x in command))
        return

    study.optimize(make_objective(args, output_dir), n_trials=args.n_trials, show_progress_bar=True)
    save_best_params(study, output_dir, args)

    print("\nBest trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:")
    print(json.dumps(study.best_params, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
