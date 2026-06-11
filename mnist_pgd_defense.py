"""
MRP: Adversarial Defense using PGD Adversarial Training on MNIST.

Run one or more experiment iterations with different hyperparameters.
Each iteration saves its own checkpoints and results; a summary compares all runs.

Examples:
  python mnist_pgd_defense.py                  # run all iterations in EXPERIMENTS
  python mnist_pgd_defense.py --iteration iter1
  python mnist_pgd_defense.py --iteration iter2 --retrain
  python mnist_pgd_defense.py --list
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from adversarial_utils import (
  BATCH_SIZE,
  CNN,
  DEVICE,
  RESULTS_DIR,
  collect_correct_examples,
  compute_asr,
  ensure_results_dir,
  evaluate_clean,
  evaluate_robust,
  fgsm,
  get_dataloaders,
  pgd,
  save_adversarial_comparison,
  save_asr_comparison_chart,
  save_defense_comparison_chart,
  save_dual_model_comparison,
  save_iterations_summary_chart,
  save_perturbation_comparison,
  save_robustness_epsilon_chart,
  save_training_curves,
  train_epoch_pgd_adversarial,
  train_standard,
)

CHECKPOINT_DIR = "checkpoints"
STANDARD_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "standard_cnn.pth")

STANDARD_EPOCHS = 5
LR = 1e-3
EVAL_BATCHES = 20
EXAMPLE_COUNT = 8
EPSILON_SWEEP = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


@dataclass
class IterationConfig:
  """One experiment iteration with its own PGD-AT hyperparameters."""

  name: str
  label: str
  eps: float = 0.3
  pgd_epochs: int = 10
  train_steps: int = 7
  train_alpha: float | None = None  # None → eps / 10
  eval_steps: int = 40
  lr: float = 1e-3

  def train_alpha_value(self) -> float:
    return self.eps / 10 if self.train_alpha is None else self.train_alpha

  def eval_alpha_value(self) -> float:
    return self.eps / 10

  def pgd_model_path(self) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{self.name}_pgd_at.pth")

  def results_dir(self) -> str:
    return os.path.join(RESULTS_DIR, self.name)


# ---------------------------------------------------------------------------
# Edit this list to add / tune iterations.
# Run iter1 first; if PGD robust acc is low, uncomment or run iter2 with --retrain.
# ---------------------------------------------------------------------------
EXPERIMENTS: list[IterationConfig] = [
  IterationConfig(
    name="iter1",
    label="Iter 1: default (7 steps, 10 ep)",
    eps=0.3,
    pgd_epochs=10,
    train_steps=7,
    train_alpha=0.01,
  ),
  IterationConfig(
    name="iter2",
    label="Iter 2: stronger (20 steps, 20 ep)",
    eps=0.3,
    pgd_epochs=20,
    train_steps=20,
    train_alpha=None,
  ),
  IterationConfig(
    name="iter3",
    label="Iter 3: lower eps=0.2 (20 steps, 20 ep)",
    eps=0.2,
    pgd_epochs=20,
    train_steps=20,
    train_alpha=None,
  ),
]


def new_training_history() -> dict:
  return {"epoch": [], "loss": [], "clean_acc": []}


def load_or_train_standard_model(
  train_loader,
  test_loader,
  criterion: nn.Module,
  retrain: bool = False,
) -> tuple[CNN, dict | None]:
  model = CNN().to(DEVICE)
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)

  if os.path.exists(STANDARD_MODEL_PATH) and not retrain:
    print(f"Loading standard model from {STANDARD_MODEL_PATH}")
    model.load_state_dict(torch.load(STANDARD_MODEL_PATH, map_location=DEVICE))
    return model, None

  print("Training baseline CNN …")
  optimizer = optim.Adam(model.parameters(), lr=LR)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
  history = new_training_history()

  for epoch in range(1, STANDARD_EPOCHS + 1):
    loss = train_standard(model, train_loader, optimizer, criterion)
    acc = evaluate_clean(model, test_loader)
    history["epoch"].append(epoch)
    history["loss"].append(loss)
    history["clean_acc"].append(acc)
    print(f"  [standard] epoch {epoch}/{STANDARD_EPOCHS}  loss={loss:.4f}  clean_acc={acc * 100:.2f}%")
    scheduler.step()

  torch.save(model.state_dict(), STANDARD_MODEL_PATH)
  print(f"  Saved standard model → {STANDARD_MODEL_PATH}")
  return model, history


def load_or_train_pgd_model(
  config: IterationConfig,
  train_loader,
  test_loader,
  criterion: nn.Module,
  retrain: bool = False,
) -> tuple[CNN, dict | None]:
  model = CNN().to(DEVICE)
  model_path = config.pgd_model_path()

  if os.path.exists(model_path) and not retrain:
    print(f"Loading PGD-AT model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model, None

  print(f"Training PGD-AT for {config.name} …")
  print(
    f"  ε={config.eps}, train_steps={config.train_steps}, "
    f"train_alpha={config.train_alpha_value()}, epochs={config.pgd_epochs}"
  )
  optimizer = optim.Adam(model.parameters(), lr=config.lr)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
  history = new_training_history()

  for epoch in range(1, config.pgd_epochs + 1):
    loss = train_epoch_pgd_adversarial(
      model,
      train_loader,
      optimizer,
      criterion,
      epsilon=config.eps,
      alpha=config.train_alpha_value(),
      num_steps=config.train_steps,
    )
    clean_acc = evaluate_clean(model, test_loader)
    history["epoch"].append(epoch)
    history["loss"].append(loss)
    history["clean_acc"].append(clean_acc)
    print(
      f"  [{config.name}] epoch {epoch}/{config.pgd_epochs}  "
      f"loss={loss:.4f}  clean_acc={clean_acc * 100:.2f}%"
    )
    scheduler.step()

  torch.save(model.state_dict(), model_path)
  print(f"  Saved PGD-AT model → {model_path}")
  return model, history


def build_attack_fns(epsilon: float, eval_alpha: float, eval_steps: int) -> dict:
  return {
    "FGSM": lambda m, x, y, c: fgsm(m, x, y, epsilon, c),
    "PGD": lambda m, x, y, c: pgd(m, x, y, epsilon, eval_alpha, eval_steps, c),
  }


def evaluate_model_suite(
  model,
  model_name: str,
  test_loader,
  attacks: dict,
  criterion: nn.Module,
  n_batches: int = EVAL_BATCHES,
) -> dict:
  clean_acc = evaluate_clean(model, test_loader)
  fgsm_fn = attacks["FGSM"]
  pgd_fn = attacks["PGD"]

  robust_fgsm = evaluate_robust(model, test_loader, fgsm_fn, criterion, n_batches=n_batches)
  robust_pgd = evaluate_robust(model, test_loader, pgd_fn, criterion, n_batches=n_batches)
  asr_fgsm = compute_asr(model, test_loader, fgsm_fn, criterion, n_batches=n_batches)
  asr_pgd = compute_asr(model, test_loader, pgd_fn, criterion, n_batches=n_batches)

  row = {
    "Model": model_name,
    "Clean Acc (%)": round(clean_acc * 100, 2),
    "Robust Acc FGSM (%)": round(robust_fgsm * 100, 2),
    "Robust Acc PGD (%)": round(robust_pgd * 100, 2),
    "ASR FGSM (%)": round(asr_fgsm * 100, 2),
    "ASR PGD (%)": round(asr_pgd * 100, 2),
  }

  print(f"\n{model_name}")
  print(f"  clean accuracy      = {row['Clean Acc (%)']:.2f}%")
  print(f"  robust acc (FGSM)   = {row['Robust Acc FGSM (%)']:.2f}%")
  print(f"  robust acc (PGD)    = {row['Robust Acc PGD (%)']:.2f}%")
  print(f"  ASR (FGSM)          = {row['ASR FGSM (%)']:.2f}%")
  print(f"  ASR (PGD)           = {row['ASR PGD (%)']:.2f}%")
  return row


def evaluate_epsilon_sweep(
  standard_model: CNN,
  pgd_model: CNN,
  test_loader,
  criterion: nn.Module,
  epsilons: list[float],
  eval_steps: int,
  n_batches: int,
) -> pd.DataFrame:
  rows = []

  for model, model_name in [
    (standard_model, "Standard CNN"),
    (pgd_model, "PGD Adversarial Training"),
  ]:
    for epsilon in epsilons:
      if epsilon == 0.0:
        robust_fgsm = robust_pgd = evaluate_clean(model, test_loader)
      else:
        eval_alpha = epsilon / 10
        attacks = build_attack_fns(epsilon, eval_alpha, eval_steps)
        robust_fgsm = evaluate_robust(model, test_loader, attacks["FGSM"], criterion, n_batches=n_batches)
        robust_pgd = evaluate_robust(model, test_loader, attacks["PGD"], criterion, n_batches=n_batches)

      rows.append({
        "Epsilon": epsilon,
        "Model": model_name,
        "Attack": "FGSM",
        "Robust Acc (%)": round(robust_fgsm * 100, 2),
      })
      rows.append({
        "Epsilon": epsilon,
        "Model": model_name,
        "Attack": "PGD",
        "Robust Acc (%)": round(robust_pgd * 100, 2),
      })

  return pd.DataFrame(rows)


def save_training_history_csv(
  iteration_name: str,
  standard_history: dict | None,
  pgd_history: dict | None,
  results_dir: str,
) -> None:
  frames = []
  if standard_history is not None:
    df = pd.DataFrame(standard_history)
    df["Model"] = "Standard CNN"
    df["Clean Acc (%)"] = (df["clean_acc"] * 100).round(2)
    frames.append(df[["Model", "epoch", "loss", "Clean Acc (%)"]])
  if pgd_history is not None:
    df = pd.DataFrame(pgd_history)
    df["Model"] = "PGD Adversarial Training"
    df["Clean Acc (%)"] = (df["clean_acc"] * 100).round(2)
    frames.append(df[["Model", "epoch", "loss", "Clean Acc (%)"]])

  if frames:
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "Iteration", iteration_name)
    path = os.path.join(results_dir, "training_history.csv")
    out.to_csv(path, index=False)
    print(f"Saved training history → {path}")


def run_iteration(
  config: IterationConfig,
  standard_model: CNN,
  train_loader,
  test_loader,
  criterion: nn.Module,
  retrain: bool = False,
) -> dict:
  """Run one experiment iteration; save per-iteration results and return summary row."""
  print(f"\n{'=' * 60}")
  print(f"Iteration: {config.name} — {config.label}")
  print(f"{'=' * 60}")

  results_dir = ensure_results_dir(config.results_dir())
  attacks = build_attack_fns(config.eps, config.eval_alpha_value(), config.eval_steps)

  pgd_model, pgd_history = load_or_train_pgd_model(
    config, train_loader, test_loader, criterion, retrain=retrain
  )

  if pgd_history is not None:
    save_training_history_csv(config.name, None, pgd_history, results_dir)
    curves_path = os.path.join(results_dir, "training_curves.png")
    save_training_curves(None, pgd_history, curves_path)
    print(f"Saved training curves → {curves_path}")
  elif retrain:
    print("No new training curves (retrain was requested but model was not retrained).")
  else:
    print("Skipping training curves (PGD-AT loaded from checkpoint; use --retrain to retrain)")

  print(f"\nEvaluating {config.name} under FGSM and PGD (ε={config.eps}) …")
  rows = [
    evaluate_model_suite(standard_model, "Standard CNN", test_loader, attacks, criterion),
    evaluate_model_suite(pgd_model, "PGD Adversarial Training", test_loader, attacks, criterion),
  ]
  results_df = pd.DataFrame(rows)

  csv_path = os.path.join(results_dir, "defense_results.csv")
  results_df.to_csv(csv_path, index=False)
  print(f"\nResults saved → {csv_path}")
  print(results_df.to_string(index=False))

  save_defense_comparison_chart(
    results_df,
    os.path.join(results_dir, "defense_comparison.png"),
    title=f"{config.label} (ε={config.eps})",
  )
  save_asr_comparison_chart(
    results_df,
    os.path.join(results_dir, "asr_comparison.png"),
    title=f"ASR — {config.label} (ε={config.eps})",
  )
  print(f"Saved charts → {results_dir}/defense_comparison.png, asr_comparison.png")

  print("\nEvaluating robustness across epsilon values …")
  epsilon_df = evaluate_epsilon_sweep(
    standard_model,
    pgd_model,
    test_loader,
    criterion,
    EPSILON_SWEEP,
    config.eval_steps,
    EVAL_BATCHES,
  )
  epsilon_df.insert(0, "Iteration", config.name)
  epsilon_df.to_csv(os.path.join(results_dir, "epsilon_robustness.csv"), index=False)
  save_robustness_epsilon_chart(
    epsilon_df,
    os.path.join(results_dir, "robustness_vs_epsilon.png"),
    title=f"Robust Accuracy vs ε — {config.label}",
  )
  print(f"Saved epsilon sweep → {results_dir}/epsilon_robustness.csv")

  examples, example_labels = collect_correct_examples(standard_model, test_loader, n=EXAMPLE_COUNT)
  if examples is not None:
    for model, name, filename in [
      (standard_model, "Standard CNN", "defense_standard_examples.png"),
      (pgd_model, "PGD-AT CNN", "defense_pgd_at_examples.png"),
    ]:
      save_adversarial_comparison(
        model,
        examples,
        example_labels,
        attacks,
        criterion,
        os.path.join(results_dir, filename),
        suptitle=f"{config.label} — {name} (ε={config.eps})",
      )

    save_dual_model_comparison(
      standard_model,
      pgd_model,
      examples,
      example_labels,
      attacks,
      criterion,
      os.path.join(results_dir, "defense_dual_model_comparison.png"),
      config.eps,
    )

    for model, name, filename in [
      (standard_model, "Standard CNN", "perturbation_standard_pgd.png"),
      (pgd_model, "PGD-AT CNN", "perturbation_pgd_at_pgd.png"),
    ]:
      save_perturbation_comparison(
        model,
        examples[:4],
        example_labels[:4],
        attacks["PGD"],
        criterion,
        os.path.join(results_dir, filename),
        attack_name=f"PGD (ε={config.eps})",
      )
    print(f"Saved MNIST visualizations → {results_dir}/")

  pgd_row = results_df[results_df["Model"] == "PGD Adversarial Training"].iloc[0]
  return {
    "Iteration": config.name,
    "Label": config.label,
    "epsilon": config.eps,
    "pgd_epochs": config.pgd_epochs,
    "train_steps": config.train_steps,
    "train_alpha": config.train_alpha_value(),
    "eval_steps": config.eval_steps,
    "Clean Acc (%)": pgd_row["Clean Acc (%)"],
    "Robust Acc FGSM (%)": pgd_row["Robust Acc FGSM (%)"],
    "Robust Acc PGD (%)": pgd_row["Robust Acc PGD (%)"],
    "ASR FGSM (%)": pgd_row["ASR FGSM (%)"],
    "ASR PGD (%)": pgd_row["ASR PGD (%)"],
  }


def save_iterations_summary(summary_rows: list[dict]) -> None:
  if not summary_rows:
    return

  ensure_results_dir(RESULTS_DIR)
  summary_df = pd.DataFrame(summary_rows)
  csv_path = os.path.join(RESULTS_DIR, "iterations_summary.csv")
  summary_df.to_csv(csv_path, index=False)
  chart_path = os.path.join(RESULTS_DIR, "iterations_summary.png")
  save_iterations_summary_chart(summary_df, chart_path)

  print(f"\n{'=' * 60}")
  print("Iterations summary (PGD-AT only)")
  print(f"{'=' * 60}")
  print(summary_df.to_string(index=False))
  print(f"\nSaved summary → {csv_path}")
  print(f"Saved summary chart → {chart_path}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run PGD adversarial defense experiments in configurable iterations.",
  )
  parser.add_argument(
    "--iteration",
    action="append",
    dest="iterations",
    metavar="NAME",
    help="Run a specific iteration (e.g. iter1). Can be repeated. Default: run all in EXPERIMENTS.",
  )
  parser.add_argument(
    "--retrain",
    action="store_true",
    help="Force retrain PGD-AT model for the selected iteration(s).",
  )
  parser.add_argument(
    "--retrain-standard",
    action="store_true",
    help="Force retrain the shared standard CNN baseline.",
  )
  parser.add_argument(
    "--list",
    action="store_true",
    help="List available iterations and exit.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  if args.list:
    print("Available iterations:\n")
    for cfg in EXPERIMENTS:
      print(f"  {cfg.name}: {cfg.label}")
      print(
        f"    ε={cfg.eps}, epochs={cfg.pgd_epochs}, train_steps={cfg.train_steps}, "
        f"train_alpha={cfg.train_alpha_value()}, results→{cfg.results_dir()}"
      )
    return

  if args.iterations:
    selected = []
    known = {cfg.name: cfg for cfg in EXPERIMENTS}
    for name in args.iterations:
      if name not in known:
        raise SystemExit(f"Unknown iteration '{name}'. Use --list to see options.")
      selected.append(known[name])
  else:
    selected = EXPERIMENTS

  print(f"Using device: {DEVICE}")
  print(f"Will run {len(selected)} iteration(s): {[c.name for c in selected]}")

  train_loader, test_loader = get_dataloaders(BATCH_SIZE, Path("data"))
  criterion = nn.CrossEntropyLoss()

  standard_model, _ = load_or_train_standard_model(
    train_loader,
    test_loader,
    criterion,
    retrain=args.retrain_standard,
  )

  summary_rows = []
  for config in selected:
    summary_rows.append(
      run_iteration(
        config,
        standard_model,
        train_loader,
        test_loader,
        criterion,
        retrain=args.retrain,
      )
    )

  save_iterations_summary(summary_rows)
  print("\nDone! MRP defense experiments complete.")


if __name__ == "__main__":
  main()
