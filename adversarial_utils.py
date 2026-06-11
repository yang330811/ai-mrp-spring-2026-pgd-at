"""
Shared utilities for the MRP project:
Adversarial Defense using PGD on MNIST.

CNN architecture and attack helpers are adapted from the course Assignment 7.1
MNIST adversarial attack pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LR = 1e-3
RESULTS_DIR = "results"


def get_dataloaders(batch_size: int = BATCH_SIZE, data_dir: Path | str = "data") -> tuple[DataLoader, DataLoader]:
  """Load MNIST train/test sets with pixels in [0, 1]."""
  tf = transforms.Compose([transforms.ToTensor()])
  data_dir = Path(data_dir)
  train_set = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
  test_set = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
  return train_loader, test_loader


class CNN(nn.Module):
  """MNIST CNN used as the baseline classifier."""

  def __init__(self, num_classes: int = 10):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64 * 7 * 7, 128),
      nn.ReLU(),
      nn.Linear(128, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.features(x))


def train_standard(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
  """Standard supervised training on clean images."""
  model.train()
  total_loss = 0.0
  for images, labels in loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  return total_loss / len(loader)


def evaluate_clean(model: nn.Module, loader: DataLoader) -> float:
  """Clean accuracy on unperturbed images."""
  model.eval()
  correct = total = 0
  with torch.no_grad():
    for images, labels in loader:
      images, labels = images.to(DEVICE), labels.to(DEVICE)
      preds = model(images).argmax(dim=1)
      correct += (preds == labels).sum().item()
      total += labels.size(0)
  return correct / total if total > 0 else 0.0


def fgsm(
  model: nn.Module,
  images: torch.Tensor,
  labels: torch.Tensor,
  epsilon: float,
  criterion: nn.Module,
) -> torch.Tensor:
  """Fast Gradient Sign Method (Goodfellow et al., ICLR 2015)."""
  images = images.clone().detach().requires_grad_(True)
  model.zero_grad()
  loss = criterion(model(images), labels)
  loss.backward()
  adv = images + epsilon * images.grad.sign()
  adv = adv.detach()
  return torch.clamp(adv, 0.0, 1.0)


def pgd(
  model: nn.Module,
  images: torch.Tensor,
  labels: torch.Tensor,
  epsilon: float,
  alpha: float,
  num_steps: int,
  criterion: nn.Module,
  random_start: bool = True,
) -> torch.Tensor:
  """Projected Gradient Descent attack (Madry et al., ICLR 2018)."""
  adv = images.clone().detach()

  if random_start:
    adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)

  for _ in range(num_steps):
    adv.requires_grad_(True)
    model.zero_grad()
    loss = criterion(model(adv), labels)
    grad = torch.autograd.grad(loss, adv)[0]

    adv = adv.detach() + alpha * grad.sign()
    delta = torch.clamp(adv - images, min=-epsilon, max=epsilon)
    adv = torch.clamp(images + delta, 0.0, 1.0)

  return adv.detach()


def train_epoch_pgd_adversarial(
  model: nn.Module,
  loader: DataLoader,
  optimizer: optim.Optimizer,
  criterion: nn.Module,
  epsilon: float,
  alpha: float,
  num_steps: int,
) -> float:
  """
  One epoch of PGD adversarial training (Madry et al.).

  For each mini-batch, generate adversarial examples with PGD and update
  the model to classify those perturbed inputs correctly.
  """
  model.train()
  total_loss = 0.0

  for images, labels in loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    adv = pgd(
      model,
      images,
      labels,
      epsilon=epsilon,
      alpha=alpha,
      num_steps=num_steps,
      criterion=criterion,
      random_start=True,
    )

    optimizer.zero_grad()
    loss = criterion(model(adv), labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  return total_loss / len(loader)


def evaluate_robust(
  model: nn.Module,
  loader: DataLoader,
  attack_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, nn.Module], torch.Tensor],
  criterion: nn.Module,
  n_batches: Optional[int] = None,
) -> float:
  """Robust accuracy: fraction of samples still classified correctly after attack."""
  model.eval()
  correct = total = 0

  for batch_idx, (images, labels) in enumerate(loader):
    if n_batches is not None and batch_idx >= n_batches:
      break

    images, labels = images.to(DEVICE), labels.to(DEVICE)
    adv = attack_fn(model, images, labels, criterion)

    with torch.no_grad():
      preds = model(adv).argmax(dim=1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)

  return correct / total if total > 0 else 0.0


def compute_asr(
  model: nn.Module,
  loader: DataLoader,
  attack_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, nn.Module], torch.Tensor],
  criterion: nn.Module,
  n_batches: int = 20,
) -> float:
  """Attack success rate among samples that were correct on clean data."""
  model.eval()
  fooled = total_correct = 0

  for batch_idx, (images, labels) in enumerate(loader):
    if batch_idx >= n_batches:
      break

    images, labels = images.to(DEVICE), labels.to(DEVICE)
    with torch.no_grad():
      clean_preds = model(images).argmax(dim=1)

    correct_mask = clean_preds == labels
    if correct_mask.sum() == 0:
      continue

    imgs_correct = images[correct_mask]
    lbls_correct = labels[correct_mask]
    adv = attack_fn(model, imgs_correct, lbls_correct, criterion)

    with torch.no_grad():
      adv_preds = model(adv).argmax(dim=1)

    fooled += (adv_preds != lbls_correct).sum().item()
    total_correct += correct_mask.sum().item()

  return fooled / total_correct if total_correct > 0 else 0.0


def collect_correct_examples(
  model: nn.Module,
  loader: DataLoader,
  n: int = 4,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
  """Return n test images the model classifies correctly."""
  model.eval()
  imgs, lbls = [], []

  with torch.no_grad():
    for images, labels in loader:
      images, labels = images.to(DEVICE), labels.to(DEVICE)
      preds = model(images).argmax(dim=1)
      mask = preds == labels
      if not mask.any():
        continue

      idx = mask.nonzero(as_tuple=True)[0]
      for j in idx:
        imgs.append(images[j])
        lbls.append(labels[j])
        if len(imgs) >= n:
          return torch.stack(imgs), torch.stack(lbls)

  if not imgs:
    return None, None
  return torch.stack(imgs), torch.stack(lbls)


def save_training_curves(
  standard_history: Optional[dict],
  pgd_history: Optional[dict],
  save_path: str,
) -> None:
  """Plot training loss and clean accuracy curves for both models."""
  fig, axes = plt.subplots(1, 2, figsize=(10, 4))

  for history, label, color in [
    (standard_history, "Standard CNN", "#4C72B0"),
    (pgd_history, "PGD-AT CNN", "#DD8452"),
  ]:
    if history is None:
      continue
    epochs = history["epoch"]
    axes[0].plot(epochs, history["loss"], marker="o", label=label, color=color, linewidth=2)
    axes[1].plot(
      epochs,
      [acc * 100 for acc in history["clean_acc"]],
      marker="o",
      label=label,
      color=color,
      linewidth=2,
    )

  axes[0].set_xlabel("Epoch")
  axes[0].set_ylabel("Loss")
  axes[0].set_title("Training Loss")
  axes[0].legend()
  axes[0].grid(True, alpha=0.3)

  axes[1].set_xlabel("Epoch")
  axes[1].set_ylabel("Clean Accuracy (%)")
  axes[1].set_title("Clean Accuracy on Test Set")
  axes[1].set_ylim(0, 105)
  axes[1].legend()
  axes[1].grid(True, alpha=0.3)

  fig.tight_layout()
  fig.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def save_asr_comparison_chart(
  results_df: pd.DataFrame,
  save_path: str,
  title: str = "Attack Success Rate (ASR)",
) -> None:
  """Grouped bar chart of ASR under FGSM and PGD for each model."""
  models = results_df["Model"].tolist()
  x = range(len(models))
  width = 0.35

  fig, ax = plt.subplots(figsize=(8, 4.5))
  asr_fgsm = results_df["ASR FGSM (%)"].tolist()
  asr_pgd = results_df["ASR PGD (%)"].tolist()

  ax.bar([i - width / 2 for i in x], asr_fgsm, width, label="ASR (FGSM)", color="#C44E52", edgecolor="black", linewidth=0.6)
  ax.bar([i + width / 2 for i in x], asr_pgd, width, label="ASR (PGD)", color="#8172B3", edgecolor="black", linewidth=0.6)

  ax.set_ylabel("Attack Success Rate (%)")
  ax.set_title(title)
  ax.set_xticks(list(x))
  ax.set_xticklabels(models)
  ax.set_ylim(0, 105)
  ax.legend()
  fig.tight_layout()
  fig.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def save_iterations_summary_chart(
  summary_df: pd.DataFrame,
  save_path: str,
  title: str = "PGD-AT Results Across Iterations",
) -> None:
  """Compare key PGD-AT metrics across experiment iterations."""
  labels = summary_df["Label"].tolist()
  x = range(len(labels))
  width = 0.25

  fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.5), 4.5))
  metrics = [
    ("Clean Acc (%)", "#4C72B0"),
    ("Robust Acc FGSM (%)", "#DD8452"),
    ("Robust Acc PGD (%)", "#55A868"),
  ]

  for offset, (col, color) in enumerate(metrics):
    positions = [i + (offset - 1) * width for i in x]
    ax.bar(
      positions,
      summary_df[col].tolist(),
      width,
      label=col.replace(" (%)", ""),
      color=color,
      edgecolor="black",
      linewidth=0.6,
    )

  ax.set_ylabel("Accuracy (%)")
  ax.set_title(title)
  ax.set_xticks(list(x))
  ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
  ax.set_ylim(0, 105)
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def save_robustness_epsilon_chart(
  epsilon_df: pd.DataFrame,
  save_path: str,
  title: str = "Robust Accuracy vs Epsilon",
) -> None:
  """Line chart of robust accuracy vs epsilon for FGSM and PGD attacks."""
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
  model_styles = {
    "Standard CNN": {"color": "#4C72B0", "marker": "o"},
    "PGD Adversarial Training": {"color": "#DD8452", "marker": "s"},
  }

  for ax, attack in zip(axes, ["FGSM", "PGD"]):
    subset = epsilon_df[epsilon_df["Attack"] == attack]
    for model_name, style in model_styles.items():
      model_rows = subset[subset["Model"] == model_name].sort_values("Epsilon")
      ax.plot(
        model_rows["Epsilon"],
        model_rows["Robust Acc (%)"],
        label=model_name,
        linewidth=2,
        markersize=6,
        **style,
      )

    ax.set_xlabel("Epsilon (L∞)")
    ax.set_ylabel("Robust Accuracy (%)")
    ax.set_title(f"Robust Accuracy under {attack}")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

  fig.suptitle(title, fontsize=12, y=1.02)
  fig.tight_layout()
  fig.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def save_dual_model_comparison(
  standard_model: nn.Module,
  pgd_model: nn.Module,
  images: torch.Tensor,
  labels: torch.Tensor,
  attacks_dict: dict,
  criterion: nn.Module,
  save_path: str,
  epsilon: float,
) -> None:
  """Side-by-side comparison: same digits under both models and attacks."""
  n = images.size(0)
  attack_names = list(attacks_dict.keys())
  n_cols = 1 + 2 * len(attack_names)

  fig, axes = plt.subplots(n, n_cols, figsize=(2.2 * n_cols, 2.4 * n))
  if n == 1:
    axes = axes.reshape(1, -1)

  standard_model.eval()
  pgd_model.eval()

  for row in range(n):
    img = images[row : row + 1]
    lbl = labels[row : row + 1]
    true_lbl = int(lbl.item())

    ax = axes[row, 0]
    ax.imshow(img[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Clean\ntrue={true_lbl}", fontsize=8)
    ax.axis("off")

    col = 1
    for model, prefix in [(standard_model, "Std"), (pgd_model, "PGD-AT")]:
      for name in attack_names:
        adv = attacks_dict[name](model, img, lbl, criterion)
        with torch.no_grad():
          pred = int(model(adv).argmax(dim=1).item())
        fooled = pred != true_lbl
        status = "FOOL" if fooled else "ok"

        ax = axes[row, col]
        ax.imshow(adv[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{prefix} {name}\npred={pred} ({status})", fontsize=7)
        ax.axis("off")
        col += 1

  fig.suptitle(f"Model Comparison: Standard vs PGD-AT (epsilon={epsilon})", fontsize=12, y=1.02)
  fig.tight_layout()
  fig.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def save_perturbation_comparison(
  model: nn.Module,
  images: torch.Tensor,
  labels: torch.Tensor,
  attack_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, nn.Module], torch.Tensor],
  criterion: nn.Module,
  save_path: str,
  attack_name: str = "PGD",
) -> None:
  """Grid: clean | perturbation (magnified) | adversarial for each sample."""
  n = images.size(0)
  model.eval()

  fig, axes = plt.subplots(n, 3, figsize=(8, 2.6 * n))
  if n == 1:
    axes = axes.reshape(1, -1)

  for row in range(n):
    img = images[row : row + 1]
    lbl = labels[row : row + 1]
    true_lbl = int(lbl.item())

    adv = attack_fn(model, img, lbl, criterion)
    perturbation = (adv - img).detach()

    with torch.no_grad():
      clean_pred = int(model(img).argmax(dim=1).item())
      adv_pred = int(model(adv).argmax(dim=1).item())

    panels = [
      (img[0, 0].cpu(), f"Clean\ntrue={true_lbl}, pred={clean_pred}"),
      (perturbation[0, 0].cpu(), f"Perturbation\n(x10)"),
      (adv[0, 0].cpu(), f"{attack_name}\npred={adv_pred}"),
    ]

    for col, (tensor, title) in enumerate(panels):
      ax = axes[row, col]
      if col == 1:
        ax.imshow(tensor * 10, cmap="RdBu_r", vmin=-1, vmax=1)
      else:
        ax.imshow(tensor, cmap="gray", vmin=0, vmax=1)
      ax.set_title(title, fontsize=9)
      ax.axis("off")

  fig.suptitle(f"Adversarial Perturbations ({attack_name})", fontsize=12, y=1.01)
  fig.tight_layout()
  fig.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def save_defense_comparison_chart(
  results_df: pd.DataFrame,
  save_path: str,
  title: str = "Standard vs PGD Adversarial Training",
) -> None:
  """Grouped bar chart of clean and robust accuracy for each model."""
  models = results_df["Model"].tolist()
  x = range(len(models))
  width = 0.35

  fig, ax = plt.subplots(figsize=(8, 4.5))
  clean = results_df["Clean Acc (%)"].tolist()
  fgsm_robust = results_df["Robust Acc FGSM (%)"].tolist()
  pgd_robust = results_df["Robust Acc PGD (%)"].tolist()

  ax.bar([i - width for i in x], clean, width, label="Clean", color="#4C72B0", edgecolor="black", linewidth=0.6)
  ax.bar(x, fgsm_robust, width, label="Robust (FGSM)", color="#DD8452", edgecolor="black", linewidth=0.6)
  ax.bar([i + width for i in x], pgd_robust, width, label="Robust (PGD)", color="#55A868", edgecolor="black", linewidth=0.6)

  ax.set_ylabel("Accuracy (%)")
  ax.set_title(title)
  ax.set_xticks(list(x))
  ax.set_xticklabels(models)
  ax.set_ylim(0, 105)
  ax.legend()
  fig.tight_layout()
  fig.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def save_adversarial_comparison(
  model: nn.Module,
  images: torch.Tensor,
  labels: torch.Tensor,
  attacks_dict: dict,
  criterion: nn.Module,
  save_path: str,
  suptitle: str,
) -> None:
  """Grid: each row = one digit; columns = clean + one column per attack."""
  n = images.size(0)
  attack_names = list(attacks_dict.keys())
  n_cols = 1 + len(attack_names)

  fig, axes = plt.subplots(n, n_cols, figsize=(2.8 * n_cols, 2.8 * n))
  if n == 1:
    axes = axes.reshape(1, -1)

  model.eval()
  for row in range(n):
    img = images[row : row + 1]
    lbl = labels[row : row + 1]
    true_lbl = int(lbl.item())

    with torch.no_grad():
      clean_pred = int(model(img).argmax(dim=1).item())

    ax = axes[row, 0]
    ax.imshow(img[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Clean\ntrue={true_lbl}, pred={clean_pred}", fontsize=9)
    ax.axis("off")

    for col, name in enumerate(attack_names, start=1):
      adv = attacks_dict[name](model, img, lbl, criterion)
      with torch.no_grad():
        adv_pred = int(model(adv).argmax(dim=1).item())
      fooled = adv_pred != true_lbl

      ax = axes[row, col]
      ax.imshow(adv[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
      status = "FOOLed" if fooled else "same"
      ax.set_title(f"{name}\npred={adv_pred} ({status})", fontsize=9)
      ax.axis("off")

  fig.suptitle(suptitle, fontsize=12, y=1.02)
  fig.tight_layout()
  fig.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def ensure_results_dir(results_dir: str = RESULTS_DIR) -> str:
  os.makedirs(results_dir, exist_ok=True)
  return results_dir
