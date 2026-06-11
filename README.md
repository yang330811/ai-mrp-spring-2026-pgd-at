# Adversarial Defense using PGD (MRP)

Mini Research Project — Spring 2026

## Links

| Item | Link |
|------|------|
| **Code** | this repository |
| **Slides (PDF)** | [slides/mrp_presentation.pdf](slides/mrp_presentation.pdf) *(add your PDF here)* |
| **Demo (YouTube)** | *(add your YouTube link here)* |

## tl;dr

- **Motivation**: Deep neural networks achieve high accuracy on clean data but are vulnerable to small adversarial perturbations.
- **Main idea**: Use **PGD adversarial training** (Madry et al.) to improve robustness against FGSM and PGD attacks.
- **Result**: PGD-AT trades some clean accuracy for much higher robust accuracy on MNIST.

## Literature Review

1. **Szegedy et al., 2013** — [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)  
   Discovered adversarial examples: imperceptible perturbations can fool high-accuracy models.

2. **Goodfellow et al., ICLR 2015** — [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)  
   Explained vulnerability via linear behavior in high dimensions; introduced **FGSM**.

3. **Madry et al., ICLR 2018** — [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)  
   Framed robustness as min-max optimization; introduced **PGD** and **PGD adversarial training**.

## Approach

1. Train a **standard CNN** on MNIST.
2. Train a second CNN with **PGD adversarial training**:
   - For each batch, generate adversarial examples with PGD.
   - Update the model to classify those perturbed inputs correctly.
3. Compare both models on:
   - **Clean accuracy**
   - **Robust accuracy** under FGSM and PGD
   - **Attack success rate (ASR)**

### Settings

| Setting | Value |
|---------|-------|
| Dataset | MNIST |
| Model | Small CNN (Conv → Conv → FC) |
| ε (L∞) | 0.3 |
| Training PGD | 7 steps, α = 0.01 |
| Evaluation attacks | FGSM; PGD with 40 steps, α = ε/10 |

## Setup

```bash
pip install -r requirements.txt
python mnist_pgd_defense.py --list          # see available iterations
python mnist_pgd_defense.py --iteration iter1   # run first experiment
```

MNIST is downloaded automatically on first run. Training PGD-AT on CPU may take a while.

## Iteration workflow

Experiments are defined in `EXPERIMENTS` inside `mnist_pgd_defense.py`. Each iteration has its own hyperparameters, checkpoint, and results folder.

| Iteration | Settings |
|-----------|----------|
| `iter1` | default: ε=0.3, 7 PGD steps, 10 epochs |
| `iter2` | stronger: ε=0.3, 20 PGD steps, 20 epochs |
| `iter3` | lower ε: ε=0.2, 20 PGD steps, 20 epochs |

**Typical workflow** — run iter1 first; if PGD robust acc is low, try iter2 or iter3:

```bash
# 1) First run (baseline params)
python mnist_pgd_defense.py --iteration iter1

# 2) Results not good? Try stronger training
python mnist_pgd_defense.py --iteration iter2 --retrain

# 3) Or try a smaller epsilon
python mnist_pgd_defense.py --iteration iter3 --retrain

# 4) Compare all completed iterations
python mnist_pgd_defense.py --iteration iter1 --iteration iter2 --iteration iter3
```

Add your own iteration by appending to `EXPERIMENTS` in the script.

## Outputs

**Shared**
- `checkpoints/standard_cnn.pth` — baseline CNN (trained once, shared across iterations)
- `results/iterations_summary.csv` — PGD-AT metrics across all runs
- `results/iterations_summary.png` — bar chart comparing iterations

**Per iteration** (`results/iter1/`, `results/iter2/`, …)
- `checkpoints/iterN_pgd_at.pth` — PGD-AT model for that iteration
- `defense_results.csv`, `training_history.csv`, `epsilon_robustness.csv`
- `training_curves.png`, `defense_comparison.png`, `asr_comparison.png`, `robustness_vs_epsilon.png`
- MNIST example grids (`defense_*_examples.png`, `defense_dual_model_comparison.png`, `perturbation_*.png`)

## Project Structure

```text
mnist-pgd-defense-mrp/
├── README.md
├── requirements.txt
├── adversarial_utils.py      # CNN, attacks, training helpers
├── mnist_pgd_defense.py      # main experiment script (EXPERIMENTS config)
├── checkpoints/              # model weights per iteration
├── slides/
│   └── mrp_presentation.pdf    # add your presentation PDF
└── results/
    ├── iterations_summary.csv
    ├── iter1/                # per-iteration figures and CSVs
    └── iter2/
```

## Note on Course Work

The CNN and attack utilities are adapted from the course MNIST adversarial attack assignment. This repository is a **standalone MRP project** and is separate from the homework submission folder.
