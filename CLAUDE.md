# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Diploma thesis: simulating rising bubble dynamics with Physics-Informed Neural Networks (PINN). The repo has two parts:
- **`diploma/`** — Python ML code (PINN training, evaluation, CFD data pipeline)
- **`main.tex`** — LaTeX thesis (Russian, pdflatex, all chapters complete)

## Python environment

```bash
source .venv/bin/activate          # from repo root
pip install -r diploma/requirements.txt
```

All Python commands below assume the venv is active and `cwd` is `diploma/`.

## Training

```bash
# Parametric PINN on 5 training radii (folder of .h5 files)
python train_rising_bubble.py --preset paper \
  --data-path exp/train/ \
  --output-dir checkpoints_param/my_run

# Simple (non-parametric) PINN on one radius
python train_rising_bubble.py --preset paper \
  --ordinary-pinn \
  --data-path exp/train/rising_bubble_R025.h5 \
  --output-dir checkpoints/my_simple

# Smoke test (fast, tiny arch, 2 epochs)
python train_rising_bubble.py --preset smoke --data-path exp/train/
```

Available presets in `pinn/config.py`: `smoke`, `default`, `paper`, `paper_light`.  
Resume from checkpoint: `--resume checkpoints_param/my_run/last.pt`

## Evaluation

```bash
cd diploma

# Predefined reports
python run_eval.py --report arch        # 3 arch comparison on R=0.30
python run_eval.py --report stages      # 2-stage vs 4-stage
python run_eval.py --report ablation    # sampling ablation on R=0.20, R=0.30
python run_eval.py --report simple      # simple PINN: train vs test radii
python run_eval.py --report all         # everything

# Custom evaluation with figures
python run_eval.py \
  --checkpoint checkpoints_param/hard_2stage/best.pt \
  --data exp/test/rising_bubble_R030.h5 \
  --figures ../figures/param_hard/
```

**Important**: pressure L2 errors are computed after gauge-alignment (matching mean at the top boundary). Without this, pressure errors appear ~68% due to an additive constant inherent to incompressible flow.

## CFD data generation (Basilisk)

Basilisk binary: `/Users/erklyavlin/repos/basilisk/src/qcc` (add to PATH if needed).

```bash
cd diploma/cfd_basilisk

# Compile and run (LEVEL=8 is production resolution)
qcc -O2 -Wall rising_bubble.c -o rising_bubble -lm
./rising_bubble > log.txt

# Different radius
qcc -O2 -Wall -DRADIUS=0.20 rising_bubble.c -o rising_bubble -lm

# Convert snapshots → HDF5
python convert_snapshots_to_h5.py --input-dir . \
  --output ../cfd_data/rising_bubble_basilisk_r020.h5
```

HDF5 files store: `X`, `Y`, `time`, `levelset`, `pressure`, `velocityX`, `velocityY`.  
Radius is stored as an attribute and also encoded in filename as `_R020` → `R=0.20`.

## Architecture summary

### `pinn/` package

| File | Responsibility |
|---|---|
| `config.py` | `PhysicsConfig`, `TrainingConfig`, `PointConfig`; `preset_config()` factory |
| `model.py` | `MLP` backbone + 4 separate output heads; `TwoPhasePINN` wraps MLP, implements `pde_residuals()` and `loss()` via autograd |
| `data.py` | Samples training points from HDF5 CFD files; interface-aware sampling (normal/tangent directions); `make_training_data()` is the main entry point |
| `train.py` | `TrainingRun` class: multi-stage Adam training, checkpoint save/resume, `train_rising_bubble.py` CLI |
| `evaluate.py` | Grid prediction, L2 metrics |
| `visualize.py` | `predict_cfd_grid()`, `plot_field_comparison()`, `align_pressure_gauge()` |

### Model inputs/outputs

- **Parametric PINN** (`input_dim=4`): `(x*, y*, t*, R*)` → `(u*, v*, p*, α)`
- **Simple PINN** (`input_dim=3`): `(x*, y*, t*)` → `(u*, v*, p*, α)`

All coordinates are non-dimensionalized by `l_ref=0.25`. Radius is `R/l_ref` (so physical R=0.30 → `R*=1.20`). Pressure head uses `exp(clip(...))` for positivity; alpha head uses `sigmoid` for [0,1] range.

### Training point types

`TrainingData` has 5 tensors, each with appended targets/labels as trailing columns:
- `alpha`: interface + domain points with CFD α values
- `pde`: collocation points for Navier-Stokes + continuity + α-transport residuals (target = 0)
- `north`: top boundary points (pressure target = `p_inf* = 1`)
- `east_west`: paired E/W boundary points for periodicity
- `nsew`: all-wall no-slip velocity points

PDE loss weights: `(λ_mass, λ_mom_x, λ_mom_y, λ_α) = (1, 10, 10, 1)`.

### Checkpoint format

Each run saves to a timestamped subdirectory: `best.pt`, `last.pt`, `config.json`, `history.json`.  
Checkpoint dict keys: `model_state`, `optimizer_state`, `epoch`, `stage`, `loss`, `history`, `config`.

## Key experimental checkpoints

All under `diploma/checkpoints_param/`:

| Directory | Description |
|---|---|
| `hard_2stage/` | Best model: 8×350 tanh, 2 stages (10k epochs) |
| `medium_2stage/` | 6×256 tanh |
| `light_2stage/` | 4×128 tanh |
| `hard_4stage/` | 8×350 tanh, 4 stages (20k epochs) |
| `hard_2stage_alpha_plus/` | Baseline + doubled α-points |
| `hard_2stage_pde_plus/` | Baseline + doubled PDE-points |
| `hard_2stage_alpha_pde_plus/` | Baseline + both doubled |

Simple PINN: `diploma/checkpoints/simple_pinn/` (trained on R=0.25 only).

Training data: `diploma/exp/train/` (R ∈ {0.18, 0.22, 0.25, 0.28, 0.32})  
Test data: `diploma/exp/test/` (R ∈ {0.20, 0.30})

## Thesis structure (main.tex)

The thesis must follow this section structure exactly. Do not add, remove, or reorder top-level sections or subsections without explicit instruction.

```latex
\section{Глава 1. Литературный обзор}
  \subsection{Проблематика}
  \subsection{Основные уравнения многофазной гидродинамики}
  \subsection{Численные методы CFD}

\section{Глава 2. Теоретическое описание}
  \subsection{Постановка задачи}          % includes the mathematical model (PDE system, dimensionless form)
  \subsection{Архитектура простой PINN}
  \subsection{Архитектура параметрической PINN}

\section{Глава 3. Практика}
  \subsection{Моделирование с использованием простой PINN}
  \subsection{Моделирование с использованием параметрической PINN}
    % subsubsections per experiment: light, medium, hard, stages, ablation, comparison

\section{Глава 4. Обсуждение результатов}

\section{Заключение}
```

Rules for Chapter 3:
- Each architecture / experiment gets its own `\subsubsection` with its own setup + results.
- Pipeline context (data, metric definition) is stated once at the top of the chapter, then each subsubsection references what differs — do not copy-paste identical paragraphs.
- The R=0.20 figure (`figures/param_hard/hard_2stage_R0.80_t3.000.png`) and training history figure (`figures/training_history_all.png`) must be included.

## Thesis (main.tex)

Compiled from repo root with `pdflatex`. Figures referenced as `figures/simple_pinn/...` and `figures/param_hard/...` — paths are relative to the compilation directory. All `\includegraphics` paths work when compiling from `/Users/erklyavlin/repos/diploma/`.

Available but not yet included in thesis:
- `figures/param_hard/hard_2stage_R0.80_t3.000.png` — parametric PINN on R=0.20
- `figures/training_history_all.png` — training curves for all parametric models
- `figures/r020_timeline_puv.png` — temporal evolution plot
