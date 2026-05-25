# HANDOFF: Diploma Thesis — PINN for Rising Bubble Dynamics

## Context

This is a diploma thesis project by Emil Klyavlin (emilkliavlin2@gmail.com).
**Topic:** Simulating bubble rise dynamics using Physics-Informed Neural Networks (PINN).
**Working directory:** `/Users/erklyavlin/repos/diploma`

The thesis (`main.tex`) is the primary artifact. It is written in Russian and compiled with `pdflatex` (not installed locally — use Overleaf or a TeX Live machine).

A `CLAUDE.md` was created at the repo root this session — load it for commands, architecture, and checkpoint details.

---

## What has been done (this session)

### Restructuring of main.tex (completed)

The document was restructured from its previous form into the following canonical structure (now locked in `CLAUDE.md` as a project rule):

```
\section{Глава 1. Литературный обзор}
  \subsection{Проблематика}
  \subsection{Основные уравнения многофазной гидродинамики}
  \subsection{Численные методы CFD}

\section{Глава 2. Теоретическое описание}
  \subsection{Постановка задачи}          ← merged with old "Математическая модель"
  \subsection{Архитектура простой PINN}   ← renamed (was "...без параметра")
  \subsection{Архитектура параметрической PINN}  ← renamed (was "...с радиусом")

\section{Глава 3. Практика}
  \subsection{Моделирование с использованием простой PINN}
    \subsubsection{Настройка эксперимента}
    \subsubsection{Результаты на обучающем радиусе}
    \subsubsection{Результаты на новом радиусе}
    \subsubsection{Вывод по простой PINN}
  \subsection{Моделирование с использованием параметрической PINN}
    \subsubsection{Архитектура light}
    \subsubsection{Архитектура medium}
    \subsubsection{Архитектура hard}
    \subsubsection{Итоговое сравнение архитектур}
    \subsubsection{Влияние числа стадий обучения}
    \subsubsection{Аблация числа обучающих точек}
    \subsubsection{Итоговое сравнение с простой PINN}

\section{Глава 4. Обсуждение результатов}
  \subsection{Физическая интерпретация ошибок}
  \subsection{Качество параметрической интерполяции}
  \subsection{Роль числа стадий обучения и размера обучающей выборки}
  \subsection{Сравнение с работой-прототипом}
  \subsection{Ограничения метода и направления дальнейшего развития}

\section{Заключение}
```

**What changed vs. the old version:**
- `\subsection{Математическая модель двухфазного течения}` heading removed; content merged into `Постановка задачи`
- `\subsection{Подготовка CFD-данных и метрика качества}` heading removed; content stays as intro paragraphs to Ch3
- `\subsection{Подбор архитектуры параметрической PINN}` replaced with per-architecture subsubsections (light / medium / hard), each with its own setup + results table
- `\subsection{Влияние числа стадий обучения}` and `\subsection{Аблация}` demoted to `\subsubsection` under the parametric subsection
- `\subsection{Сравнение простой и параметрической PINN}` → `\subsubsection{Итоговое сравнение с простой PINN}`
- `\subsection{Выводы по главе}` removed entirely (Ch4 covers all of it)

### New figures added (completed)

Two figures that existed on disk but were not in the thesis are now included:

| Figure file | Where added | LaTeX label |
|---|---|---|
| `figures/param_hard/hard_2stage_R0.80_t3.000.png` | `\subsubsection{Архитектура hard}` | `fig:param_r020` |
| `figures/training_history_all.png` | `\subsubsection{Влияние числа стадий обучения}` | `fig:training_history` |

Note: the old label `fig:param_r030` (for the R=0.30 parametric figure) was renamed to `fig:param_r030_hard` and moved into `\subsubsection{Архитектура hard}`. No other section referenced this label by name so no broken refs.

### Training times added (completed)

`time.txt` files exist in every checkpoint directory with wall-clock training times. All are now referenced in the thesis:

| Model | Time | Where in thesis |
|---|---|---|
| simple_pinn | 2 ч 26 мин | `\subsubsection{Настройка эксперимента}` (simple PINN) |
| light_2stage | 1 ч 36 мин | `\subsubsection{Архитектура light}` |
| medium_2stage | 3 ч 56 мин | `\subsubsection{Архитектура medium}` |
| hard_2stage | 9 ч 40 мин | `\subsubsection{Архитектура hard}` + stages section |
| hard_4stage | 19 ч 17 мин | `\subsubsection{Влияние числа стадий обучения}` |
| hard_2stage_alpha_plus | 17 ч 24 мин | `tab:ablation_time` |
| hard_2stage_pde_plus | 17 ч 30 мин | `tab:ablation_time` |
| hard_2stage_alpha_pde_plus | 18 ч 40 мин | `tab:ablation_time` |

### Future work section (completed)

Added temperature as a future parameter direction to `\subsection{Ограничения метода и направления дальнейшего развития}` — motivated by its effect on viscosity, surface tension, and phase transitions (boiling).

### CLAUDE.md created (completed)

Full project guide at `/Users/erklyavlin/repos/diploma/CLAUDE.md` covering Python commands, training, evaluation, CFD data generation, package architecture, checkpoint inventory, and thesis structure rules.

---

## Current state of main.tex

- **1671 lines** (was 1586 before this session)
- **No `\todo{}`** calls in body (only the macro definition in preamble at line 18)
- All figures referenced in the text exist on disk

---

## What still needs to be done

### 1. High priority: missing structural elements

The thesis currently starts with `\maketitle` and immediately jumps to `\section{Глава 1}`. Three standard diploma elements are missing:

**a) `\tableofcontents`** — add right after `\maketitle`:
```latex
\maketitle
\tableofcontents
\newpage
```

**b) Аннотация (Abstract)** — add before or after `\tableofcontents`. Typical format:
```latex
\begin{abstract}
...
\end{abstract}
```
or as `\section*{Аннотация}` if the university requires a specific heading.

**c) Введение (Introduction)** — add as `\section*{Введение}` before Chapter 1. Should contain:
- Актуальность темы (~1 paragraph, already covered in Ch1.1 "Проблематика" — can summarize)
- Цели и задачи работы (bullet list — mandatory for most Russian diplomas)
- Структура работы (one sentence per chapter)

The "Проблематика" subsection in Ch1 already has good motivation text; the Введение should be shorter and more formal, with explicit цели/задачи.

### 2. Medium priority: volume additions

The user explicitly asked for more volume earlier in the session. What was discussed but not yet implemented:

**a) PINN-specific subsection in Ch1** (~1.5 pages, not done)  
Currently PINN is introduced only in the last 2 paragraphs of `\subsection{Проблематика}`. A dedicated `\subsection{Физико-информированные нейронные сети}` was proposed covering:
- The Raissi et al. 2019 paper (already in bibliography as `\cite{Raissi2019}`)
- PINN applications to two-phase flows
- Key advantages (embedding physics in loss, mesh-free, inverse problems)
- Limitations (training cost, sensitivity to hyperparameters)
- Buhendwa 2021 as the direct predecessor (`\cite{Buhendwa2021}`)

Place it as the 4th subsection of Ch1, after "Численные методы CFD".

**b) Basilisk / CFD data generation details in Ch3** (~0.5 pages, not done)  
The intro of Ch3 currently just says "данные получены с помощью Basilisk". Should describe:
- Grid resolution (adaptive, up to level 8 = 256×512 effective cells)
- Domain: x ∈ [−0.5, 0.5], y ∈ [0, 2], t ∈ [0, 3]
- Number of snapshots available per simulation
- Which snapshots are selected for training (see `selected_time_indices()` in `diploma/pinn/data.py`: indices [0:30:15] + [30:100:5] + [100::5] + [1:3])
- Time to run one simulation in Basilisk (not documented, but could be estimated from `diploma/cfd_basilisk/README.md`)
- HDF5 fields stored: X, Y, time, levelset, pressure, velocityX, velocityY

**c) Dimensionless numbers** (~0.5 pages, not done)  
Standard for a fluid dynamics thesis. Compute Re, Eo (Eötvös/Bond), Mo (Morton) for the reference case R=0.25:
- ρ₁=100, ρ₂=1000, μ₁=1, μ₂=10, σ=24.5, g=0.98, R=0.25
- Eo = (ρ₂−ρ₁)gD²/σ = 900×0.98×0.25/24.5 ≈ 2.25 (using D=2R=0.5... check formula)
- Mo = gμ₂⁴(ρ₂−ρ₁)/ρ₂²σ³

Add to `\subsection{Постановка задачи}` after the physical parameters table.

### 3. Low priority: polish

- **Page layout check**: compile and check that 4-panel figures don't overflow. `\includegraphics[width=\textwidth]` should be fine but needs visual verification.
- **Table column widths**: `tab:ablation` has 9 columns and may be tight on A4. Consider `\small` or splitting into two rows.
- **Bibliography**: only 9 entries. Could add 1–2 more on parametric PINN or neural operator methods if more citations are needed.

---

## Key facts to remember (don't re-derive)

- **Pressure gauge alignment**: all L2 errors for `p` are computed after aligning the additive constant by matching mean pressure at the top boundary. Implemented in `diploma/pinn/visualize.py::align_pressure_gauge()`. Without this, pressure errors appear ~68%.
- **Radius in figure filenames is dimensionless**: `R0.80` = physical R=0.20 / l_ref=0.25. `R1.20` = physical R=0.30. Captions use physical radius — don't confuse.
- **Training data**: `diploma/exp/train/` (R ∈ {0.18, 0.22, 0.25, 0.28, 0.32}). **Test data**: `diploma/exp/test/` (R ∈ {0.20, 0.30}).
- **Best model**: `checkpoints_param/hard_2stage/best.pt` — 8×350 tanh, 2 stages, 10k epochs.
- **Figures on disk not yet in thesis**: `figures/r020_timeline_puv.png`, `figures/r020_comparison_t1p50004.png`, `figures/r030_comparison_t1p50004.png` — could be used for showing temporal evolution at intermediate times.
