# Cast & Crew Resonance Recommender

Built in 3 days as a rapid exploration of an actor/director slate recommender. I deliberately scoped this as a speed run to study how far a single developer can get with an LLM coding assistant. Expect rough edges and open TODOs; see Limitations & Next steps.
LLM collaboration statement

Parts of this repo were drafted with the help of a large language model acting as a code co-pilot. All prompts, design decisions, and final edits are mine, and I reviewed/modified generated code before inclusion. The goal of documenting this is transparency and to share what rapid, human-in-the-loop workflows look like in practice.

**One-line:** Given a project brief (year, budget, genres, bill order), produce a ranked **actor slate** with alternates and rationale — trained with strict **temporal hygiene**, calibrated for intuitive probability printouts, and optionally injected with **cycle prior** and **director synergy** signals.

---

## 1) Problem & Outcome

Studios repeatedly ask: _“Who should we cast for **this** film, **in this era**, at **this budget**?”_  
This project turns that problem into a **ranked slate** of actors plus:
- **Alternates** (“if not A, consider B/C/D”),
- **Reason codes** (top features driving each recommendation),
- **Manifest** (parameters, timestamps, flags) for reproducibility.

It’s designed as an **assistant** to reduce the search space, and eventually paired with a talent database that predicts similar audience reactions from lesser known persons, potentially cutting costs on budget, breaking new talent into the entertainment space, and 
engaging with audiences more successfully per project.
---

## 2) System Architecture

flowchart TB
  %% =============== INPUTS ===============
  subgraph Inputs
    BRIEF[[Project Brief\n(year, budget, genres, bill_order)]]
    DATA[(fa long table)]
  end

  %% =============== TRAINING ===============
  subgraph Training Pipeline
    A1[Leakage-safe labels\nTop quartile resonance\ncomputed on TRAIN years only]
    A2[Feature engineering\natr3, dtr3, release_density,\ngenre_*_aff, year_sin/cos,\nBUDGET_BIN_EDGES, bfd]
    A3[Temporal split\nby TRAIN_CUTOFF_YEAR]
    A4[Actor head model\n(HGB/RF)]
    A5[Director head model\n(HGB)]
    A6[Calibration fit\nPlatt on FEATURES (prefit),\nIsotonic on base probs]
    A7[Cycle head fit\nResidual r = y - p_base\n~ year_sin/cos]
  end

  %% =============== SERVING ===============
  subgraph Serving (Recommender)
    S1[Build candidate matrix (as-of year)\nactor means (stable_feats),\ntrailing aggregates, film features\n— enforce feature contract]
    S2[Score base probs (p_base)\nusing use_model]
    S3[Cycle blend to ranking score\np_rank = blend(p_base, uplift)]
    S4[Calibrated probs for display only\np_cal = isotonic(p_base) or Platt(X)]
    S5[Unsung alternates\nkNN on actor_feat (cosine)]
    S6[Reason codes\nperturb features → Δprob]
    S7[Optional synergy\nadjust actor p via chosen director]
  end

  %% =============== OUTPUTS ===============
  subgraph Outputs
    O1[[Actor slate CSV\nTop-N with p_high, p_base, p_cal]]
    O2[[Alternates CSV]]
    O3[[Reason-codes CSV + plots]]
    O4[[Manifest JSON\n(params, timestamps, flags)]]
  end

  %% Connections
  BRIEF --> S1
  DATA --> A1 --> A2 --> A3 --> A4
  A3 --> A5
  A3 --> A6
  A3 --> A7
  DATA --> S1
  A4 --> S2
  A7 --> S3
  S2 --> S3 --> S4
  S2 --> S5
  S2 --> S6
  S3 --> O1
  S5 --> O2
  S6 --> O3
  S4 --> O1
  S7 --> O1
  A5 --> S7

```

---

## 3) Data & Features

- **Long table `fa`**: one row per (movie, actor, role). Includes `movie_year`, `budget`, `bill_order`, `resonance`, genre flags, etc.
- **Key engineered features**
  - `atr3`, `dtr3`: trailing 3-role exposure (actor/director)
  - `release_density`: actor pace (rolling)
  - `genre_*_aff`: rolling share over prior roles (affinity)
  - `year_sin`, `year_cos`: cyclical time encoding
  - **Budget bins** from **train-era** only → `budget_decile`
  - `bfd`: | film_budget_decile - actor_budget_decile_trailing |

**Leakage control**: All thresholds (top quartile labels) and budget bins are computed on **TRAIN years only** (<= `TRAIN_CUTOFF_YEAR`), in order to assure temporal fidelity within the larger patterns throughout the datasets.

---

## 4) Training & Evaluation

- **Temporal split** by `TRAIN_CUTOFF_YEAR` (e.g. 2003).
- **Actor head**: HistGradientBoosting or RandomForest (choose by holdout AUROC).
- **Director head**: HistGradientBoosting.
- **Calibration (display only)**:
  - **Platt (sigmoid)** with `CalibratedClassifierCV(..., cv='prefit')` on **features**, temporal calibration slice.
  - **Isotonic** mapping on **base probs** (`p_base` → `p_cal`) with collapse guard.
- **Cycle head**: Linear regression on residuals `r = y - p_base` vs. `year_sin/cos` (train-only).

**Metrics surfaced**: AUROC, PR-AUC, reliability (Brier/ECE), per-year curves, horizon plots.

---

## 5) Inference (Serving) Workflow

1. **Brief in**: (year, budget, genres, bill_order).
2. **Candidate builder (as-of)**:
   - Actor means of stable_feats up to the year; trailing aggregates;
   - Film-side features (budget, time, bill order);
   - **Enforce feature contract** (add missing cols, correct order).
3. **Score base probs** (`p_base`) using the chosen actor model.
4. **Cycle blend** → **ranking score** (`p_rank` = `p_high` in outputs).
5. **Calibration**: Only for display (`p_cal` = isotonic(p_base) or Platt(X)).
6. **Alternates**: kNN in cosine space over actor-level feature means.
7. **Reason codes**: per-row perturbation deltas.
8. **Optional synergy**: if a director slate is chosen, bump compatible actors.
9. **Export**: slate/alternates/reason-codes CSV + manifest JSON + plots.

---

## 6) Outputs & Artifacts

- `slate_*.csv` — columns: `actor_name`, `p_base`, `p_cal`, `p_high` (ranking)
- `*_alternates.csv` — unsung alternates for top recommendations
- `*_reason_codes.csv` — leading feature deltas
- `*_manifest.json` — brief, flags (`use_cycle`, `calibrate`, etc.), timestamps
- Key plots — reliability, PR curves, per-year PR-AUC, permutation importance

---

## 7) Streamlit App 

A simple app lets you set year, budget, genres, bill order; it calls the same inference path and displays the slate and alternates.  
Tip: use integer inputs for year/bill order; budget as float in **millions** for a clean UX.

---

## 8) Repro & Quickstart

1. **Environment**  
   ```bash
   pip install -r requirements.txt  # or conda env create -f environment.yml
   ```
2. **Run the notebook**  
   Execute `02_model_and_recs.ipynb` top-to-bottom.
3. **Generate demo artifacts**  
   Use the “Demo” cell (`run_demo_plus(...)`) — artifacts appear under `Cast_and_Crew/reports/demo_YYYYMMDD_HHMMSS/`.
4. **Launch app**  
   ```bash
   streamlit run app_streamlit.py
   ```

---

## 9) Configuration

- `TRAIN_CUTOFF_YEAR`: temporal boundary for labels/splits and calibration slice selection.
- `ALPHA_CYCLE`: blend weight for cycle prior (default 0.7).
- `synergy_weight`: actor–director synergy bump (e.g., 0.25).
- Feature contract: the model uses `feature_cols` fixed at train-time.

---

## 10) Limitations & Next Steps

- **Coverage & recency**: farther-from-train eras/genres are noisier; add fresh data pipelines.
- **Real-world constraints**: availability, contracts, release windows; add filters and live metadata.
- **Synergy**: convert heuristic weight into a cross-validated parameter by brief type.
- **Monitoring**: ECE drift, per-year PR-AUC drift, threshold stability alarms.
- **Packaging**: small library/module + unit tests for the feature contract, calibration, and no-peek guards.

---

## 11) Attribution & Intended Use

This is a case study to demonstrate **leakage-safe temporal modeling**, **calibration for comms**, and **product-minded recommender design**. It’s a decision aid, not a standalone greenlight system.
