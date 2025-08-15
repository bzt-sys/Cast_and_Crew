# Casting Resonance Case Study — 2‑Page Brief

## 1) Problem
Can public data provide a useful *planning signal* for casting and portfolio risk? We propose a time-aware recommender that scores film–talent fit and assembles candidate casts with strong expected audience resonance.

## 2) Data
- **Netflix Prize (1998–2005):** user→movie ratings; yields audience preference signals.
- **Box Office + Cast (1998–2005):** budget + domestic/international/worldwide gross; top-billed cast; director/writer/crew; genres; MPAA; runtime.

## 3) Method
- **Target (“Resonance”):** 0.5·z(log(1+worldwide)_by_year) + 0.5·scaled(avg_rating).
- **Features (compact, actuarial):**
  1. Actor Trailing Resonance (3-film, smoothed)
  2. Director Trailing Resonance (3-film, smoothed)
  3. Genre Affinity + market momentum (2y trailing)
  4. Budget Fit Delta (project vs. actor budget decile)
  5. Collaboration Memory (actor–director history + performance)
  6. Release Velocity (roles/yr vs. recency)
  7. MPAA Fit; 8. Top-Billing Propensity; 9. Co-cast Prior (mean ATR of attached cast)

- **Models:** Logistic baseline → Gradient Boosting (temporal split).
- **Composer:** Greedy selection maximizing expected resonance with budget + size constraints; retrieve “unsung similar” alternates via nearest neighbors.

## 4) Results (to be filled by notebook)
- Classification AUC / Regression RMSE
- NDCG@10 for cast ranking
- Feature importance chart
- Top-N recommendations for 2 prototype film profiles (e.g., Action ’03, Drama ’04)
- Unsung alternates for each lead

## 5) How We’d Extend Internally
- Add **proprietary talent profiles** (auditions, availability, cost, qualitative notes) → dual-engine scoring.
- Upgrade to **temporal graph embeddings** for talent and co-cast synergy.
- Formalize set selection as submodular/MIP with real constraints.
- Human-in-the-loop review + fairness checks.

## 6) Responsible Use
Public performance signals only; no personal attributes. Use as a signal alongside expert judgment.

