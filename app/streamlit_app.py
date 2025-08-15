# app/streamlit_app.py
import os, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Film Slate Recommender", layout="wide")

# --------- Load artifacts ----------
@st.cache_resource
def load_artifacts(root: str):
    root = os.path.abspath(root)
    # Required
    with open(os.path.join(root, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(root, "stable_feats.json")) as f:
        stable_feats = json.load(f)
    with open(os.path.join(root, "genre_cols.json")) as f:
        genre_cols = json.load(f)
    with open(os.path.join(root, "meta.json")) as f:
        meta = json.load(f)

    model = joblib.load(os.path.join(root, "model.pkl"))
    actor_feat = pd.read_parquet(os.path.join(root, "actor_feat.parquet")).set_index("actor_name")

    # Optional
    pair_synergy = None
    sy_path = os.path.join(root, "pair_synergy.csv")
    if os.path.exists(sy_path):
        try:
            pair_synergy = pd.read_csv(sy_path)
        except Exception:
            pair_synergy = None

    return model, feature_cols, stable_feats, genre_cols, meta, actor_feat, pair_synergy

# --------- UI: sidebar ----------
st.sidebar.header("Artifacts")
art_dir = st.sidebar.text_input("Artifacts folder", value="../Cast_and_Crew/artifacts")
if not art_dir:
    st.stop()

try:
    model, feature_cols, stable_feats, genre_cols, meta, actor_feat, pair_synergy = load_artifacts(art_dir)
except Exception as e:
    st.error(f"Could not load artifacts from {art_dir}: {e}")
    st.stop()

st.sidebar.header("Brief")
year = st.sidebar.slider("Year", int(meta["year_min"]), int(meta["year_max"]),
                         value=int(min(meta["year_max"], 2006)))
budget = st.sidebar.number_input("Budget ($)", min_value=1_000, value=20_000_000, step=1_000_000, format="%i")
genre_labels = [g.replace("genre_", "").replace("_aff", "").title() for g in genre_cols]
default_genres = [g for g in ["Action","Thriller"] if g in [x.title() for x in genre_labels]]
genres = st.sidebar.multiselect("Genres", genre_labels, default=default_genres)
bill_order = st.sidebar.selectbox("Bill Order (role)", [1,2,3,4], index=0)

st.sidebar.header("Options")
top_n = st.sidebar.slider("Top N", 5, 50, 15)
use_synergy = st.sidebar.checkbox("Apply actor–director synergy", value=True)
director_name = st.sidebar.text_input("Director for synergy (optional)", value="")
synergy_weight = st.sidebar.slider("Synergy weight", 0.0, 0.5, 0.25, 0.05)

st.title("Film Slate Recommender")
st.caption("Historical, leakage-safe ranking with explainability-friendly outputs. Uses your exported artifacts.")

# --------- Helpers ----------
def year_sin_cos(y, y0, y1):
    span = (y1 - y0) + 1.0
    return np.sin(2*np.pi*(y - y0)/span), np.cos(2*np.pi*(y - y0)/span)

def film_vector(year: int, budget: float, genres_in: list[str], y0: int, y1: int, bill_order: int):
    s, c = year_sin_cos(year, y0, y1)
    # map pretty labels back to training columns
    gflags = {col: 0 for col in genre_cols}
    base = {col.replace("genre_","").replace("_aff","").lower(): col for col in genre_cols}
    for g in genres_in:
        k = g.strip().lower()
        if k in base:
            gflags[base[k]] = 1
    fv = {
        "year_sin": s,
        "year_cos": c,
        "budget": float(budget),
        "bill_order": int(bill_order),
        # You may not have these film-side cols in artifacts; we add zeros later if missing
        "release_density": 0.0,
        "bfd": 0.0,
    }
    fv.update(gflags)
    return fv

def predict_proba_aligned(clf, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X2 = X.copy()
    # add any missing columns required by the model
    for col in feature_cols:
        if col not in X2.columns:
            X2[col] = 0.0
    # exact column order
    X2 = X2[feature_cols].astype(float)
    proba = clf.predict_proba(X2)[:, list(clf.classes_).index(1)]
    return proba

def apply_synergy(actor_slate: pd.DataFrame, director: str, pair_synergy: pd.DataFrame, weight: float = 0.25):
    if pair_synergy is None or not director:
        return actor_slate
    ps = pair_synergy[pair_synergy["director_name"] == director]
    if ps.empty:
        return actor_slate
    m = ps.set_index("actor_name")["phat"].to_dict()
    eps = 1e-6
    out = actor_slate.copy()
    p = out["p_high"].clip(eps,1-eps).values
    z = np.log(p/(1-p))
    t = out["actor_name"].map(m).astype(float)
    mask = t.notna().values
    if mask.any():
        z_t = np.log(np.clip(t[mask].values, eps, 1-eps) / np.clip(1 - np.array(t[mask].values), eps, 1-eps))
        z[mask] = (1.0 - weight)*z[mask] + weight*z_t
        out["p_high"] = 1/(1+np.exp(-z))
    return out

# --------- Score actors ----------
fv = film_vector(year, budget, genres, int(meta["year_min"]), int(meta["year_max"]), bill_order)
cand = actor_feat.copy()
for k, v in fv.items():
    cand[k] = v

proba = predict_proba_aligned(model, cand, feature_cols)
slate = cand.assign(p_high=proba).reset_index().rename(columns={"index": "actor_name"})
slate = slate.sort_values("p_high", ascending=False).reset_index(drop=True)

# --------- Optional synergy ----------
if use_synergy and director_name.strip():
    slate = apply_synergy(slate, director_name.strip(), pair_synergy, weight=synergy_weight)
    slate = slate.sort_values("p_high", ascending=False).reset_index(drop=True)

# --------- UI outputs ----------
st.subheader("Actor slate")
st.dataframe(slate.head(top_n))

csv_data = slate.head(top_n).to_csv(index=False).encode("utf-8")
st.download_button("Download actor slate (CSV)", data=csv_data, file_name="actor_slate.csv", mime="text/csv")

with st.expander("Details"):
    st.write("Feature contract length:", len(feature_cols))
    st.write("Stable actor features used:", stable_feats)
    st.write("Genres available:", genre_labels)
    st.json({"year_min": meta["year_min"], "year_max": meta["year_max"], "has_pair_synergy": pair_synergy is not None})
