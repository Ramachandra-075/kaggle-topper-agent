from __future__ import annotations
import re
import numpy as np
import pandas as pd

NUMERIC_HINTS = (
    "mileage","odometer","mile","year","engine","cylinder","price",
    "doors","seat","horsepower","hp","mpg","km","kilometer"
)

def _numeric_from_text(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.replace(",", "", regex=False)
    x = x.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(x, errors="coerce")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    current_year = 2026

    # Missingness is often predictive in scraped marketplace data.
    out["__missing_count"] = out.isna().sum(axis=1)
    out["__present_count"] = out.notna().sum(axis=1)

    # Convert numeric-looking text columns into parallel numeric features.
    for c in list(out.columns):
        if not pd.api.types.is_object_dtype(out[c]) and not pd.api.types.is_string_dtype(out[c]):
            continue
        s = out[c].astype("string")
        non_null = s.dropna()
        if non_null.empty:
            continue

        name = c.lower()
        parsed = _numeric_from_text(s)
        ratio = float(parsed.notna().mean())
        if ratio >= 0.60 or any(h in name for h in NUMERIC_HINTS):
            out[f"{c}__num"] = parsed

        # Long/raw text gets compact structural features rather than one giant category.
        median_len = float(non_null.str.len().median()) if len(non_null) else 0.0
        if median_len >= 35 or "feature" in name or "description" in name or "note" in name:
            out[f"{c}__char_len"] = s.str.len().fillna(0).astype(float)
            out[f"{c}__word_count"] = s.str.split().str.len().fillna(0).astype(float)
            out[f"{c}__digit_count"] = s.str.count(r"\d").fillna(0).astype(float)
            out[f"{c}__comma_count"] = s.str.count(",").fillna(0).astype(float)
            out[f"{c}__pipe_count"] = s.str.count(r"\|").fillna(0).astype(float)

    # Domain features for used-car data.
    year_cols = [c for c in out.columns if c.lower() in {"year","model_year","manufacture_year"}]
    for c in year_cols:
        y = pd.to_numeric(out[c], errors="coerce")
        if y.notna().any():
            out[f"{c}__vehicle_age"] = (current_year - y).clip(lower=0, upper=80)

    mileage_candidates = []
    for c in out.columns:
        lc = c.lower()
        if lc in {"mileage","odometer","miles","kilometers","kilometres"} or "odometer" in lc:
            if pd.api.types.is_numeric_dtype(out[c]):
                mileage_candidates.append(pd.to_numeric(out[c], errors="coerce"))
            elif f"{c}__num" in out.columns:
                mileage_candidates.append(out[f"{c}__num"])
    if mileage_candidates:
        m = pd.concat(mileage_candidates, axis=1)
        out["__mileage_merged"] = m.median(axis=1, skipna=True)
        if len(mileage_candidates) >= 2:
            out["__mileage_disagreement"] = m.max(axis=1, skipna=True) - m.min(axis=1, skipna=True)

    # Cross feature: miles per year, robustly clipped.
    age_cols = [c for c in out.columns if c.endswith("__vehicle_age")]
    if age_cols and "__mileage_merged" in out.columns:
        age = out[age_cols[0]].replace(0, 1)
        out["__miles_per_year"] = (out["__mileage_merged"] / age).clip(lower=0, upper=100000)

    # Normalize infinities created by odd scraped values.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
