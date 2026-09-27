from __future__ import annotations
import re
import numpy as np
import pandas as pd

NUMERIC_HINTS = (
    "mileage","odometer","mile","year","engine","cylinder","price",
    "doors","seat","horsepower","hp","mpg","km","kilometer"
)

TEXT_KEYWORDS = {
    "leather": r"\bleather\b",
    "sunroof": r"\bsun\s*roof\b|\bmoon\s*roof\b",
    "navigation": r"\bnavigation\b|\bgps\b",
    "bluetooth": r"\bbluetooth\b",
    "camera": r"\bbackup camera\b|\brear camera\b|\bcamera\b",
    "heated_seats": r"\bheated seats?\b",
    "cooled_seats": r"\bcooled seats?\b|\bventilated seats?\b",
    "awd": r"\bawd\b|all[- ]wheel drive",
    "4wd": r"\b4wd\b|four[- ]wheel drive",
    "cruise": r"\bcruise control\b",
    "carplay": r"\bcarplay\b",
    "android_auto": r"\bandroid auto\b",
    "premium_audio": r"\bbose\b|\bharman\b|\bpremium audio\b",
    "remote_start": r"\bremote start\b",
    "third_row": r"\bthird row\b|\b3rd row\b",
    "tow": r"\btow(ing)?\b|\btrailer\b",
}

CONDITION_MAP = {
    "new": 6.0,
    "like new": 5.0,
    "excellent": 4.0,
    "good": 3.0,
    "fair": 2.0,
    "salvage": 1.0,
}

ID_LIKE = {
    "id","listing_id","listingid","vin","url","link","listing_url",
    "dealer_url","image_url","stock_number","stock_no"
}


def _numeric_from_text(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.replace(",", "", regex=False)
    x = x.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(x, errors="coerce")


def _clean_category(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    current_year = 2026

    # IDs generally do not generalize because train/test are sequentially split.
    for c in list(out.columns):
        lc = c.lower().strip()
        if lc in ID_LIKE or lc.endswith("_id"):
            out = out.drop(columns=[c])

    # Drop columns with no information at all.
    all_missing = [c for c in out.columns if out[c].isna().all()]
    if all_missing:
        out = out.drop(columns=all_missing)

    out["__missing_count"] = out.isna().sum(axis=1)
    out["__present_count"] = out.notna().sum(axis=1)

    raw_text_to_drop = []

    for c in list(out.columns):
        if not (
            pd.api.types.is_object_dtype(out[c])
            or pd.api.types.is_string_dtype(out[c])
            or str(out[c].dtype).startswith("category")
        ):
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

        cleaned = _clean_category(s)
        nunique = int(cleaned.nunique(dropna=True))

        # Frequency encoding is target-free and useful for tree models.
        if 1 < nunique <= max(300, int(len(out) * 0.40)):
            counts = cleaned.value_counts(dropna=False)
            out[f"{c}__freq"] = cleaned.map(counts).fillna(0).astype(float) / max(len(out), 1)

        median_len = float(non_null.str.len().median()) if len(non_null) else 0.0
        is_long_text = (
            median_len >= 35
            or "feature" in name
            or "description" in name
            or "note" in name
            or "comment" in name
        )
        if is_long_text:
            lower = s.str.lower()
            out[f"{c}__char_len"] = s.str.len().fillna(0).astype(float)
            out[f"{c}__word_count"] = s.str.split().str.len().fillna(0).astype(float)
            out[f"{c}__digit_count"] = s.str.count(r"\d").fillna(0).astype(float)
            out[f"{c}__comma_count"] = s.str.count(",").fillna(0).astype(float)
            out[f"{c}__pipe_count"] = s.str.count(r"\|").fillna(0).astype(float)
            out[f"{c}__semicolon_count"] = s.str.count(";").fillna(0).astype(float)
            for key, pat in TEXT_KEYWORDS.items():
                out[f"{c}__has_{key}"] = lower.str.contains(pat, regex=True, na=False).astype(float)

            # Huge free-text categories hurt ordinal/tree encoders; retain the
            # engineered signals and remove the raw text itself.
            raw_text_to_drop.append(c)

    if raw_text_to_drop:
        out = out.drop(columns=list(dict.fromkeys(raw_text_to_drop)), errors="ignore")

    # Vehicle age and year buckets.
    year_cols = [c for c in out.columns if c.lower() in {"year","model_year","manufacture_year"}]
    for c in year_cols:
        y = pd.to_numeric(out[c], errors="coerce")
        if y.notna().any():
            age = (current_year - y).clip(lower=0, upper=80)
            out[f"{c}__vehicle_age"] = age
            out[f"{c}__decade"] = (y // 10) * 10
            out[f"{c}__age_sq"] = age ** 2

    # Condition has a natural ordering.
    for c in list(out.columns):
        if c.lower() == "condition":
            cond = _clean_category(out[c])
            out[f"{c}__score"] = cond.map(CONDITION_MAP).astype(float)

    # Merge redundant mileage / odometer fields.
    mileage_candidates = []
    for c in out.columns:
        lc = c.lower()
        if lc in {"mileage","odometer","miles","kilometers","kilometres"} or "odometer" in lc:
            if pd.api.types.is_numeric_dtype(out[c]):
                mileage_candidates.append(pd.to_numeric(out[c], errors="coerce").rename(c))
            elif f"{c}__num" in out.columns:
                mileage_candidates.append(out[f"{c}__num"].rename(c))

    if mileage_candidates:
        m = pd.concat(mileage_candidates, axis=1)
        out["__mileage_merged"] = m.median(axis=1, skipna=True)
        out["__mileage_log1p"] = np.log1p(out["__mileage_merged"].clip(lower=0))
        if len(mileage_candidates) >= 2:
            out["__mileage_disagreement"] = m.max(axis=1, skipna=True) - m.min(axis=1, skipna=True)
            out["__mileage_sources_present"] = m.notna().sum(axis=1).astype(float)

    age_cols = [c for c in out.columns if c.endswith("__vehicle_age")]
    if age_cols and "__mileage_merged" in out.columns:
        age = out[age_cols[0]].replace(0, 1)
        out["__miles_per_year"] = (out["__mileage_merged"] / age).clip(lower=0, upper=100000)
        out["__age_x_mileage"] = age * np.log1p(out["__mileage_merged"].clip(lower=0))

    # Manufacturer-model interaction usually carries a large amount of value.
    cols_lower = {c.lower(): c for c in out.columns}
    manufacturer = next((cols_lower[k] for k in ("manufacturer","make","brand") if k in cols_lower), None)
    model = cols_lower.get("model")
    if manufacturer and model:
        out["__brand_model"] = (
            _clean_category(out[manufacturer]).fillna("__missing__")
            + "__"
            + _clean_category(out[model]).fillna("__missing__")
        )
        counts = out["__brand_model"].value_counts(dropna=False)
        out["__brand_model_freq"] = out["__brand_model"].map(counts).astype(float) / max(len(out), 1)

    # Playground S6E9 EV-purchase features. Keep this gated by schema so
    # competition-specific tricks never leak into unrelated datasets.
    ev_cols = {col.lower(): col for col in out.columns}
    is_ev_purchase = (
        "subsidy_available" in ev_cols
        and any(k in ev_cols for k in ("income", "annual_income"))
        and any(k in ev_cols for k in ("range_anxiety_level", "range_anxiety"))
    )
    if is_ev_purchase:
        # Digit decomposition is unusually effective on synthetic Playground
        # Series tables because digit-position structure can survive generation.
        # Skip low-cardinality/binary columns where decomposition is redundant.
        numeric_cols = [
            col for col in list(out.columns)
            if pd.api.types.is_numeric_dtype(out[col])
            and out[col].nunique(dropna=True) > 20
            and not col.startswith("__")
            and "__" not in col
        ]
        for col in numeric_cols:
            values = pd.to_numeric(out[col], errors="coerce").abs()
            rounded = np.floor(values.fillna(0)).astype("int64")
            max_value = int(rounded.max()) if len(rounded) else 0
            max_digits = min(7, max(1, len(str(max_value))))
            for pos in range(max_digits):
                out[f"{col}__digit_{pos}"] = ((rounded // (10 ** pos)) % 10).astype(float)
            # Preserve a little decimal structure when the source is continuous.
            frac = (values - np.floor(values)).fillna(0)
            if float(frac.abs().max()) > 1e-9:
                out[f"{col}__decimal_digit_1"] = np.floor(frac * 10).astype(float)
                out[f"{col}__decimal_digit_2"] = np.floor(frac * 100).astype(float) % 10

        subsidy = ev_cols["subsidy_available"]
        income = next(ev_cols[k] for k in ("income", "annual_income") if k in ev_cols)
        income_num = pd.to_numeric(out[income], errors="coerce")
        subsidy_raw = out[subsidy]
        if pd.api.types.is_numeric_dtype(subsidy_raw):
            subsidy_num = pd.to_numeric(subsidy_raw, errors="coerce").fillna(0)
        else:
            subsidy_num = (
                _clean_category(subsidy_raw)
                .isin({"yes", "y", "true", "1", "available"})
                .astype(float)
            )
        out["__ev_subsidy_x_income"] = subsidy_num * income_num
        out["__ev_income_log1p"] = np.log1p(income_num.clip(lower=0))
        # Quantile-like fixed-width rank proxy without target information.
        out["__ev_income_10k_bin"] = np.floor(income_num / 10000.0)

    # Normalize posting-date information when present.
    for c in list(out.columns):
        lc = c.lower()
        if "date" in lc or "posted" in lc or "posting" in lc:
            if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
                dt = pd.to_datetime(out[c], errors="coerce", utc=True)
                if dt.notna().mean() >= 0.40:
                    out[f"{c}__year"] = dt.dt.year.astype(float)
                    out[f"{c}__month"] = dt.dt.month.astype(float)
                    out[f"{c}__dow"] = dt.dt.dayofweek.astype(float)
                    out[f"{c}__hour"] = dt.dt.hour.astype(float)
                    out = out.drop(columns=[c])

    out = out.replace([np.inf, -np.inf], np.nan)

    # Final cleanup: columns that became completely missing after parsing.
    all_missing = [c for c in out.columns if out[c].isna().all()]
    if all_missing:
        out = out.drop(columns=all_missing)

    return out
