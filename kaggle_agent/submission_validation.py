from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_submission(
    submission_path: str | Path,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> list[str]:
    """Return blocking errors for a Kaggle submission; an empty list means valid."""
    errors: list[str] = []
    path = Path(submission_path)

    if not path.is_file():
        return [f"Submission file does not exist: {path}"]

    try:
        submission = pd.read_csv(path)
    except Exception as exc:
        return [f"Submission CSV could not be read: {type(exc).__name__}: {exc}"]

    expected_columns = list(sample_submission.columns)
    actual_columns = list(submission.columns)
    if actual_columns != expected_columns:
        errors.append(
            f"Column schema mismatch: expected {expected_columns}, got {actual_columns}"
        )

    if len(submission) != len(test):
        errors.append(
            f"Row count mismatch: expected {len(test)} test rows, got {len(submission)}"
        )

    if submission.columns.duplicated().any():
        errors.append("Submission contains duplicate column names")

    prediction_columns = [c for c in expected_columns if c not in test.columns]
    if not prediction_columns:
        prediction_columns = expected_columns[-1:] if expected_columns else []

    for column in prediction_columns:
        if column not in submission.columns:
            continue
        if submission[column].isna().any():
            errors.append(f"Prediction column {column!r} contains missing values")
            continue
        if pd.api.types.is_numeric_dtype(submission[column]):
            values = pd.to_numeric(submission[column], errors="coerce").to_numpy()
            if not np.isfinite(values).all():
                errors.append(f"Prediction column {column!r} contains NaN or infinity")

    # Verify shared identifier columns when the sample submission includes them.
    # Prediction columns are excluded even if a competition uses an unusual schema.
    for column in expected_columns:
        if column in test.columns and column not in prediction_columns:
            if column not in submission.columns:
                continue
            expected_ids = test[column].reset_index(drop=True)
            actual_ids = submission[column].reset_index(drop=True)
            if len(expected_ids) == len(actual_ids) and not expected_ids.equals(actual_ids):
                errors.append(f"Identifier alignment mismatch in column {column!r}")

    return errors
