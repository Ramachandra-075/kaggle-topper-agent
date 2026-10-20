"""Conservative validation before any Kaggle submission.

The guard intentionally fails closed: uncertain competition metadata or malformed
prediction files must block submission rather than be guessed around.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SubmissionCheck:
    ok: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deadline_is_future(value: str | None, now: datetime | None = None) -> bool:
    if not value:
        return False
    try:
        deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return deadline > current
    except (TypeError, ValueError):
        return False


def validate_submission(
    submission_path: str | Path,
    *,
    expected_rows: int,
    expected_columns: list[str],
    deadline: str | None,
    rules_fetched: bool,
    metric: str | None,
    approved: bool,
    enabled: bool,
    now: datetime | None = None,
) -> SubmissionCheck:
    """Validate file structure and authorization; no network calls or submission."""
    errors: list[str] = []
    warnings: list[str] = []
    path = Path(submission_path)

    if not enabled:
        errors.append("Automatic submission is disabled.")
    if not approved:
        errors.append("Competition is not explicitly approved for automatic submission.")
    if not rules_fetched:
        errors.append("Competition rules were not successfully fetched; human review is required.")
    if not metric:
        errors.append("Evaluation metric was not confidently detected.")
    if not _deadline_is_future(deadline, now):
        errors.append("Competition deadline is missing, invalid, or has passed.")
    if not path.is_file():
        errors.append("Submission file does not exist.")
        return SubmissionCheck(False, errors, warnings)

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"Submission CSV could not be read: {type(exc).__name__}.")
        return SubmissionCheck(False, errors, warnings)

    if frame.empty:
        errors.append("Submission CSV contains no prediction rows.")
    if len(frame) != expected_rows:
        errors.append(f"Row count mismatch: expected {expected_rows}, found {len(frame)}.")
    if list(frame.columns) != expected_columns:
        errors.append("Submission columns or column order do not match the expected schema.")
    if frame.columns.duplicated().any():
        errors.append("Submission contains duplicate column names.")
    if frame.isna().any().any():
        errors.append("Submission contains missing values.")
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        # IDs may be strings, so only enforce finite values when the entire
        # column is numeric or can be converted without introducing NaNs.
        if numeric.notna().all():
            values = numeric.to_numpy(dtype=float)
            if not all(math.isfinite(float(value)) for value in values):
                errors.append(f"Column {column!r} contains infinite or non-finite values.")
    return SubmissionCheck(not errors, errors, warnings)
