from pathlib import Path

import pandas as pd

from kaggle_agent.submission_validation import validate_submission


def test_valid_submission_passes(tmp_path: Path):
    test = pd.DataFrame({"id": [10, 11, 12], "feature": [1, 2, 3]})
    sample = pd.DataFrame({"id": [10, 11, 12], "target": [0, 0, 0]})
    path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [10, 11, 12], "target": [0.2, 0.4, 0.8]}).to_csv(path, index=False)

    assert validate_submission(path, test, sample) == []


def test_rejects_wrong_row_count_and_schema(tmp_path: Path):
    test = pd.DataFrame({"id": [10, 11], "feature": [1, 2]})
    sample = pd.DataFrame({"id": [10, 11], "target": [0, 0]})
    path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [10], "prediction": [0.5]}).to_csv(path, index=False)

    errors = validate_submission(path, test, sample)
    assert any("schema mismatch" in error for error in errors)
    assert any("Row count mismatch" in error for error in errors)


def test_rejects_non_finite_predictions(tmp_path: Path):
    test = pd.DataFrame({"id": [1, 2], "feature": [4, 5]})
    sample = pd.DataFrame({"id": [1, 2], "target": [0, 0]})
    path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.2, float("inf")]}).to_csv(path, index=False)

    errors = validate_submission(path, test, sample)
    assert any("NaN or infinity" in error for error in errors)


def test_rejects_misaligned_ids(tmp_path: Path):
    test = pd.DataFrame({"id": [1, 2], "feature": [4, 5]})
    sample = pd.DataFrame({"id": [1, 2], "target": [0, 0]})
    path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [2, 1], "target": [0.2, 0.8]}).to_csv(path, index=False)

    errors = validate_submission(path, test, sample)
    assert any("Identifier alignment mismatch" in error for error in errors)
