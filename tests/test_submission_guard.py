from datetime import datetime, timezone

import pandas as pd

from kaggle_agent.submission_guard import validate_submission


def _valid_args(path):
    return dict(
        submission_path=path,
        expected_rows=3,
        expected_columns=["id", "prediction"],
        deadline="2027-01-01T00:00:00Z",
        rules_fetched=True,
        metric="roc_auc",
        approved=True,
        enabled=True,
        now=datetime(2026, 10, 20, tzinfo=timezone.utc),
    )


def test_valid_submission_passes(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "prediction": [0.1, 0.2, 0.9]}).to_csv(path, index=False)
    result = validate_submission(**_valid_args(path))
    assert result.ok
    assert result.errors == []


def test_auto_submission_disabled_blocks(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "prediction": [0.1, 0.2, 0.9]}).to_csv(path, index=False)
    args = _valid_args(path)
    args["enabled"] = False
    result = validate_submission(**args)
    assert not result.ok
    assert any("disabled" in error for error in result.errors)


def test_closed_competition_blocks(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "prediction": [0.1, 0.2, 0.9]}).to_csv(path, index=False)
    args = _valid_args(path)
    args["deadline"] = "2026-01-01T00:00:00Z"
    result = validate_submission(**args)
    assert not result.ok
    assert any("deadline" in error for error in result.errors)


def test_schema_and_row_mismatch_block(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "wrong": [0.1, 0.2]}).to_csv(path, index=False)
    result = validate_submission(**_valid_args(path))
    assert not result.ok
    assert any("Row count mismatch" in error for error in result.errors)
    assert any("columns" in error for error in result.errors)


def test_missing_rules_and_metric_block(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "prediction": [0.1, 0.2, 0.9]}).to_csv(path, index=False)
    args = _valid_args(path)
    args["rules_fetched"] = False
    args["metric"] = None
    result = validate_submission(**args)
    assert not result.ok
    assert any("rules" in error for error in result.errors)
    assert any("metric" in error for error in result.errors)
