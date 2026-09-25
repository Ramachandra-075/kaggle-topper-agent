from datetime import datetime, timedelta, timezone

from kaggle_agent.discovery import rank_candidates


def test_prefers_fewer_teams():
    cfg = {
        "competition": {
            "max_teams": 150,
            "min_teams": 1,
            "min_days_left": 1,
            "max_days_left": 90,
            "target_days_left": 28,
            "exclude_refs": [],
        }
    }
    deadline = (datetime.now(timezone.utc) + timedelta(days=28)).isoformat()
    rows = [
        {"ref": "a", "teamCount": 10, "deadline": deadline, "category": "playground"},
        {"ref": "b", "teamCount": 100, "deadline": deadline, "category": "playground"},
    ]
    out = rank_candidates(rows, cfg)
    assert out[0].ref == "a"
