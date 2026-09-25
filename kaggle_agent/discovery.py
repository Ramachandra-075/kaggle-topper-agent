from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

@dataclass
class Candidate:
    ref: str
    deadline: str
    category: str
    reward: str
    teams: int
    entered: bool
    user_rank: int
    days_left: float
    score: float
    def to_dict(self): return asdict(self)

def _days_left(deadline: str) -> float:
    if not deadline: return 9999
    try:
        dt = datetime.fromisoformat(deadline.replace("Z","+00:00"))
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return (dt-datetime.now(timezone.utc)).total_seconds()/86400
    except Exception:
        return 9999

def rank_candidates(rows: list[dict], cfg: dict) -> list[Candidate]:
    c = cfg["competition"]; out=[]; excluded=set(c.get("exclude_refs",[]))
    max_teams=int(c["max_teams"]); min_teams=int(c.get("min_teams",1)); target=float(c.get("target_days_left",28))
    for r in rows:
        ref=str(r.get("ref") or r.get("Ref") or "").strip()
        if not ref or ref in excluded: continue
        try: teams=int(float(r.get("teamCount") or r.get("TeamCount") or 0))
        except Exception: teams=0
        if not min_teams <= teams <= max_teams: continue
        deadline=str(r.get("deadline") or r.get("Deadline") or "")
        days=_days_left(deadline)
        if not float(c["min_days_left"]) <= days <= float(c["max_days_left"]): continue
        entered=str(r.get("userHasEntered") or r.get("UserHasEntered") or "false").lower() in {"true","1","yes"}
        try: user_rank=int(float(r.get("userRank") or r.get("UserRank") or 0))
        except Exception: user_rank=0
        team_component=max(0,(max_teams-teams)/max(1,max_teams))
        time_component=max(0,1-abs(days-target)/max(target,float(c["max_days_left"])))
        score=.78*team_component+.17*time_component+(.08 if entered else 0)
        out.append(Candidate(ref,deadline,str(r.get("category") or r.get("Category") or "unknown"),str(r.get("reward") or r.get("Reward") or ""),teams,entered,user_rank,round(days,2),round(score,6)))
    return sorted(out,key=lambda x:(-x.score,x.teams,x.deadline))
