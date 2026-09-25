from __future__ import annotations
import argparse, json
from .config import load_config
from .kaggle_client import KaggleClient
from .runner import AgentRunner

def doctor(_):
    c=KaggleClient(); rows=c.list_competitions(page_size=3); print(f"OK: Kaggle returned {len(rows)} competitions")
def discover(a):
    r=AgentRunner(load_config(a.config)).scout(a.limit)
    for i,c in enumerate(r,1): print(f"{i:>2} | teams={c.teams:<4} days={c.days_left:<6} entered={c.entered} | {c.ref}")
def scout(a):
    r=AgentRunner(load_config(a.config)); cs=r.scout(a.limit); r.write_scout_report(cs,a.out,a.json_out); print(f"Wrote {a.out}")
def prepare(a):
    print(json.dumps(AgentRunner(load_config(a.config)).prepare(a.competition),indent=2))
def train(a):
    for eid,r in AgentRunner(load_config(a.config)).run_suite(a.competition): print(f"exp={eid} model={r.model_name} {r.metric}={r.cv_score:.6f}")
def auto(a):
    print(json.dumps(AgentRunner(load_config(a.config)).auto_cycle(a.competition),indent=2))
def monitor(a):
    c=KaggleClient(); print("=== SUBMISSIONS ==="); print(c.submissions(a.competition)); print("=== LEADERBOARD ==="); print(c.leaderboard(a.competition))

def main():
    p=argparse.ArgumentParser(prog="kaggle-agent"); p.add_argument("--config",default="config/default.yaml"); s=p.add_subparsers(dest="command",required=True)
    x=s.add_parser("doctor"); x.set_defaults(func=doctor)
    x=s.add_parser("discover"); x.add_argument("--limit",type=int,default=15); x.set_defaults(func=discover)
    x=s.add_parser("scout"); x.add_argument("--limit",type=int,default=20); x.add_argument("--out",default="reports/scout.md"); x.add_argument("--json-out",default="reports/scout.json"); x.set_defaults(func=scout)
    for name,func in [("prepare",prepare),("train",train),("auto",auto),("monitor",monitor)]:
        x=s.add_parser(name); x.add_argument("competition"); x.set_defaults(func=func)
    a=p.parse_args(); a.func(a)
if __name__=="__main__": main()
