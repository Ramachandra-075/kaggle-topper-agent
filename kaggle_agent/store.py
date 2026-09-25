from __future__ import annotations
import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone

class ExperimentStore:
    def __init__(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        self.conn=sqlite3.connect(path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS experiments(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, competition TEXT, model TEXT,
        cv_score REAL, metric TEXT, higher_is_better INTEGER, public_score REAL,
        submission_ref INTEGER, params_json TEXT, submission_file TEXT)"""); self.conn.commit()
    def add(self,competition,model,cv_score,metric,higher_is_better,params,submission_file=""):
        cur=self.conn.execute("INSERT INTO experiments(created_at,competition,model,cv_score,metric,higher_is_better,params_json,submission_file) VALUES(?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(),competition,model,cv_score,metric,int(higher_is_better),json.dumps(params,sort_keys=True),submission_file)); self.conn.commit(); return int(cur.lastrowid)
    def update_submission(self,experiment_id,score,submission_ref):
        self.conn.execute("UPDATE experiments SET public_score=?,submission_ref=? WHERE id=?",(score,submission_ref,experiment_id)); self.conn.commit()
