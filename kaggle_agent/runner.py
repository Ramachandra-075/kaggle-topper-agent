from __future__ import annotations
import json, os
from pathlib import Path
from .discovery import rank_candidates
from .kaggle_client import KaggleClient
from .rules import inspect_rules
from .state import load_state, save_state
from .store import ExperimentStore
from .trainer import train_model, build_ensemble

class AgentRunner:
    def __init__(self,cfg):
        self.cfg=cfg; self.workspace=Path(cfg["paths"]["workspace"]); self.state_dir=Path(cfg["paths"].get("state","state")); self.report_dir=Path(cfg["paths"].get("reports","reports"))
        for p in (self.workspace,self.state_dir,self.report_dir): p.mkdir(parents=True,exist_ok=True)
        self.client=KaggleClient(); self.store=ExperimentStore(self.workspace/"experiments.db")
    def scout(self,limit=20):
        rows=[]
        for cat in self.cfg["competition"]["categories"]:
            try: rows.extend(self.client.list_competitions(cat,100))
            except Exception as e: print(f"warning {cat}: {e}")
        seen={}
        for r in rows:
            ref=r.get("ref") or r.get("Ref")
            if ref: seen[str(ref)]=r
        return rank_candidates(list(seen.values()),self.cfg)[:limit]
    def write_scout_report(self,cands,md,json_path):
        md=Path(md); jp=Path(json_path); md.parent.mkdir(parents=True,exist_ok=True); jp.parent.mkdir(parents=True,exist_ok=True)
        jp.write_text(json.dumps([c.to_dict() for c in cands],indent=2))
        lines=["# Kaggle Competition Scout","","|Rank|Teams|Days|Entered|Competition|","|---:|---:|---:|:---:|---|"]
        for i,c in enumerate(cands,1): lines.append(f"|{i}|{c.teams}|{c.days_left:.1f}|{'yes' if c.entered else 'no'}|\`{c.ref}\`|")
        md.write_text("\n".join(lines)+"\n")
    def prepare(self,competition):
        d=self.workspace/competition; d.mkdir(parents=True,exist_ok=True)
        rules=self.client.competition_page(competition,"rules"); evaluation=self.client.competition_page(competition,"evaluation")
        report=inspect_rules(competition,rules,evaluation); (d/"rules.txt").write_text(rules); (d/"evaluation.txt").write_text(evaluation); (d/"rule_report.json").write_text(json.dumps(report.to_dict(),indent=2))
        self.client.download(competition,d/"data"); return report.to_dict()
    def run_suite(self,competition,variant_index=0):
        d=self.workspace/competition
        if not (d/"data").exists(): raise FileNotFoundError("Competition not prepared")
        metric=None
        if (d/"rule_report.json").exists():
            try: metric=json.loads((d/"rule_report.json").read_text()).get("metric")
            except Exception: pass
        t=self.cfg["training"]; out=[]; trained=[]
        for name in t["models"]:
            try:
                r=train_model(competition,d/"data",d/"runs"/name,name,int(t["folds"]),int(t["random_state"]),str(t.get("prediction_mode","auto")),int(t.get("max_rows",400000)),metric,variant_index)
            except Exception as e:
                print(f"skip {name}: {e}"); continue
            eid=self.store.add(competition,r.model_name,r.cv_score,r.metric,r.higher_is_better,r.params,str(r.submission_path)); out.append((eid,r)); trained.append(r)
        ens=build_ensemble(trained,d/"data",d/"runs"/"ensemble",int(t.get("ensemble_top_k",3)),str(t.get("prediction_mode","auto")),metric)
        if ens:
            eid=self.store.add(competition,ens.model_name,ens.cv_score,ens.metric,ens.higher_is_better,ens.params,str(ens.submission_path)); out.append((eid,ens))
        return out
    def _approved(self,competition):
        approved=set(self.cfg["safety"].get("approved_competitions",[])); approved.update(x.strip() for x in os.getenv("KAGGLE_APPROVED_COMPETITIONS","").split(",") if x.strip()); return competition in approved
    def _enabled(self): return os.getenv("KAGGLE_ALLOW_AUTO_SUBMIT",str(self.cfg["safety"].get("allow_auto_submit",False))).lower() in {"1","true","yes","on"}
    def auto_cycle(self,competition):
        d=self.workspace/competition; state_path=self.state_dir/f"{competition}.json"; state=load_state(state_path)
        if not (d/"data").exists(): self.prepare(competition)
        cycle=int(state.get("cycle_index",0)); results=self.run_suite(competition,cycle%5)
        if not results: raise RuntimeError("No successful experiments")
        higher=results[0][1].higher_is_better; eid,best=sorted(results,key=lambda x:x[1].cv_score,reverse=higher)[0]
        prev=state.get("best_cv"); improved=prev is None or ((best.cv_score-float(prev)) if higher else (float(prev)-best.cv_score))>=float(self.cfg["safety"].get("min_cv_improvement",.0001))
        result={
            "competition":competition,
            "cycle_index":cycle,
            "best_experiment_id":eid,
            "best_model":best.model_name,
            "metric":best.metric,
            "best_cv":best.cv_score,
            "previous_best_cv":prev,
            "improved":improved,
            "submitted":False,
            "public_score":None,
            "experiments":[
                {
                    "experiment_id":exp_id,
                    "model":res.model_name,
                    "cv_score":res.cv_score,
                    "metric":res.metric,
                    "params":res.params,
                    "submission_file":str(res.submission_path),
                }
                for exp_id,res in sorted(
                    results,
                    key=lambda x:x[1].cv_score,
                    reverse=higher,
                )
            ],
        }
        if improved and self._enabled() and self._approved(competition):
            text=self.client.submit(competition,best.submission_path,f"kaggle-topper-agent exp={eid} model={best.model_name} cv={best.cv_score:.6f}",int(self.cfg["safety"].get("submission_wait_seconds",1200)))
            score=self.client.extract_public_score(text); ref=self.client.extract_submission_ref(text); self.store.update_submission(eid,score,ref); result.update(submitted=True,public_score=score,submission_ref=ref)
        if prev is None or improved: state.update(best_cv=best.cv_score,best_model=best.model_name,metric=best.metric)
        state["cycle_index"]=cycle+1; state["last_cycle"]=result; save_state(state_path,state); return result
