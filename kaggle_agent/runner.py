from __future__ import annotations
import json, os, traceback, time, shutil
from pathlib import Path
from .discovery import rank_candidates
from .kaggle_client import KaggleClient
from .rules import inspect_rules
from .state import load_state, save_state
from .store import ExperimentStore
from .trainer import train_model, build_ensemble
from .submission_guard import validate_submission
import pandas as pd

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
        t=dict(self.cfg["training"])
        # Optional per-competition training profile. This lets large/specialized
        # competitions use a focused suite without changing the global agent.
        overrides=self.cfg.get("competition_training",{}).get(competition,{})
        t.update(overrides)
        out=[]; trained=[]
        # V8: allow competition profiles to train the same proven model across
        # several random seeds. Seed diversity changes fold assignment and
        # stochastic tree sampling, reducing variance without speculative
        # feature changes.
        seeds=[int(t["random_state"])]
        if competition=="playground-series-s6e9":
            seeds=[int(x) for x in t.get("seeds",[t["random_state"]])]
        for name in t["models"]:
            for seed in seeds:
                run_name=name if len(seeds)==1 else f"{name}_seed{seed}"
                try:
                    r=train_model(competition,d/"data",d/"runs"/run_name,name,int(t["folds"]),seed,str(t.get("prediction_mode","auto")),int(t.get("max_rows",400000)),metric,variant_index)
                    if len(seeds)>1:
                        r.model_name=run_name
                        r.params["seed"]=seed
                except Exception as e:
                    print(f"skip {run_name}: {type(e).__name__}: {e!r}"); traceback.print_exc(); continue
                eid=self.store.add(competition,r.model_name,r.cv_score,r.metric,r.higher_is_better,r.params,str(r.submission_path)); out.append((eid,r)); trained.append(r)
        ens=build_ensemble(trained,d/"data",d/"runs"/"ensemble",int(t.get("ensemble_top_k",3)),str(t.get("prediction_mode","auto")),metric)
        if ens:
            eid=self.store.add(competition,ens.model_name,ens.cv_score,ens.metric,ens.higher_is_better,ens.params,str(ens.submission_path)); out.append((eid,ens))
        return out
    def _approved(self,competition):
        approved=set(self.cfg["safety"].get("approved_competitions",[])); approved.update(x.strip() for x in os.getenv("KAGGLE_APPROVED_COMPETITIONS","").split(",") if x.strip()); return competition in approved
    def _enabled(self): return os.getenv("KAGGLE_ALLOW_AUTO_SUBMIT",str(self.cfg["safety"].get("allow_auto_submit",False))).lower() in {"1","true","yes","on"}

    def research_loop(self, competition, target_score=0.90, max_hours=5.0):
        """Run research-only cycles until target CV ROC-AUC or the runner budget expires.

        This method never submits to Kaggle. It persists the best CSV and OOF
        diagnostics so scheduled GitHub Actions runs can resume from the last best.
        """
        d = self.workspace / competition
        state_path = self.state_dir / f"{competition}.json"
        state = load_state(state_path)
        report_dir = self.report_dir / competition
        report_dir.mkdir(parents=True, exist_ok=True)
        best_csv = report_dir / "best_submission.csv"
        best_meta = report_dir / "best_model.json"
        target_reached = bool(state.get("research_target_reached", False))
        previous_best = state.get("research_best_cv")
        if target_reached and best_csv.is_file() and best_meta.is_file():
            return {
                "competition": competition,
                "status": "target_already_reached",
                "target_score": target_score,
                "best_cv": previous_best,
                "best_submission": str(best_csv),
                "submitted": False,
                "message": "Research target was reached previously; preserved candidate CSV. No Kaggle submission was made."
            }

        if not (d / "data").exists():
            self.prepare(competition)
        deadline = time.monotonic() + max(0.1, float(max_hours)) * 3600
        iterations = 0
        latest = None
        while time.monotonic() < deadline:
            state = load_state(state_path)
            cycle = int(state.get("research_cycle_index", 0))
            results = self.run_suite(competition, cycle % 5)
            iterations += 1
            if not results:
                state["research_cycle_index"] = cycle + 1
                state["research_last_error"] = "No successful experiments in this cycle; stopping this runner to avoid a busy loop."
                state["research_status"] = "no_successful_experiments"
                save_state(state_path, state)
                latest = {"competition": competition, "status": "no_successful_experiments", "submitted": False}
                break

            best_id, best = sorted(
                results,
                key=lambda item: item[1].cv_score,
                reverse=results[0][1].higher_is_better,
            )[0]
            current_best = state.get("research_best_cv")
            improved = current_best is None or (
                best.cv_score > float(current_best)
                if best.higher_is_better
                else best.cv_score < float(current_best)
            )
            metric_is_target = best.metric == "roc_auc"
            reached = metric_is_target and best.higher_is_better and float(best.cv_score) >= float(target_score)

            if improved or not best_csv.is_file():
                shutil.copy2(best.submission_path, best_csv)
                source_report = best.submission_path.parent / "validation_report.json"
                if source_report.is_file():
                    shutil.copy2(source_report, report_dir / "validation_report.json")
                meta = {
                    "competition": competition,
                    "experiment_id": best_id,
                    "model": best.model_name,
                    "metric": best.metric,
                    "cv_score": float(best.cv_score),
                    "target_metric": "roc_auc",
                    "target_score": float(target_score),
                    "target_reached": bool(reached),
                    "submission_file": str(best_csv),
                    "submitted": False,
                    "note": "CV ROC-AUC is a local out-of-fold estimate, not a guaranteed Kaggle leaderboard score. No submission was made."
                }
                best_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                state["research_best_cv"] = float(best.cv_score)
                state["research_best_model"] = best.model_name
                state["research_best_metric"] = best.metric

            if reached:
                state["research_target_reached"] = True
                state["research_status"] = "target_reached_ready_for_review"
                # Mark metadata as target reached even if the best result was
                # already stored from an earlier iteration.
                if best_meta.is_file():
                    meta = json.loads(best_meta.read_text(encoding="utf-8"))
                    meta["target_reached"] = True
                    best_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                latest = {
                    "competition": competition,
                    "status": "target_reached_ready_for_review",
                    "target_metric": "roc_auc",
                    "target_score": float(target_score),
                    "best_cv": float(best.cv_score),
                    "best_model": best.model_name,
                    "best_submission": str(best_csv),
                    "confusion_matrix_report": str(report_dir / "validation_report.json"),
                    "submitted": False
                }
                state["research_last_cycle"] = latest
                state["research_cycle_index"] = cycle + 1
                save_state(state_path, state)
                return latest

            state["research_status"] = "researching"
            state["research_target_reached"] = False
            state["research_cycle_index"] = cycle + 1
            latest = {
                "competition": competition,
                "status": "researching",
                "target_metric": "roc_auc",
                "target_score": float(target_score),
                "best_cv": state.get("research_best_cv"),
                "best_model": state.get("research_best_model"),
                "iterations_this_run": iterations,
                "best_submission": str(best_csv) if best_csv.is_file() else None,
                "confusion_matrix_report": str(report_dir / "validation_report.json") if (report_dir / "validation_report.json").is_file() else None,
                "submitted": False
            }
            state["research_last_cycle"] = latest
            save_state(state_path, state)
        if latest is None:
            latest = {
                "competition": competition,
                "status": "no_successful_experiments",
                "target_score": float(target_score),
                "submitted": False
            }
        if latest.get("status") != "no_successful_experiments" and not state.get("research_target_reached", False):
            latest["status"] = "target_not_reached_runner_budget_exhausted"
        latest["iterations_this_run"] = iterations
        latest["submitted"] = False
        state = load_state(state_path)
        state["research_last_cycle"] = latest
        save_state(state_path, state)
        return latest

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
            # Fail closed: submission is blocked unless metadata and CSV checks pass.
            data_dir = d / "data"
            table_files = sorted([*data_dir.rglob("*.csv")])
            sample_files = [p for p in table_files if any(k in p.stem.lower() for k in ("sample_submission", "sample-submission"))]
            test_files = [p for p in table_files if "test" in p.stem.lower() and "contest" not in p.stem.lower()]
            rule_report = {}
            try:
                rule_report = json.loads((d / "rule_report.json").read_text(encoding="utf-8"))
            except Exception:
                pass
            deadline = None
            try:
                for category in self.cfg["competition"].get("categories", ["all"]):
                    for row in self.client.list_competitions(category, 100):
                        ref = str(row.get("ref") or row.get("Ref") or "")
                        if ref == competition:
                            deadline = row.get("deadline") or row.get("Deadline")
                            break
                    if deadline:
                        break
            except Exception as exc:
                print(f"Submission blocked: could not verify competition deadline ({type(exc).__name__}).")
            if not sample_files or not test_files:
                print("Submission blocked: unable to identify test and sample-submission CSV files.")
            else:
                try:
                    sample = pd.read_csv(sample_files[0], nrows=0)
                    test = pd.read_csv(test_files[0])
                    check = validate_submission(
                        best.submission_path,
                        expected_rows=len(test),
                        expected_columns=list(sample.columns),
                        deadline=str(deadline) if deadline else None,
                        rules_fetched=bool(rule_report.get("fetched")),
                        metric=best.metric if rule_report.get("metric") == best.metric else None,
                        approved=self._approved(competition),
                        enabled=self._enabled(),
                    )
                    result["submission_preflight"] = check.to_dict()
                    if check.ok:
                        text=self.client.submit(competition,best.submission_path,f"kaggle-topper-agent exp={eid} model={best.model_name} cv={best.cv_score:.6f}",int(self.cfg["safety"].get("submission_wait_seconds",1200)))
                        score=self.client.extract_public_score(text); ref=self.client.extract_submission_ref(text); self.store.update_submission(eid,score,ref); result.update(submitted=True,public_score=score,submission_ref=ref)
                    else:
                        print("Submission blocked by preflight: " + "; ".join(check.errors))
                except Exception as exc:
                    result["submission_preflight"] = {"ok": False, "errors": [f"Preflight failed closed: {type(exc).__name__}"], "warnings": []}
                    print(f"Submission blocked: preflight failed ({type(exc).__name__}).")
        if prev is None or improved: state.update(best_cv=best.cv_score,best_model=best.model_name,metric=best.metric)
        state["cycle_index"]=cycle+1; state["last_cycle"]=result; save_state(state_path,state); return result
