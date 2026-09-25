from __future__ import annotations
import re
from dataclasses import dataclass, asdict

@dataclass
class RuleReport:
    competition: str
    fetched: bool
    warnings: list[str]
    detected: dict[str,str]
    metric: str|None=None
    def to_dict(self): return asdict(self)

def detect_metric(text:str):
    t=text.lower()
    checks=[("rmsle",["root mean squared logarithmic error","rmsle"]),("rmse",["root mean squared error","rmse"]),("mae",["mean absolute error","mae"]),("roc_auc",["roc-auc","roc auc","area under the roc","auc"]),("log_loss",["log loss","logloss","cross entropy"]),("f1",["f1-score","f1 score"]),("accuracy",["accuracy"]),("r2",["coefficient of determination","r2"])]
    for name,needles in checks:
        if any(n in t for n in needles): return name
    return None

def inspect_rules(competition,rules_text,evaluation_text):
    combined=(rules_text+"\n"+evaluation_text).lower(); detected={}; warnings=[]
    patterns={"external_data":"external data","pretrained_models":"pretrained","submission_limit":"submission","code_competition":"code competition","internet":"no internet"}
    for k,p in patterns.items():
        if p in combined: detected[k]="mentioned"
    metric=detect_metric(evaluation_text)
    if metric: detected["metric"]=metric
    else: warnings.append("Metric not confidently detected.")
    if not rules_text.strip(): warnings.append("Rules page unavailable; do not auto-submit.")
    else: warnings.append("Human review required before enabling auto-submit.")
    if "code_competition" in detected: warnings.append("Generic CSV trainer may not apply.")
    return RuleReport(competition,bool(rules_text.strip()),warnings,detected,metric)
