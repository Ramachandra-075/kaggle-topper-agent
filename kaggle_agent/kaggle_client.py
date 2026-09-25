from __future__ import annotations
import csv, io, json, re, subprocess, zipfile
from pathlib import Path

class KaggleCommandError(RuntimeError): pass

class KaggleClient:
    def _run(self,args,cwd=None):
        p=subprocess.run(["kaggle",*args],cwd=str(cwd) if cwd else None,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if p.returncode: raise KaggleCommandError(p.stdout)
        return p.stdout
    def list_competitions(self,category="all",page_size=100,group="general"):
        out=self._run(["competitions","list","--group",group,"--category",category,"--sort-by","numberOfTeams","--page-size",str(page_size),"--format","json"])
        try:
            data=json.loads(out)
            if isinstance(data,list): return data
            for k in ("competitions","items","results"):
                if isinstance(data,dict) and isinstance(data.get(k),list): return data[k]
        except Exception: pass
        out=self._run(["competitions","list","--group",group,"--category",category,"--sort-by","numberOfTeams","--page-size",str(page_size),"-v"])
        return list(csv.DictReader(io.StringIO(out)))
    def competition_page(self,competition,page_name):
        return self._run(["competitions","pages","list",competition,"--page-name",page_name,"--content"])
    def download(self,competition,dest):
        dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
        self._run(["competitions","download",competition,"-p",str(dest),"-o","-q"])
        for z in dest.rglob("*.zip"):
            try:
                with zipfile.ZipFile(z) as f: f.extractall(z.parent)
            except zipfile.BadZipFile: pass
        return dest
    def submit(self,competition,file,message,wait_seconds=1200):
        return self._run(["competitions","submit",competition,"-f",str(file),"-m",message,"--wait",str(wait_seconds),"--poll-interval","15"])
    def submissions(self,competition): return self._run(["competitions","submissions",competition,"-v","-q"])
    def leaderboard(self,competition): return self._run(["competitions","leaderboard",competition,"-s","-v","-q"])
    @staticmethod
    def extract_public_score(text):
        m=re.search(r"Public Score:\s*([-+]?\d+(?:\.\d+)?)",text,re.I); return float(m.group(1)) if m else None
    @staticmethod
    def extract_submission_ref(text):
        m=re.search(r"Submission ref:\s*(\d+)",text,re.I); return int(m.group(1)) if m else None
