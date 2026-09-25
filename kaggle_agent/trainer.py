from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math, warnings
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from .feature_engineering import engineer_features

@dataclass
class TrainResult:
    competition:str; model_name:str; task:str; metric:str; higher_is_better:bool
    cv_score:float; submission_path:Path; model_path:Path; target:str; features:list[str]
    params:dict; validation_predictions:np.ndarray; test_predictions:np.ndarray

def _table_files(data):
    root = Path(data)
    return sorted([*root.rglob("*.csv"), *root.rglob("*.parquet")])

def _read_table(path, nrows=None):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return df if nrows is None else df.head(nrows)
    return pd.read_csv(path, nrows=nrows)

def _columns(path):
    return list(_read_table(path, nrows=5).columns)

def _discover_dataset_files(data):
    files = _table_files(data)
    if len(files) < 3:
        raise FileNotFoundError(
            f"Need at least 3 table files (train/test/submission); found {[p.name for p in files]}"
        )

    def name_score(path, words):
        name = path.stem.lower().replace("-", "_")
        return sum(3 if name == w else 1 for w in words if w in name)

    sample_words = ("sample_submission", "sample", "submission", "submit")
    train_words = ("train", "training")
    test_words = ("test", "testing")

    sample_ranked = sorted(files, key=lambda p: name_score(p, sample_words), reverse=True)
    sample = sample_ranked[0] if name_score(sample_ranked[0], sample_words) > 0 else None

    remaining = [p for p in files if p != sample]
    train_ranked = sorted(remaining, key=lambda p: name_score(p, train_words), reverse=True)
    test_ranked = sorted(remaining, key=lambda p: name_score(p, test_words), reverse=True)
    train = train_ranked[0] if train_ranked and name_score(train_ranked[0], train_words) > 0 else None
    test = test_ranked[0] if test_ranked and name_score(test_ranked[0], test_words) > 0 else None

    # Fallback: infer train/test from schema. In a standard Kaggle tabular
    # competition train has exactly one extra target column relative to test.
    if train is None or test is None or train == test:
        schemas = {p: set(_columns(p)) for p in remaining}
        pairs = []
        for a in remaining:
            for b in remaining:
                if a == b:
                    continue
                extra = schemas[a] - schemas[b]
                missing = schemas[b] - schemas[a]
                if len(extra) == 1 and len(missing) == 0:
                    score = name_score(a, train_words) + name_score(b, test_words)
                    pairs.append((score, a, b))
        if pairs:
            _, train, test = max(pairs, key=lambda x: x[0])

    if train is None or test is None:
        raise FileNotFoundError(
            "Could not infer train/test files. Downloaded tables: "
            + ", ".join(p.name for p in files)
        )

    # If the sample submission did not have a recognizable filename, identify
    # the remaining file whose columns are mostly ID + target-like outputs.
    if sample is None or sample in {train, test}:
        leftovers = [p for p in files if p not in {train, test}]
        if leftovers:
            test_cols = set(_columns(test))
            def sample_likelihood(p):
                cols = set(_columns(p))
                # Reward files that share identifiers with test but also contain
                # one or more columns not present in test.
                return (len(cols & test_cols), len(cols - test_cols), -len(cols))
            sample = max(leftovers, key=sample_likelihood)

    if sample is None:
        raise FileNotFoundError(
            "Could not infer sample submission file. Downloaded tables: "
            + ", ".join(p.name for p in files)
        )

    return train, test, sample
def _target(train,test):
    c=[x for x in train.columns if x not in test.columns]
    if len(c)!=1: raise ValueError(f"Need exactly one target; found {c}")
    return c[0]
def _task(y):
    if y.dtype==object or str(y.dtype).startswith("category") or str(y.dtype)=="bool": return "classification"
    return "classification" if y.nunique(dropna=True)<=max(20,int(len(y)*.01)) else "regression"
def _prep(X,features):
    # Use pandas' dtype helpers instead of string equality. Kaggle CSV data can
    # arrive as object, pandas string dtype, categorical, boolean or extension
    # dtypes; only genuinely numeric columns should ever reach median imputation.
    nums=[c for c in features if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    cats=[c for c in features if c not in nums]
    return ColumnTransformer([
        ("num",Pipeline([
            ("imp",SimpleImputer(strategy="median"))
        ]),nums),
        ("cat",Pipeline([
            ("imp",SimpleImputer(strategy="most_frequent")),
            ("ord",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))
        ]),cats)
    ])
def _regression_splits(X, y, folds, random_state):
    """Balance target ranges across folds so rare expensive cars are represented."""
    try:
        n_bins = min(10, max(folds, int(len(y) / 80)))
        ranked = pd.Series(y).rank(method="first")
        bins = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
        if pd.Series(bins).nunique() >= folds:
            return list(StratifiedKFold(folds, shuffle=True, random_state=random_state).split(X, bins))
    except Exception:
        pass
    return list(KFold(folds, shuffle=True, random_state=random_state).split(X))


def _catboost_frames(X, Xt):
    Xc = X.copy()
    Xtc = Xt.copy()
    cat_cols = [
        col for col in Xc.columns
        if not pd.api.types.is_numeric_dtype(Xc[col]) or pd.api.types.is_bool_dtype(Xc[col])
    ]
    num_cols = [col for col in Xc.columns if col not in cat_cols]

    for col in cat_cols:
        Xc[col] = Xc[col].astype("string").fillna("__MISSING__").astype(str)
        Xtc[col] = Xtc[col].astype("string").fillna("__MISSING__").astype(str)

    for col in num_cols:
        Xc[col] = pd.to_numeric(Xc[col], errors="coerce")
        Xtc[col] = pd.to_numeric(Xtc[col], errors="coerce")
        med = Xc[col].median()
        if pd.isna(med):
            med = 0.0
        Xc[col] = Xc[col].fillna(float(med))
        Xtc[col] = Xtc[col].fillna(float(med))

    return Xc, Xtc, cat_cols


def _model(name,task,seed,variant):
    v=variant%5
    if name=="histgb":
        lr=[.06,.035,.08,.045,.025][v]; leaves=[31,63,15,31,47][v]; it=[500,850,380,750,1100][v]
        p=dict(learning_rate=lr,max_iter=it,max_leaf_nodes=leaves,l2_regularization=[1,2,3,5,1.5][v])
        return ((HistGradientBoostingClassifier if task=="classification" else HistGradientBoostingRegressor)(**p,random_state=seed),p)
    if name=="extratrees":
        p=dict(n_estimators=[700,900,800,1100,1000][v],min_samples_leaf=[2,1,4,2,3][v],max_features=["sqrt",.75,1.0,.6,"sqrt"][v])
        cls=ExtraTreesClassifier if task=="classification" else ExtraTreesRegressor
        kw=dict(**p,n_jobs=-1,random_state=seed)
        if task=="classification": kw["class_weight"]="balanced"
        return cls(**kw),p
    if name=="catboost":
        try:
            from catboost import CatBoostClassifier,CatBoostRegressor
        except ImportError as e: raise RuntimeError("catboost not installed") from e
        p=dict(iterations=[900,1200,750,1350,1050][v],depth=[7,8,6,9,7][v],learning_rate=[.04,.03,.055,.025,.035][v])
        cls=CatBoostClassifier if task=="classification" else CatBoostRegressor
        return cls(**p,verbose=False,random_seed=seed,allow_writing_files=False),p
    if name=="lightgbm":
        try:
            from lightgbm import LGBMClassifier,LGBMRegressor
        except ImportError as e: raise RuntimeError("lightgbm not installed") from e
        p=dict(n_estimators=[1200,1600,900,1800,1400][v],learning_rate=[.03,.022,.04,.018,.025][v],num_leaves=[31,63,15,47,39][v])
        cls=LGBMClassifier if task=="classification" else LGBMRegressor
        return cls(**p,subsample=.9,colsample_bytree=.9,random_state=seed,n_jobs=-1,verbosity=-1),p
    if name=="xgboost":
        try:
            from xgboost import XGBClassifier,XGBRegressor
        except ImportError as e: raise RuntimeError("xgboost not installed") from e
        p=dict(n_estimators=[1200,1500,900,1750,1350][v],learning_rate=[.03,.024,.04,.02,.026][v],max_depth=[6,8,4,7,5][v])
        cls=XGBClassifier if task=="classification" else XGBRegressor
        return cls(**p,subsample=.9,colsample_bytree=.9,random_state=seed,n_jobs=-1),p
    raise ValueError(name)
def _score(task,y,pred,nc,metric=None,labels=None):
    if task=="regression":
        m=metric if metric in {"rmse","mse","mae","rmsle","r2"} else "rmse"
        if m=="mae":return m,False,float(mean_absolute_error(y,pred))
        if m=="mse":return m,False,float(mean_squared_error(y,pred))
        if m=="rmsle":return m,False,float(math.sqrt(mean_squared_log_error(y,np.clip(pred,0,None))))
        if m=="r2":return m,True,float(r2_score(y,pred))
        return "rmse",False,float(math.sqrt(mean_squared_error(y,pred)))
    m=metric if metric in {"roc_auc","log_loss","accuracy","f1"} else ("roc_auc" if nc==2 else "log_loss")
    labels=np.array(labels if labels is not None else np.unique(y))
    if nc==2:
        if m=="roc_auc":return m,True,float(roc_auc_score(y,pred))
        if m=="log_loss":return m,False,float(log_loss(y,np.c_[1-pred,pred],labels=labels))
        hard=np.where(pred>=.5,labels[1],labels[0])
        if m=="f1":return m,True,float(f1_score(y,hard,pos_label=labels[1]))
        return "accuracy",True,float(accuracy_score(y,hard))
    if m=="log_loss":return m,False,float(log_loss(y,pred,labels=labels))
    hard=labels[np.argmax(pred,axis=1)]
    return (("f1",True,float(f1_score(y,hard,average="macro"))) if m=="f1" else ("accuracy",True,float(accuracy_score(y,hard))))
def train_model(competition,data_dir,out_dir,model_name,folds=5,random_state=42,prediction_mode="auto",max_rows=400000,metric_override=None,variant_index=0):
    data=Path(data_dir); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    tp, sp, pp = _discover_dataset_files(data)
    print(f"dataset files: train={tp.name} test={sp.name} submission={pp.name}")
    train=_read_table(tp); test=_read_table(sp); sample=_read_table(pp)
    if len(train)>max_rows: train=train.sample(max_rows,random_state=random_state).reset_index(drop=True)
    target=_target(train,test); features=[c for c in test.columns if c in train.columns]
    X=train[features].copy(); y=train[target].copy(); Xt=test[features].copy(); task=_task(y)
    combined = pd.concat([X, Xt], axis=0, ignore_index=True)
    combined = engineer_features(combined)
    X = combined.iloc[:len(X)].reset_index(drop=True)
    Xt = combined.iloc[len(X):].reset_index(drop=True)
    features = list(X.columns)
    nc=int(y.nunique()) if task=="classification" else 0; labels=np.unique(y.dropna()) if task=="classification" else None
    est,params=_model(model_name,task,random_state,variant_index)
    params=dict(params,variant_index=variant_index%5)

    if model_name == "catboost":
        Xcb, Xtcb, cat_cols = _catboost_frames(X, Xt)
        params["native_categorical_count"] = len(cat_cols)
        if task=="classification":
            split=StratifiedKFold(folds,shuffle=True,random_state=random_state)
            split_iter=list(split.split(Xcb,y))
            oof=np.zeros((len(train),nc)) if nc>2 else np.zeros(len(train))
        else:
            split_iter=_regression_splits(Xcb,y,folds,random_state)
            oof=np.zeros(len(train))
        preds=[]
        for tr,va in split_iter:
            fold_model=clone(est)
            fold_model.fit(Xcb.iloc[tr],y.iloc[tr],cat_features=cat_cols,verbose=False)
            if task=="classification":
                vp=fold_model.predict_proba(Xcb.iloc[va]); tp2=fold_model.predict_proba(Xtcb)
                if nc==2:
                    oof[va]=vp[:,1]; preds.append(tp2[:,1])
                else:
                    oof[va]=vp; preds.append(tp2)
            else:
                oof[va]=np.clip(fold_model.predict(Xcb.iloc[va]),0,None)
                preds.append(np.clip(fold_model.predict(Xtcb),0,None))
        test_pred=np.mean(preds,axis=0)
        metric,higher,cv=_score(task,y,oof,nc,metric_override,labels)
        final=clone(est)
        final.fit(Xcb,y,cat_features=cat_cols,verbose=False)
        model_path=out/"model.joblib"
        joblib.dump({"model":final,"features":features,"cat_cols":cat_cols},model_path)
    else:
        prep=_prep(X,features)
        if task=="classification":
            split=StratifiedKFold(folds,shuffle=True,random_state=random_state)
            split_iter=list(split.split(X,y))
            oof=np.zeros((len(train),nc)) if nc>2 else np.zeros(len(train))
        else:
            split_iter=_regression_splits(X,y,folds,random_state)
            oof=np.zeros(len(train))
        preds=[]
        for tr,va in split_iter:
            pipe=Pipeline([("prep",clone(prep)),("model",clone(est))]); pipe.fit(X.iloc[tr],y.iloc[tr])
            if task=="classification":
                vp=pipe.predict_proba(X.iloc[va]); tp2=pipe.predict_proba(Xt)
                if nc==2:
                    oof[va]=vp[:,1]; preds.append(tp2[:,1])
                else:
                    oof[va]=vp; preds.append(tp2)
            else:
                oof[va]=np.clip(pipe.predict(X.iloc[va]),0,None)
                preds.append(np.clip(pipe.predict(Xt),0,None))
        test_pred=np.mean(preds,axis=0)
        metric,higher,cv=_score(task,y,oof,nc,metric_override,labels)
        final=Pipeline([("prep",prep),("model",est)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final.fit(X,y)
        model_path=out/"model.joblib"; joblib.dump(final,model_path)
    targets=[c for c in sample.columns if c not in test.columns] or [sample.columns[-1]]
    if len(targets)==1:
        col=targets[0]
        if task=="classification" and nc==2:
            values=sample[col].dropna(); probabilistic=len(values)>0 and pd.api.types.is_numeric_dtype(values) and values.between(0,1).all() and not np.allclose(values,np.round(values))
            if prediction_mode=="proba" or (prediction_mode=="auto" and probabilistic): sample[col]=test_pred
            else: sample[col]=np.where(test_pred>=.5,labels[1],labels[0])
        else: sample[col]=np.clip(test_pred,0,None) if task=="regression" else test_pred
    elif task=="classification" and nc>2 and len(targets)==nc:
        for i,c in enumerate(targets): sample[c]=test_pred[:,i]
    else: raise ValueError("Unsupported submission layout")
    sub=out/"submission.csv"; sample.to_csv(sub,index=False)
    return TrainResult(competition,model_name,task,metric,higher,cv,sub,model_path,target,features,params,oof,test_pred)

def build_ensemble(results,data_dir,out_dir,top_k=3,prediction_mode="auto",metric_override=None):
    if len(results)<2:return None
    first=results[0]; same=[r for r in results if r.task==first.task and r.metric==first.metric]
    if len(same)<2:return None
    ranked=sorted(same,key=lambda r:r.cv_score,reverse=first.higher_is_better)[:top_k]
    data=Path(data_dir); tp, sp, pp = _discover_dataset_files(data)
    train=_read_table(tp); test=_read_table(sp); sample=_read_table(pp)
    target=_target(train,test); y=train[target]; nc=int(y.nunique()) if first.task=="classification" else 0; labels=np.unique(y.dropna()) if first.task=="classification" else None

    if first.task=="regression":
        P=np.column_stack([r.validation_predictions for r in ranked])
        T=np.column_stack([r.test_predictions for r in ranked])
        try:
            weights=np.linalg.lstsq(P,np.asarray(y,dtype=float),rcond=None)[0]
            weights=np.clip(weights,0,None)
            if float(weights.sum()) <= 0:
                weights=np.ones(len(ranked),dtype=float)
            weights=weights/weights.sum()
        except Exception:
            weights=np.ones(len(ranked),dtype=float)/len(ranked)
        val=np.clip(P @ weights,0,None)
        testp=np.clip(T @ weights,0,None)
        ensemble_weights={r.model_name:float(w) for r,w in zip(ranked,weights)}
    else:
        val=np.mean([r.validation_predictions for r in ranked],axis=0)
        testp=np.mean([r.test_predictions for r in ranked],axis=0)
        ensemble_weights={r.model_name:1.0/len(ranked) for r in ranked}

    metric,higher,cv=_score(first.task,y,val,nc,metric_override,labels)
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); targets=[c for c in sample.columns if c not in test.columns] or [sample.columns[-1]]
    if len(targets)==1:
        if first.task=="classification" and nc==2: sample[targets[0]]=testp
        else: sample[targets[0]]=testp
    elif first.task=="classification" and nc>2 and len(targets)==nc:
        for i,c in enumerate(targets): sample[c]=testp[:,i]
    else:return None
    sub=out/"submission.csv"; sample.to_csv(sub,index=False); meta=out/"ensemble.json"; meta.write_text(str(ensemble_weights))
    return TrainResult(first.competition,"ensemble_"+"_".join(r.model_name for r in ranked),first.task,metric,higher,cv,sub,meta,target,first.features,{"models":[r.model_name for r in ranked],"weights":ensemble_weights},val,testp)
