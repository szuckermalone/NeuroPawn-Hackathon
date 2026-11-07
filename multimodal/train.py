"""Train RF from independently labelled multimodal windows. Group holdout is mandatory."""
import argparse
from pathlib import Path
from .features import ALL_FEATURES

def main():
    p=argparse.ArgumentParser()
    p.add_argument('csv',help='Human-labeled CSV with all 10 feature columns and label')
    p.add_argument('--output',default='models/neuroguard_multimodal_rf.joblib')
    p.add_argument('--group',choices=('participant_id','session_id'),default='participant_id')
    args=p.parse_args()
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import classification_report,confusion_matrix
    df=pd.read_csv(args.csv)
    required=list(ALL_FEATURES)+['label',args.group]
    missing=[x for x in required if x not in df]
    if missing:raise ValueError('Required columns missing: '+', '.join(missing))
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=required)
    df=df[df.label.isin([0,1])]
    if len(df)<100:raise ValueError('Need >=100 labeled windows, with independent labels')
    if df[args.group].nunique()<3:raise ValueError('Need >=3 independent groups; otherwise no valid holdout')
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42)
    train,test=next(split.split(df,df.label,groups=df[args.group]))
    tr,te=df.iloc[train],df.iloc[test]
    if tr.label.nunique()!=2 or te.label.nunique()!=2:
        raise ValueError('Both training and held-out group must contain both labels')
    model=RandomForestClassifier(n_estimators=150,min_samples_leaf=5,random_state=42,
                                 class_weight='balanced',n_jobs=-1)
    model.fit(tr[list(ALL_FEATURES)],tr.label.astype(int))
    pred=model.predict(te[list(ALL_FEATURES)])
    print('Train groups:',sorted(map(str,tr[args.group].unique())))
    print('Held-out groups:',sorted(map(str,te[args.group].unique())))
    print('Held-out validation (not clinical performance):')
    print(classification_report(te.label.astype(int),pred,zero_division=0))
    print('Confusion matrix:',confusion_matrix(te.label.astype(int),pred).tolist())
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,dest)
    print('Saved',dest)

if __name__=='__main__':main()
