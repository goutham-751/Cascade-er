"""Generate group out-of-fold matching probabilities for a second matching stage."""
import argparse,json,time
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from pipeline import bucket,log

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--trees',type=int,default=800);p.add_argument('--depth',type=int,default=7);a=p.parse_args()
    cache,out=Path(a.cache),Path(a.out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((cache/'metadata.json').read_text());m=meta['pairs'];n=len(meta['feature_names'])
    x=np.memmap(cache/'features.f32',dtype=np.float32,mode='r',shape=(m,n));y=np.memmap(cache/'labels.i1',dtype=np.int8,mode='r');g=np.memmap(cache/'groups.i4',dtype=np.int32,mode='r')
    split=np.array([r['split'] for r in meta['anchors']]);fold=np.array([bucket('crossfit:'+r['id'],3) for r in meta['anchors']]);train=split[g]<6;pfold=fold[g]
    predictions=np.zeros(m,dtype=np.float32);start=time.monotonic()
    for k in range(4):
        fit=train&(pfold!=k) if k<3 else train
        predict=train&(pfold==k) if k<3 else ~train
        model=XGBClassifier(n_estimators=a.trees,max_depth=a.depth,learning_rate=.05,subsample=.85,colsample_bytree=.85,max_bin=512,tree_method='hist',n_jobs=4,random_state=42,eval_metric='logloss')
        model.fit(x[fit],y[fit]);predictions[predict]=model.predict_proba(x[predict])[:,1]
        if k==3:model.save_model(out/'base_model.ubj')
        log(f'Crossfit stage {k+1}/4 complete; {time.monotonic()-start:.1f}s')
    np.save(out/'crossfit_scores.npy',predictions)
    (out/'base_config.json').write_text(json.dumps(dict(trees=a.trees,depth=a.depth,feature_names=meta['feature_names'],cache=str(cache.resolve()),folds=3),indent=2))

if __name__=='__main__':main()
