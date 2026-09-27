"""Select model and threshold on tuning groups; evaluate holdout only after selection."""
import argparse
import json
from pathlib import Path
import time
import shutil
import sqlite3
import numpy as np
from xgboost import XGBClassifier
from pipeline import log,FEATURE_NAMES,normalize
from group_policy import best_indices,chosen_pairs,apply_policies

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--experiments',default='600:7,1200:8,1600:6')
    p.add_argument('--model-kind',choices=['xgboost','lightgbm'],default='xgboost')
    p.add_argument('--skip-holdout',action='store_true')
    a=p.parse_args();cache=Path(a.cache);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((cache/'metadata.json').read_text());assert meta['complete']
    with sqlite3.connect(Path(meta['db']).resolve().as_uri()+'?mode=ro',uri=True) as db:
        indexed_target_records=db.execute('SELECT count(*) FROM records').fetchone()[0]
    anchors=meta['anchors'];n=len(anchors);m=meta['pairs']
    x=np.memmap(cache/'features.f32',dtype=np.float32,mode='r',shape=(m,len(meta['feature_names'])))
    y=np.memmap(cache/'labels.i1',dtype=np.int8,mode='r',shape=(m,))
    group=np.memmap(cache/'groups.i4',dtype=np.int32,mode='r',shape=(m,))
    targets=np.memmap(cache/'targets.s32',dtype='S32',mode='r',shape=(m,))
    splits=np.array([r['split'] for r in anchors]);actual=np.array([r['true_count'] for r in anchors])
    countries=np.array([normalize(r['country']) for r in anchors])
    masks={'train':splits<6,'tune':(splits>=6)&(splits<8),'holdout':splits>=8}
    train=masks['train'][group];tune=masks['tune'][group]
    # Keep threshold sweeps restricted to tuning pairs, avoiding repeated full-array copies.
    tg=np.asarray(group[tune]);ty=np.asarray(y[tune]);tt=np.asarray(targets[tune]);tune_x=np.asarray(x[tune]);train_x=np.asarray(x[train]);train_y=np.asarray(y[train])
    train_weight=np.asarray(np.memmap(cache/'weights.f32',dtype=np.float32,mode='r',shape=(m,))[train]) if (cache/'weights.f32').exists() else None
    def anchor_scores(scores,threshold,groups,labels,rescue=None,best=None,target_ids=None,policies=None):
        chosen=apply_policies(scores,groups,countries,dict(threshold=threshold,rescue_threshold=rescue,country_policies=policies),target_ids) if policies else chosen_pairs(scores,threshold,groups,n,rescue,best,target_ids)
        counts=np.bincount(groups[chosen],minlength=n)
        hits=np.bincount(groups[chosen],weights=labels[chosen],minlength=n)
        den=.25*actual+counts
        f=np.divide(1.25*hits,den,out=np.zeros(n),where=den>0)
        f[actual==0]=(counts[actual==0]==0)
        return f
    results=[];best=None;started=time.monotonic()
    for item in a.experiments.split(','):
        trees,depth=map(int,item.split(':'));began=time.monotonic()
        if a.model_kind=='lightgbm':
            from lightgbm import LGBMClassifier
            model=LGBMClassifier(n_estimators=trees,max_depth=depth,num_leaves=min(2**depth-1,127),learning_rate=.05,
                subsample=.85,subsample_freq=1,colsample_bytree=.9,reg_lambda=1.0,min_child_samples=40,
                n_jobs=4,random_state=42,verbosity=-1,importance_type='gain')
        else:
            model=XGBClassifier(n_estimators=trees,max_depth=depth,learning_rate=.05,subsample=.85,colsample_bytree=.85,max_bin=512,tree_method='hist',n_jobs=4,random_state=42,eval_metric='logloss')
        model.fit(train_x,train_y,sample_weight=train_weight)
        score=np.asarray(model.predict_proba(tune_x)[:,1],dtype=np.float32)
        top=best_indices(score,tg,n)
        threshold,value=max(((float(t),float(anchor_scores(score,t,tg,ty,target_ids=tt)[masks['tune']].mean())) for t in np.arange(.1,.951,.01)),key=lambda r:(r[1],r[0]))
        rescue=None
        for t in np.arange(max(.1,threshold-.05),min(.951,threshold+.101),.025):
            for fallback in np.arange(.1,min(.801,t),.05):
                candidate=float(anchor_scores(score,float(t),tg,ty,float(fallback),top,tt)[masks['tune']].mean())
                if candidate>value:
                    threshold,value,rescue=float(t),candidate,float(fallback)
        policies={}
        for country in np.unique(countries[masks['tune']]):
            anchor_mask=masks['tune']&(countries==country)
            if anchor_mask.sum()<2000:continue
            pair_mask=countries[tg]==country;cs,cg,cy,ct=score[pair_mask],tg[pair_mask],ty[pair_mask],tt[pair_mask]
            cbest=best_indices(cs,cg,n)
            country_best=float(anchor_scores(cs,threshold,cg,cy,rescue,cbest,ct)[anchor_mask].mean())
            policy=dict(threshold=threshold,rescue_threshold=rescue)
            for t in np.clip(threshold+np.arange(-.12,.121,.04),.1,.97):
                for r in dict.fromkeys([rescue,None,.2,.4,.6]):
                    if r is not None and r>=t:continue
                    candidate=float(anchor_scores(cs,float(t),cg,cy,r,cbest,ct)[anchor_mask].mean())
                    if candidate>country_best:
                        country_best=candidate;policy=dict(threshold=float(t),rescue_threshold=r)
            policies[str(country)]=policy
        value=float(anchor_scores(score,threshold,tg,ty,rescue,top,tt,policies)[masks['tune']].mean())
        result=dict(trees=trees,depth=depth,threshold=threshold,rescue_threshold=rescue,country_policies=policies,tune_macro_f05=value,seconds=time.monotonic()-began)
        results.append(result);log(str(result))
        if best is None or value>best['tune_macro_f05']:
            best=result
            if a.model_kind=='lightgbm':model.booster_.save_model(str(out/'model.txt'))
            else:model.save_model(out/'model.ubj')
            np.save(out/'tune_scores.npy',score)
    del train_x,train_y,tune_x
    reports={}
    for s,mask in masks.items():
        selected=[r for r,keep in zip(anchors,mask) if keep];true=sum(r['true_count'] for r in selected)
        reports[s]=dict(anchors=len(selected),pairs=sum(r['candidates'] for r in selected),true_links=true,
            candidate_recall=sum(r['recalled'] for r in selected)/max(1,true),mean_candidates=float(np.mean([r['candidates'] for r in selected])),
            zero_candidate_anchors=sum(r['candidates']==0 for r in selected))
    config=dict(threshold=best['threshold'],top_k=meta['top_k'],max_block=meta['max_block'],model_kind=a.model_kind,
        model_file='model.txt' if a.model_kind=='lightgbm' else 'model.ubj',trees=best['trees'],depth=best['depth'],
        retrieval_version=meta['retrieval_version'],feature_names=meta['feature_names'],model_license='MIT',library_license='MIT' if a.model_kind=='lightgbm' else 'Apache-2.0',
        indexed_target_records=indexed_target_records,training_scope=meta['data'],refit_on_holdout=False,supplemental_index_required=True)
    config.update(base_feature_names=FEATURE_NAMES,contextual_features=meta.get('contextual_features',False),
                  rarity_features='anchor_address_frequency' in meta['feature_names'],posterior_features=bool(meta.get('posterior_dir')),
                  competition_features=meta.get('competition_features',False),rescue_threshold=best['rescue_threshold'],country_policies=best['country_policies'],unique_targets=True)
    if meta.get('robust_features'):
        config.update(robust_features=True,text_weights_file='text_weights.joblib')
        shutil.copy2(meta['text_weights'],out/'text_weights.joblib')
    if meta.get('negative_sampling'):config['negative_sampling']=meta['negative_sampling']
    if meta.get('posterior_dir'):
        posterior=Path(meta['posterior_dir'])
        shutil.copy2(posterior/'base_model.ubj',out/'base_model.ubj')
        shutil.copy2(posterior/'base_config.json',out/'base_config.json')
    report=dict(tune_macro_f05=best['tune_macro_f05'],threshold=best['threshold'],rescue_threshold=best['rescue_threshold'],unique_targets=True,splits=reports,model_search=results,selected_model=a.model_kind,
        indexed_target_records=indexed_target_records,scope='Local held-out anchor groups; performance on new datasets is unmeasured.',seconds=time.monotonic()-started)
    if not a.skip_holdout:
        hold=masks['holdout'][group]
        if a.model_kind=='lightgbm':
            from lightgbm import Booster
            model=Booster(model_file=str(out/'model.txt'));score=np.asarray(model.predict(x[hold],num_threads=4),dtype=np.float32)
            importance=model.feature_importance(importance_type='gain')
        else:
            model=XGBClassifier();model.load_model(out/'model.ubj');model.set_params(n_jobs=4)
            score=model.predict_proba(x[hold])[:,1];importance=model.feature_importances_
        np.save(out/'holdout_scores.npy',score)
        f=anchor_scores(score,best['threshold'],group[hold],y[hold],best['rescue_threshold'],target_ids=targets[hold],policies=best['country_policies']);hc=masks['holdout']
        report.update(holdout_macro_f05=float(f[hc].mean()),holdout_by_country={country:float(f[hc&np.array([r['country']==country for r in anchors])].mean()) for country in sorted({r['country'] for r in anchors})},holdout_singletons=int((hc&(actual==0)).sum()))
        chosen=apply_policies(score,group[hold],countries,config,targets[hold]);report['holdout_link_precision']=float(y[hold][chosen].sum()/max(1,chosen.sum()));report['holdout_link_recall']=float(y[hold][chosen].sum()/actual[hc].sum())
        report['feature_importance']=dict(zip(meta['feature_names'],map(float,importance)))
    (out/'model_config.json').write_text(json.dumps(config,indent=2));(out/'metrics.json').write_text(json.dumps(report,indent=2))
    log(json.dumps(report,indent=2))

if __name__=='__main__':main()
