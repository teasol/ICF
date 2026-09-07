"""Independent RU-81 audit: pairwise tie-aware AUC, source hashes, raw replay.

Does not import the diagnostic report's metric or ensemble implementation.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from scipy.stats import rankdata, t

ROOT=Path(__file__).resolve().parents[2]

def auc(score, label):
    p, n=score[label==1], score[label==0]
    if not len(p) or not len(n):
        raise ValueError('Single-class evaluation')
    d=p[:,None]-n[None,:]
    return float(((d>0)+0.5*(d==0)).mean())

def combine(m):
    probs=np.stack([torch.sigmoid(m[b].float()).numpy() for b in ('cv','bm','bd','qa','ds')])
    # Keep the original float32 reduction order to preserve rounding ties.
    ordered=np.sort(probs,axis=0)
    return ((ordered[1]+ordered[2])+ordered[3])/np.float32(3)

def main():
    folder=ROOT/'predictions'/sys.argv[1]
    manifest=json.loads((folder/'manifest.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    checks={'source_hashes':{},'fold_hashes':{},'task_coverage':{},'contrasts':[],
            'max_replay_error':0.,'max_path_error':0.,'max_ensemble_replay_error':0.,
            'max_summary_delta_error':0.,'max_summary_rank_error':0.,'geometry':{}}
    for name, expected in manifest['files'].items():
        # ru.py close moves the completed card into the history database.
        # Preserve and verify the pre-execution bytes, not the filled closeout.
        path = folder/'plan.json' if name=='docs/ru/RU-81.json' and (folder/'plan.json').exists() else ROOT/name
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        checks['source_hashes'][name]=(actual==expected)
        assert actual==expected, name
    for name, expected in manifest['fold_files'].items():
        actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        checks['fold_hashes'][name]=(actual==expected)
        assert actual==expected, name
    data={}
    for task in manifest['tasks']:
        name=task.replace('/','_')
        diag=torch.load(folder/f'{name}_diag.pt',weights_only=False)
        base=torch.load(ROOT/'predictions'/f'pathobench_{name}_{sys.argv[1]}_official50_bf16.pt',weights_only=False)
        assert diag['fold_indices']==base['fold_indices']==list(range(50))
        assert len(diag['rows'])==len(base['per_fold'])==50
        for d,b in zip(diag['rows'],base['per_fold']):
            assert d['slide_id']==b['slide_id'] and torch.equal(d['label'],b['label'])
            assert len(set(d['slide_id']))==len(d['slide_id'])
            assert set(d['probe'])=={'cv','bm','qa','ds'}
            for branch in ('cv','bm','bd','qa','ds'):
                torch.testing.assert_close(d['baseline'][branch],b['m_'+branch],rtol=0,atol=0)
            err=np.max(np.abs(combine(d['baseline'])-b['probability'].numpy()))
            checks['max_ensemble_replay_error']=max(checks['max_ensemble_replay_error'],float(err))
            assert err<=1e-6
            for branch,g in d['probe'].items():
                checks['max_replay_error']=max(checks['max_replay_error'],g['replay_error'])
                checks['max_path_error']=max(checks['max_path_error'],g['path_error'])
                assert g['replay_error']<=1e-5 and g['path_error']<=1e-5
                assert all(torch.isfinite(m).all() for m in g['margins'])
                eigen=g['eigenvalues'].numpy()
                positive=eigen[eigen>g['rank_tolerance']]
                assert len(positive)==g['rank']
                assert abs(positive.mean()-g['scale'])<1e-8
                for j,factor in enumerate((.1,1,10)):
                    assert abs(np.clip(g['scale']*factor,1e-4,1e4)-g['penalties'][j])<1e-8
                    assert abs((eigen/(eigen+g['penalties'][j])).sum()-g['df_probe'][j])<1e-8
        checks['task_coverage'][task]=50
        data[task]=diag['rows']
    for branch in ('cv','bm','qa','ds'):
        gs=[d['probe'][branch] for rows in data.values() for d in rows]
        checks['geometry'][branch]={
            'scale_median':float(np.median([g['scale'] for g in gs])),
            'df_fraction_mean':float(np.mean([g['df_baseline']/g['rank'] for g in gs])),
            'lambda_range':[min(min(g['penalties']) for g in gs),max(max(g['penalties']) for g in gs)],
            'clipped_penalties':sum(v in (1e-4,1e4) for g in gs for v in g['penalties'])}
        for j,factor in enumerate((.1,1.,10.)):
            per_task={}
            rhos=[]
            for task,rows in data.items():
                ensemble_delta=[];single_delta=[]
                for d in rows:
                    labels=d['label'].numpy();base=d['baseline'];changed=d['probe'][branch]['margins'][j]
                    ensemble_delta.append(auc(combine({**base,branch:changed}),labels)-auc(combine(base),labels))
                    single_delta.append(auc(changed.numpy(),labels)-auc(base[branch].numpy(),labels))
                    x,y=rankdata(base[branch].numpy()),rankdata(changed.numpy())
                    rhos.append(float(np.corrcoef(x,y)[0,1]))
                per_task[task]={'ensemble_delta':float(np.mean(ensemble_delta)),
                                'single_delta':float(np.mean(single_delta))}
            v=np.array([r['ensemble_delta'] for r in per_task.values()]); half=t.ppf(.975,6)*v.std(ddof=1)/np.sqrt(7)
            existing=next(c for c in summary['contrasts'] if c['branch']==branch and c['factor']==factor)
            error=max(abs(v.mean()-existing['mean_delta']),
                      *[abs(r['ensemble_delta']-existing['tasks'][k]['ensemble_delta']) for k,r in per_task.items()],
                      abs(v.mean()-half-existing['cluster_ci95'][0]),abs(v.mean()+half-existing['cluster_ci95'][1]),
                      abs(np.mean([r['single_delta'] for r in per_task.values()])-existing['mean_single_delta']))
            rank_error=abs(np.mean(rhos)-existing['mean_spearman'])
            checks['max_summary_delta_error']=max(checks['max_summary_delta_error'],error)
            checks['max_summary_rank_error']=max(checks['max_summary_rank_error'],rank_error)
            assert error<1e-10,(branch,factor,error)
            assert rank_error<1e-10
            checks['contrasts'].append({'branch':branch,'factor':factor,'tasks':per_task,
                'single_sign_agreement':sum(r['single_delta']>0 for r in per_task.values())})
    checks['status']='passed'
    (folder/'audit.json').write_text(json.dumps(checks,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in checks.items() if k not in ('contrasts','source_hashes','fold_hashes')},indent=2))

if __name__=='__main__':main()
