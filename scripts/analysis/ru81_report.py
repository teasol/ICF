"""Report all predeclared RU-81 contrasts; no candidate promotion/selection."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.ru81_launch import TASKS
from scripts.analysis.ru81_probe import ensemble, FACTORS
from src.utils.metrics import auroc


def main():
    tag = sys.argv[1]
    folder = ROOT/'predictions'/tag
    data = {task:torch.load(folder/f'{task.replace("/","_")}_diag.pt', weights_only=False)['rows'] for task in TASKS}
    if any(len(v) != 50 for v in data.values()):
        raise ValueError('Incomplete diagnostic')
    result = {'tag':tag, 'scope':'diagnostic only; hold-out unverified; all 12 contrasts reported',
              'geometry':{}, 'contrasts':[], 'baseline':{}}
    for task, rows in data.items():
        result['baseline'][task] = float(np.mean([auroc(ensemble(r['baseline']),r['label']) for r in rows]))
    for b in ('cv','bm','qa','ds'):
        gs = [r['probe'][b] for rows in data.values() for r in rows]
        result['geometry'][b] = {'mean_scale':float(np.mean([g['scale'] for g in gs])),
            'mean_rank':float(np.mean([g['rank'] for g in gs])),
            'baseline_df_fraction':float(np.mean([g['df_baseline']/g['rank'] for g in gs])),
            'probe_df_fractions':[float(np.mean([g['df_probe'][j]/g['rank'] for g in gs])) for j in range(3)],
            'max_replay_error':max(g['replay_error'] for g in gs)}
        for j,factor in enumerate(FACTORS):
            tasks, ranks, scales = {}, [], []
            for task, rows in data.items():
                delta, single = [], []
                for r in rows:
                    baseline = r['baseline']
                    changed = r['probe'][b]['margins'][j]
                    delta.append(auroc(ensemble({**baseline,b:changed}),r['label'])-auroc(ensemble(baseline),r['label']))
                    single.append(auroc(changed,r['label'])-auroc(baseline[b],r['label']))
                    rho = spearmanr(baseline[b].numpy(),changed.numpy()).statistic
                    ranks.append(float(rho) if np.isfinite(rho) else None)
                    scales.append(float(changed.square().mean().sqrt()/baseline[b].square().mean().sqrt().clamp_min(1e-12)))
                tasks[task] = {'ensemble_delta':float(np.mean(delta)), 'single_delta':float(np.mean(single)),
                               'fold_deltas':delta}
            differences = np.array([x['ensemble_delta'] for x in tasks.values()])
            mean, se = differences.mean(), differences.std(ddof=1)/np.sqrt(7)
            half = t.ppf(.975,6)*se
            valid_ranks = [v for v in ranks if v is not None]
            result['contrasts'].append({'branch':b, 'factor':factor, 'mean_delta':float(mean),
                'cluster_ci95':[float(mean-half),float(mean+half)], 'sign_agreement':int((differences>0).sum()),
                'regressions':[k for k,v in tasks.items() if v['ensemble_delta']<0], 'tasks':tasks,
                'mean_spearman':float(np.mean(valid_ranks)) if valid_ranks else None,
                'undefined_spearman_count':len(ranks)-len(valid_ranks), 'mean_margin_rms_ratio':float(np.mean(scales)),
                'mean_single_delta':float(np.mean([v['single_delta'] for v in tasks.values()]))})
    (folder/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['RU-81 — diagnostic; hold-out unverified; no promotion judgment',
           '| branch | Gram scale | baseline df/rank |', '|---|---:|---:|']
    for b,g in result['geometry'].items():
        lines.append(f'| {b} | {g["mean_scale"]:.3f} | {g["baseline_df_fraction"]:.3f} |')
    lines.extend(['','| branch | factor | ensemble Δ %p | cluster 95% CI %p | sign /7 | single Δ %p | rank ρ | RMS ratio | regressions |',
                  '|---|---:|---:|---|---:|---:|---:|---:|---|'])
    for c in result['contrasts']:
        lo,hi=c['cluster_ci95']
        lines.append(f'| {c["branch"]} | {c["factor"]} | {100*c["mean_delta"]:+.4f} | [{100*lo:+.4f}, {100*hi:+.4f}] | {c["sign_agreement"]} | {100*c["mean_single_delta"]:+.4f} | {c["mean_spearman"]} | {c["mean_margin_rms_ratio"]:.3f} | {", ".join(c["regressions"])} |')
    report='\n'.join(lines)+'\n'
    (folder/'summary.md').write_text(report)
    print(report)


if __name__=='__main__':
    main()
