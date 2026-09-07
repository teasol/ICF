"""RU-82 — offline re-aggregation of RU-81 stored margins into lambda ensembles.

Diagnostic only; exploratory. Four predeclared configurations, all reported.
No GPU, no new folds, no hold-out access. See docs/ru/RU-82.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.ru81_launch import TASKS
from scripts.analysis.ru81_probe import ensemble as ru81_ensemble
from src.utils.metrics import auroc

SOURCE_TAG = 'ru81_reg_20260907_r1'
RIDGE_BRANCHES = ('cv', 'bm', 'qa', 'ds')   # bd carries no ridge penalty
ALL_BRANCHES = ('cv', 'bm', 'bd', 'qa', 'ds')
GATE_CONTRASTS = (('cv', 1.0), ('bm', 10.0), ('ds', 0.1))
TOL = 1e-6


def trimmed_mean(probs):
    """Drop min and max member, average the rest. Matches ru81_probe.ensemble."""
    stacked = torch.stack(probs)
    return stacked.sort(dim=0).values[1:-1].mean(dim=0)


def members(row, branch, standardize):
    """4-point lambda grid for a ridge branch, 1 member for bd."""
    margins = [row['baseline'][branch].float()]
    if branch in RIDGE_BRANCHES:
        margins += [m.float() for m in row['probe'][branch]['margins']]
    out = []
    for m in margins:
        if standardize:
            m = m / m.square().mean().sqrt().clamp_min(1e-12)
        out.append(torch.sigmoid(m))
    return out


def config_score(row, axis, standardize):
    per_branch = {b: members(row, b, standardize) for b in ALL_BRANCHES}
    if axis == 'A1':                      # average within branch, then 5-member trim
        return trimmed_mean([torch.stack(per_branch[b]).mean(dim=0) for b in ALL_BRANCHES])
    pooled = [p for b in ALL_BRANCHES for p in per_branch[b]]   # A2: single 17-member trim
    return trimmed_mean(pooled)


def auroc_pairwise(scores, labels):
    """Independent all-pairs AUROC with tie=0.5, for cross-checking auroc()."""
    s = scores.double().numpy()
    y = labels.long().numpy()
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (pos.size * neg.size))


def cluster_interval(task_means):
    d = np.array(task_means, dtype=float)
    mean = d.mean()
    half = t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(mean), [float(mean - half), float(mean + half)], float(d.std(ddof=1))


def main():
    started = time.time()
    folder = ROOT / 'predictions' / SOURCE_TAG
    prior = json.loads((folder / 'summary.json').read_text())
    data = {task: torch.load(folder / f'{task.replace("/", "_")}_diag.pt',
                             weights_only=False)['rows'] for task in TASKS}
    if any(len(rows) != 50 for rows in data.values()):
        raise SystemExit('KILL 2: incomplete diagnostic set')

    result = {'ru': 'RU-82', 'source_tag': SOURCE_TAG, 'type': 'diagnostic (exploratory)',
              'scope': 'offline re-aggregation; hold-out unverified; no promotion judgment',
              'gate': {}, 'configs': [], 'crosscheck': {}}

    # --- KILL 1/2: integrity gate against the RU-81 stored summary -----------
    gate_err = []
    for task, rows in data.items():
        for r in rows:
            n = len(r['slide_id'])
            if n != r['label'].numel():
                raise SystemExit(f'KILL 2: slide/label mismatch in {task}')
            for b in ALL_BRANCHES:
                if r['baseline'][b].numel() != n or not torch.isfinite(r['baseline'][b]).all():
                    raise SystemExit(f'KILL 2: bad baseline margin {task}/{b}')
            for b in RIDGE_BRANCHES:
                for m in r['probe'][b]['margins']:
                    if m.numel() != n or not torch.isfinite(m).all():
                        raise SystemExit(f'KILL 2: bad probe margin {task}/{b}')
        replay = float(np.mean([auroc(ru81_ensemble(r['baseline']), r['label']) for r in rows]))
        gate_err.append(abs(replay - prior['baseline'][task]))
    result['gate']['baseline_max_error'] = max(gate_err)

    contrast_err = []
    for branch, factor in GATE_CONTRASTS:
        j = [0.1, 1.0, 10.0].index(factor)
        task_means = []
        for task, rows in data.items():
            task_means.append(np.mean([
                auroc(ru81_ensemble({**r['baseline'], branch: r['probe'][branch]['margins'][j]}), r['label'])
                - auroc(ru81_ensemble(r['baseline']), r['label']) for r in rows]))
        recorded = next(c['mean_delta'] for c in prior['contrasts']
                        if c['branch'] == branch and c['factor'] == factor)
        contrast_err.append(abs(float(np.mean(task_means)) - recorded))
    result['gate']['contrast_max_error'] = max(contrast_err)
    if max(gate_err) > TOL or max(contrast_err) > TOL:
        raise SystemExit(f'KILL 1: replay error {max(gate_err):.3e} / {max(contrast_err):.3e} > {TOL}')

    # --- the four predeclared configurations --------------------------------
    cross = []
    for axis in ('A1', 'A2'):
        for scale in ('B1', 'B2'):
            standardize = (scale == 'B2')
            tasks = {}
            for task, rows in data.items():
                deltas = []
                for r in rows:
                    base = auroc(ru81_ensemble(r['baseline']), r['label'])
                    cand_score = config_score(r, axis, standardize)
                    if not torch.isfinite(cand_score).all():
                        raise SystemExit(f'KILL 2: nonfinite score {axis}x{scale} {task}')
                    cand = auroc(cand_score, r['label'])
                    cross.append(abs(cand - auroc_pairwise(cand_score, r['label'])))
                    deltas.append(float(cand - base))
                tasks[task] = {'mean_delta': float(np.mean(deltas)), 'n_folds': len(deltas)}
            task_means = [v['mean_delta'] for v in tasks.values()]
            mean, ci, sd = cluster_interval(task_means)
            if ci[0] > 0 and mean > 0 and int(np.sum(np.array(task_means) > 0)) >= 6:
                verdict = 'supported'
            elif ci[1] < 0:
                verdict = 'refuted'
            else:
                verdict = 'indeterminate'
            result['configs'].append({
                'config': f'{axis}x{scale}',
                'axis': 'within-branch lambda average' if axis == 'A1' else 'pooled 17-member trim',
                'scale': 'current sigmoid' if scale == 'B1' else 'margin RMS standardized',
                'n_members': 5 if axis == 'A1' else 17,
                'mean_delta': mean, 'cluster_ci95': ci, 'task_sd': sd,
                'sign_agreement': int(np.sum(np.array(task_means) > 0)),
                'regressions': [k for k, v in tasks.items() if v['mean_delta'] < 0],
                'verdict': verdict, 'tasks': tasks})

    result['crosscheck'] = {'max_auroc_difference': max(cross),
                            'note': 'same-script pairwise recomputation, not an independent audit'}
    result['wall_seconds'] = time.time() - started

    out = ROOT / 'predictions' / 'ru82_lambda_ens'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'configs'}, indent=2))
    for c in result['configs']:
        print(f"{c['config']}  Δ {100*c['mean_delta']:+.4f}%p  "
              f"CI [{100*c['cluster_ci95'][0]:+.4f}, {100*c['cluster_ci95'][1]:+.4f}]  "
              f"sign {c['sign_agreement']}/7  taskSD {100*c['task_sd']:.4f}  {c['verdict']}")


if __name__ == '__main__':
    main()
