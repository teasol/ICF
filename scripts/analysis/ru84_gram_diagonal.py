"""RU-84 — is the RBF Gram diagonally dominant at the current context scale?

Reproduction measurement for the CA-04 rejection mechanism claimed in section 199.
Reads only the stored geometry from the RU-81 run; no GPU, no new folds.
See docs/ru/RU-84.json.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.ru81_launch import TASKS

SOURCE_TAG = 'ru81_reg_20260907_r1'
JUDGED = ('bm', 'qa', 'ds')      # KRR path: solver standardises, gamma = 1/dims
REPORTED_ONLY = ('cv',)          # different normalisation path, reported not judged
RHO_BOUNDS = (0.5, 2.0)


def main():
    started = time.time()
    folder = ROOT / 'predictions' / SOURCE_TAG
    rows = [r for task in TASKS
            for r in torch.load(folder / f'{task.replace("/", "_")}_diag.pt',
                                weights_only=False)['rows']]
    if len(rows) != 350:
        raise SystemExit(f'KILL 2: expected 350 folds, got {len(rows)}')

    result = {'ru': 'RU-84', 'source_tag': SOURCE_TAG, 'type': 'reproduction_measurement',
              'scope': 'stored geometry only; RBF kernel; cosine and poly not measured',
              'gamma_rule': 'krr_gamma=None -> gamma = 1/dims (solvers.kernel_matrix)',
              'branches': {}, 'n_folds': len(rows)}

    for branch in JUDGED + REPORTED_ONLY:
        rho, k_lo, s_lo, nctx = [], [], [], []
        for r in rows:
            g = r['probe'][branch]
            for key in ('eigenvalues', 'dimension', 'n_context'):
                if key not in g:
                    raise SystemExit(f'KILL 2: missing {key} for {branch}')
            trace = float(g['eigenvalues'].double().sum())
            if not math.isfinite(trace):
                raise SystemExit(f'KILL 2: nonfinite trace for {branch}')
            variance = trace / 2.0                       # class weights sum to 2
            ratio = variance / g['dimension']
            rho.append(ratio)
            k_lo.append(math.exp(-2.0 * ratio))          # Jensen lower bound
            s_lo.append((g['n_context'] - 1) * k_lo[-1])
            nctx.append(g['n_context'])
        rho, k_lo, s_lo, nctx = map(np.array, (rho, k_lo, s_lo, nctx))
        entry = {'judged': branch in JUDGED,
                 'dimension': int(rows[0]['probe'][branch]['dimension']),
                 'n_context_range': [int(nctx.min()), int(nctx.max())],
                 'rho_std': {'mean': float(rho.mean()), 'min': float(rho.min()),
                             'max': float(rho.max())},
                 'offdiag_kernel_lower_bound': {'mean': float(k_lo.mean()),
                                                'min': float(k_lo.min())},
                 'offdiag_rowsum_lower_bound': {'mean': float(s_lo.mean()),
                                                'min': float(s_lo.min()),
                                                'max': float(s_lo.max())},
                 'folds_where_diagonal_could_dominate': int((s_lo <= 1.0).sum())}
        if branch in JUDGED and not (RHO_BOUNDS[0] <= rho.mean() <= RHO_BOUNDS[1]):
            raise SystemExit(f'KILL 1: rho_std {rho.mean():.3f} outside {RHO_BOUNDS} for {branch}')
        result['branches'][branch] = entry

    judged = [result['branches'][b] for b in JUDGED]
    could = sum(e['folds_where_diagonal_could_dominate'] for e in judged)
    result['verdict'] = {
        'diagonal_dominance': 'refuted' if could == 0 else 'possible_in_some_folds',
        'folds_where_diagonal_could_dominate': could,
        'of_folds_judged': len(rows) * len(JUDGED),
        'worst_case_rowsum_lower_bound': min(e['offdiag_rowsum_lower_bound']['min'] for e in judged),
        'at_n_ctx_40': 39 * math.exp(-2.0),
        'note': ('The bound holds for any dimension, correlation structure and N_ctx: '
                 'per-feature standardisation gives E||xi-xj||^2 = 2d and gamma = 1/d, '
                 'so E[gamma*d^2] = 2 and Jensen gives mean off-diagonal >= exp(-2).'),
    }
    result['wall_seconds'] = time.time() - started

    out = ROOT / 'predictions' / 'ru84_gram_diagonal'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
