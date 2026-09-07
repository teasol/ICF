"""RU-83 — separate the rank effect from the magnitude effect in ensemble deltas.

Diagnostic only; exploratory. Two predeclared contrasts per (branch, factor) cell,
all 36 arms reported. No GPU, no new folds, no hold-out access. See docs/ru/RU-83.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.ru81_launch import TASKS
from scripts.analysis.ru81_probe import FACTORS, ensemble
from src.utils.metrics import auroc

SOURCE_TAG = 'ru81_reg_20260907_r1'
RIDGE_BRANCHES = ('cv', 'bm', 'qa', 'ds')
ALL_BRANCHES = ('cv', 'bm', 'bd', 'qa', 'ds')
GATE_CONTRASTS = (('cv', 1.0), ('bm', 10.0), ('ds', 0.1))
TOL = 1e-6
INV_TOL = 1e-9


def rms(x):
    return x.double().square().mean().sqrt()


def size_contrast(base, probe):
    """Rank-preserving: keep the baseline ordering, match the probe magnitude."""
    return (base.double() * (rms(probe) / rms(base).clamp_min(1e-12))).float()


def rank_contrast(base, probe):
    """Multiset-preserving: keep the baseline value set, follow the probe ordering."""
    order = torch.argsort(probe.double(), stable=True)          # probe rank of each slide
    out = torch.empty_like(base.double())
    out[order] = torch.sort(base.double(), stable=True).values  # ascending -> ascending
    return out.float()


def check_invariants(base, probe, m_size, m_rank, labels, report):
    """Exact invariants are kill conditions; tie-induced deviations are reported.

    The card's original 'Spearman(probe, m_rank) == 1' was unattainable whenever the
    baseline margin carries ties (see the RU-83 card revision), so the ordering check
    is stated exactly instead: sorted by probe, m_rank must be non-decreasing.
    """
    rho_size = spearmanr(base.numpy(), m_size.numpy()).statistic
    if np.isfinite(rho_size) and abs(rho_size - 1.0) > INV_TOL:
        raise SystemExit(f'KILL 2: size contrast broke the ordering (rho={rho_size})')
    by_probe = m_rank.double()[torch.argsort(probe.double(), stable=True)]
    if float((by_probe[1:] - by_probe[:-1]).min()) < -INV_TOL:
        raise SystemExit('KILL 2: rank contrast is not non-decreasing in the probe')
    ratio = float(rms(m_size) / rms(probe).clamp_min(1e-12))
    if abs(ratio - 1.0) > TOL:
        raise SystemExit(f'KILL 2: size contrast magnitude mismatch ({ratio})')
    if float((torch.sort(m_rank.double()).values - torch.sort(base.double()).values).abs().max()) > 0.0:
        raise SystemExit('KILL 2: rank contrast changed the value multiset')
    a_base, a_probe = auroc(base, labels), auroc(probe, labels)
    if abs(auroc(m_size, labels) - a_base) > INV_TOL:
        raise SystemExit('KILL 2: size contrast changed the single-branch AUROC')
    rho_rank = spearmanr(probe.numpy(), m_rank.numpy()).statistic
    report['ties_base'] += int(len(base) - torch.unique(base).numel())
    report['ties_probe'] += int(len(probe) - torch.unique(probe).numel())
    report['folds_with_base_ties'] += int(len(base) != torch.unique(base).numel())
    if np.isfinite(rho_rank):
        report['min_rank_spearman'] = min(report['min_rank_spearman'], float(rho_rank))
    report['max_rank_auroc_shift'] = max(report['max_rank_auroc_shift'],
                                         abs(auroc(m_rank, labels) - a_probe))


def cluster(task_means):
    d = np.array(task_means, dtype=float)
    mean = d.mean()
    half = t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(mean), [float(mean - half), float(mean + half)], int((d > 0).sum())


def main():
    started = time.time()
    folder = ROOT / 'predictions' / SOURCE_TAG
    prior = json.loads((folder / 'summary.json').read_text())
    data = {task: torch.load(folder / f'{task.replace("/", "_")}_diag.pt',
                             weights_only=False)['rows'] for task in TASKS}
    if any(len(rows) != 50 for rows in data.values()):
        raise SystemExit('KILL 3: incomplete diagnostic set')

    # --- KILL 1: integrity gate against the RU-81 stored summary ------------
    gate = []
    for task, rows in data.items():
        gate.append(abs(float(np.mean([auroc(ensemble(r['baseline']), r['label'])
                                       for r in rows])) - prior['baseline'][task]))
    for branch, factor in GATE_CONTRASTS:
        j = FACTORS.index(factor)
        means = [np.mean([auroc(ensemble({**r['baseline'], branch: r['probe'][branch]['margins'][j]}),
                                r['label']) - auroc(ensemble(r['baseline']), r['label'])
                          for r in rows]) for rows in data.values()]
        recorded = next(c['mean_delta'] for c in prior['contrasts']
                        if c['branch'] == branch and c['factor'] == factor)
        gate.append(abs(float(np.mean(means)) - recorded))
    if max(gate) > TOL:
        raise SystemExit(f'KILL 1: replay error {max(gate):.3e} > {TOL}')

    result = {'ru': 'RU-83', 'source_tag': SOURCE_TAG, 'type': 'diagnostic (exploratory)',
              'scope': 'offline re-aggregation; hold-out unverified; no promotion judgment',
              'gate_max_error': max(gate),
              'side_effects': {'ties_base': 0, 'ties_probe': 0, 'folds_with_base_ties': 0,
                               'min_rank_spearman': 1.0, 'max_rank_auroc_shift': 0.0},
              'cells': []}

    for branch in RIDGE_BRANCHES:
        for j, factor in enumerate(FACTORS):
            arms = {'probe': {}, 'size': {}, 'rank': {}}
            for task, rows in data.items():
                acc = {k: [] for k in arms}
                for r in rows:
                    base, labels = r['baseline'][branch], r['label']
                    probe = r['probe'][branch]['margins'][j]
                    m_size = size_contrast(base, probe)
                    m_rank = rank_contrast(base, probe)
                    check_invariants(base, probe, m_size, m_rank, labels, result['side_effects'])
                    ref = auroc(ensemble(r['baseline']), labels)
                    for name, margin in (('probe', probe), ('size', m_size), ('rank', m_rank)):
                        score = ensemble({**r['baseline'], branch: margin})
                        if not torch.isfinite(score).all():
                            raise SystemExit(f'KILL 3: nonfinite ensemble {branch}x{factor} {name}')
                        acc[name].append(float(auroc(score, labels) - ref))
                for name in arms:
                    arms[name][task] = float(np.mean(acc[name]))
            cell = {'branch': branch, 'factor': factor}
            for name in arms:
                mean, ci, sign = cluster(list(arms[name].values()))
                cell[name] = {'mean_delta': mean, 'cluster_ci95': ci, 'sign_agreement': sign,
                              'tasks': arms[name]}
            d_p, d_s, d_r = (cell[k]['mean_delta'] for k in ('probe', 'size', 'rank'))
            cell['size_dominates'] = bool(abs(d_s) > abs(d_r))
            cell['additivity_residual'] = float(d_p - (d_s + d_r))
            cell['additive'] = bool(abs(cell['additivity_residual'])
                                    <= 0.5 * max(abs(d_s), abs(d_r), 1e-12))
            lo_s, hi_s = cell['size']['cluster_ci95']
            lo_r, hi_r = cell['rank']['cluster_ci95']
            cell['ci_overlap'] = bool(lo_s <= hi_r and lo_r <= hi_s)
            result['cells'].append(cell)

    n = len(result['cells'])
    dom = sum(c['size_dominates'] for c in result['cells'])
    add = sum(c['additive'] for c in result['cells'])
    strong = [c for c in result['cells'] if c['factor'] == 10.0]
    dom10 = sum(c['size_dominates'] for c in strong)

    def call(hit, total, hi, lo):
        return 'supported' if hit >= hi else ('refuted' if hit <= lo else 'indeterminate')

    result['hypotheses'] = {
        'H1_size_dominates': {'hits': dom, 'of': n, 'verdict': call(dom, n, 7, 5)},
        'H2_additive': {'hits': add, 'of': n, 'verdict': call(add, n, 7, 5)},
        'H3_strong_shrinkage': {'hits': dom10, 'of': len(strong),
                                'verdict': call(dom10, len(strong), 4, 2)},
        'ci_overlap_cells': sum(c['ci_overlap'] for c in result['cells']),
    }
    result['wall_seconds'] = time.time() - started

    out = ROOT / 'predictions' / 'ru83_rank_size'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, indent=2))
    print(f"\n{'cell':10s} {'Δprobe':>10s} {'Δsize':>10s} {'Δrank':>10s} {'resid':>9s}  dom  add  ciOv")
    for c in result['cells']:
        print(f"{c['branch']}x{c['factor']:<7} {100*c['probe']['mean_delta']:+10.4f} "
              f"{100*c['size']['mean_delta']:+10.4f} {100*c['rank']['mean_delta']:+10.4f} "
              f"{100*c['additivity_residual']:+9.4f}  "
              f"{'S' if c['size_dominates'] else 'R'}    {'Y' if c['additive'] else 'n'}    "
              f"{'Y' if c['ci_overlap'] else 'n'}")


if __name__ == '__main__':
    main()
