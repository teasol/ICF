"""Head aggregation strategies for combining branch margins into one logit.

The probability-space math of each aggregation (everything after the
(weight, margin)/(margin, loo-margin) pool has been resolved) is factored
into the `_..._from_...` helpers below. They are the single source of truth
for these five formulas and are imported directly by
`scripts/test_pathobench.py`'s fixed-head evaluation path, which needs the
same math but returns a probability instead of a logit and resolves its own
(weight, margin) pool from `ICF_FIXED_HEAD_*` margins. This closes the
"집계 로직 이원화" technical debt (docs/current_status.md item 6): before this
refactor the two paths carried independently written copies of the same
five formulas and had already drifted once (BS/SH/SJ, see D-04x history).
`tests/test_bs_branch.py::TestPathobenchAndVotingAgree` pins the two paths
to agree on the same inputs.
"""

from __future__ import annotations

import torch

from src.models.common.solvers import fast_context_auroc

#: The fixed branch set, in the canonical combination order. Each aggregation
#: resolves (weight, margin) pairs through _fixed_branch_pairs so the on/off and
#: weighting rules live in exactly one place.
_FIXED_BRANCHES = ("dd", "ct", "bm", "bd", "qa", "ds", "lr", "de", "sw")


def _fixed_branch_pairs(config, margins: dict[str, torch.Tensor | None]) -> list[tuple[float, torch.Tensor]]:
    """(weight, margin) pairs of the fixed branches whose margin exists and whose
    configured weight is non-zero, in _FIXED_BRANCHES order."""
    pairs = []
    for name in _FIXED_BRANCHES:
        m = margins.get(name)
        if m is None:
            continue
        w = getattr(config, f"weight_{name}", 0.0)
        if w != 0.0:
            pairs.append((w, m))
    return pairs


def _shape_branch_pairs(config, m_sj, m_sh, m_shj=None, m_bs=None) -> list[tuple[float, torch.Tensor]]:
    """(weight, margin) pairs of the shape-family branches, in (SJ, SH, BS) order.

    m_shj is the legacy margin alias of m_sj (D-038): it is only consulted when
    m_sj is absent, and the SJ weight falls back to weight_shj the same way.
    """
    pairs = []
    m_sj_active = m_sj if m_sj is not None else m_shj
    w_sj = getattr(config, "weight_sj", getattr(config, "weight_shj", 0.0))
    if m_sj_active is not None and w_sj != 0.0:
        pairs.append((w_sj, m_sj_active))
    w_sh = getattr(config, "weight_sh", 0.0)
    if m_sh is not None and w_sh != 0.0:
        pairs.append((w_sh, m_sh))
    w_bs = getattr(config, "weight_bs", 0.0)
    if m_bs is not None and w_bs != 0.0:
        pairs.append((w_bs, m_bs))
    return pairs


def _to_logit(probability: torch.Tensor) -> torch.Tensor:
    """Clamp a probability into (0, 1) and convert to a logit."""
    clamped = probability.clamp(1e-7, 1.0 - 1e-7)
    return torch.log(clamped / (1.0 - clamped))


def _trimmed_mean_from_probs(probs: list[torch.Tensor]) -> torch.Tensor:
    """Core Trimmed Mean math on an already-resolved, non-empty probability
    list (drops one min and one max member when there are >= 3). Returns a
    probability, not a logit -- shared with scripts/test_pathobench.py."""
    stacked = torch.stack(probs, dim=-1)
    if len(probs) >= 3:
        sum_p = torch.sum(stacked, dim=-1)
        min_p = torch.min(stacked, dim=-1).values
        max_p = torch.max(stacked, dim=-1).values
        return (sum_p - min_p - max_p) / float(len(probs) - 2)
    return torch.mean(stacked, dim=-1)


def _soft_voting_from_pairs(pairs: list[tuple[float, torch.Tensor]]) -> torch.Tensor:
    """Core Soft Voting math on an already-resolved, non-empty (weight,
    margin) pair list. Returns a probability -- shared with
    scripts/test_pathobench.py."""
    total_weight = sum(w for w, _ in pairs)
    return sum(w * torch.sigmoid(m) for w, m in pairs) / total_weight


def _hard_gated_from_probs(probs: list[torch.Tensor], tau: float) -> torch.Tensor:
    """Core Hard Gated Voting math on an already-resolved, non-empty
    probability list. Returns a probability -- shared with
    scripts/test_pathobench.py."""
    stacked = torch.stack(probs, dim=-1)  # [N, B]
    c = (stacked - 0.5).abs()
    mask = (c >= tau).float()
    has_active = (mask.sum(dim=-1, keepdim=True) > 0)
    weights = torch.where(has_active, mask, torch.ones_like(mask))
    return (weights * stacked).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def _adaptive_trimmed_from_probs(probs: list[torch.Tensor], tau: float, ratio: float) -> torch.Tensor:
    """Core Adaptive Trimmed Mean math on an already-resolved, non-empty
    probability list. Returns a probability -- shared with
    scripts/test_pathobench.py."""
    stacked = torch.stack(probs, dim=-1)  # [N, B]
    B = stacked.shape[-1]
    if B < 3:
        return stacked.mean(dim=-1)

    sorted_p, _ = torch.sort(stacked, dim=-1)
    c = (stacked - 0.5).abs()
    c_med = torch.median(c, dim=-1).values
    min_p = sorted_p[:, 0]
    max_p = sorted_p[:, -1]
    c_min = (min_p - 0.5).abs()
    c_max = (max_p - 0.5).abs()

    drop_min = (c_min <= ratio * c_med) | (c_min <= tau)
    drop_max = (c_max <= ratio * c_med) | (c_max <= tau)

    sum_all = sorted_p.sum(dim=-1)
    count_all = torch.full_like(sum_all, float(B))
    sum_trimmed = (
        sum_all
        - torch.where(drop_min, min_p, torch.zeros_like(min_p))
        - torch.where(drop_max, max_p, torch.zeros_like(max_p))
    )
    count_trimmed = count_all - drop_min.float() - drop_max.float()
    return sum_trimmed / count_trimmed.clamp_min(1.0)


def _context_loo_from_pool(
    branch_pool: list[tuple[torch.Tensor, torch.Tensor | None]],
    context_labels: torch.Tensor,
    gamma: float,
    floor: float,
) -> torch.Tensor:
    """Core Context-LOO stacking math on an already-resolved, non-empty
    (margin, loo_margin) pool. Returns a probability -- shared with
    scripts/test_pathobench.py."""
    r_list = []
    for q_m, l_m in branch_pool:
        r = fast_context_auroc(l_m, context_labels) if l_m is not None else 0.50
        r_list.append(r)

    q_list = [max(0.0, r - floor) ** gamma for r in r_list]
    sum_q = sum(q_list)
    if sum_q > 0:
        weights = [q / sum_q for q in q_list]
    else:
        weights = [1.0 / len(branch_pool)] * len(branch_pool)

    return sum(w * torch.sigmoid(q_m) for w, (q_m, _) in zip(weights, branch_pool))


def linear_aggregation(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw, m_sj=None, m_sh=None, m_shj=None, m_bs=None):
    total_margin = config.weight_cv * m_cv
    pairs = _fixed_branch_pairs(config, {
        "dd": m_dd, "ct": m_ct, "bm": m_bm, "bd": m_bd, "qa": m_qa,
        "ds": m_ds, "lr": m_lr, "de": m_de, "sw": m_sw,
    }) + _shape_branch_pairs(config, m_sj, m_sh, m_shj, m_bs)
    for w, m in pairs:
        total_margin = total_margin + w * m
    return total_margin


def soft_voting(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw, m_sj=None, m_sh=None, m_shj=None, m_bs=None):
    active_pairs = []
    if config.weight_cv != 0.0:
        active_pairs.append((config.weight_cv, m_cv))
    active_pairs += _fixed_branch_pairs(config, {
        "dd": m_dd, "ct": m_ct, "bm": m_bm, "bd": m_bd, "qa": m_qa,
        "ds": m_ds, "lr": m_lr, "de": m_de, "sw": m_sw,
    })
    active_pairs += _shape_branch_pairs(config, m_sj, m_sh, m_shj, m_bs)

    if not active_pairs:
        return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

    avg_prob = _soft_voting_from_pairs(active_pairs)
    return _to_logit(avg_prob)


def context_loo_stacking(
    config,
    cv,
    context_labels,
    m_cv, loo_cv,
    m_dd, loo_dd,
    m_ct, loo_ct,
    m_bm, loo_bm,
    m_bd, loo_bd,
    m_qa, loo_qa,
    m_ds, loo_ds,
    m_de, loo_de,
    m_sw, loo_sw,
    m_sj=None, loo_sj=None,
    m_sh=None, loo_sh=None,
    m_shj=None, loo_shj=None,
    m_bs=None, loo_bs=None,
):
    if m_sj is None and m_shj is not None:
        m_sj, loo_sj = m_shj, loo_shj  # Legacy margin alias (D-038)

    candidates = [
        ("dd", m_dd, loo_dd),
        ("ct", m_ct, loo_ct),
        ("bm", m_bm, loo_bm),
        ("bd", m_bd, loo_bd),
        ("qa", m_qa, loo_qa),
        ("ds", m_ds, loo_ds),
        ("de", m_de, loo_de),
        ("sw", m_sw, loo_sw),
    ]
    branch_pool = []
    if config.weight_cv != 0.0:
        branch_pool.append((m_cv, loo_cv))
    for name, m, l in candidates:
        if m is None:
            continue
        w = getattr(config, f"weight_{name}", 0.0)
        if w != 0.0:
            branch_pool.append((m, l))
    w_sj = getattr(config, "weight_sj", getattr(config, "weight_shj", 0.0))
    if m_sj is not None and w_sj != 0.0:
        branch_pool.append((m_sj, loo_sj))
    w_sh = getattr(config, "weight_sh", 0.0)
    if m_sh is not None and w_sh != 0.0:
        branch_pool.append((m_sh, loo_sh))
    w_bs = getattr(config, "weight_bs", 0.0)
    if m_bs is not None and w_bs != 0.0:
        branch_pool.append((m_bs, loo_bs))

    if not branch_pool:
        return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

    gamma = getattr(config, "loo_gamma", 2.0)
    floor = getattr(config, "loo_floor", 0.50)
    avg_prob = _context_loo_from_pool(branch_pool, context_labels, gamma, floor)
    return _to_logit(avg_prob)


def trimmed_mean(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw, m_sj=None, m_sh=None, m_shj=None, m_bs=None):
    active_pairs = []
    if config.weight_cv != 0.0:
        active_pairs.append((config.weight_cv, m_cv))
    active_pairs += _fixed_branch_pairs(config, {
        "dd": m_dd, "ct": m_ct, "bm": m_bm, "bd": m_bd, "qa": m_qa,
        "ds": m_ds, "lr": m_lr, "de": m_de, "sw": m_sw,
    })
    active_pairs += _shape_branch_pairs(config, m_sj, m_sh, m_shj, m_bs)
    active_probs = [torch.sigmoid(m) for _, m in active_pairs]

    if not active_probs:
        return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

    avg_p = _trimmed_mean_from_probs(active_probs)
    return _to_logit(avg_p)


def hard_gated(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw, m_sj=None, m_sh=None, m_shj=None, m_bs=None):
    active_pairs = []
    if config.weight_cv != 0.0:
        active_pairs.append((config.weight_cv, m_cv))
    active_pairs += _fixed_branch_pairs(config, {
        "dd": m_dd, "ct": m_ct, "bm": m_bm, "bd": m_bd, "qa": m_qa,
        "ds": m_ds, "lr": m_lr, "de": m_de, "sw": m_sw,
    })
    active_pairs += _shape_branch_pairs(config, m_sj, m_sh, m_shj, m_bs)
    active_probs = [torch.sigmoid(m) for _, m in active_pairs]

    if not active_probs:
        return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

    tau = getattr(config, "gated_tau", 0.05)
    avg_p = _hard_gated_from_probs(active_probs, tau)
    return _to_logit(avg_p)


def adaptive_trimmed(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw, m_sj=None, m_sh=None, m_shj=None, m_bs=None):
    active_pairs = []
    if config.weight_cv != 0.0:
        active_pairs.append((config.weight_cv, m_cv))
    active_pairs += _fixed_branch_pairs(config, {
        "dd": m_dd, "ct": m_ct, "bm": m_bm, "bd": m_bd, "qa": m_qa,
        "ds": m_ds, "lr": m_lr, "de": m_de, "sw": m_sw,
    })
    active_pairs += _shape_branch_pairs(config, m_sj, m_sh, m_shj, m_bs)
    active_probs = [torch.sigmoid(m) for _, m in active_pairs]

    if not active_probs:
        return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

    tau = getattr(config, "adaptive_tau", 0.08)
    ratio = getattr(config, "adaptive_ratio", 1.5)
    avg_p = _adaptive_trimmed_from_probs(active_probs, tau, ratio)
    return _to_logit(avg_p)
