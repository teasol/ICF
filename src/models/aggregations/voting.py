"""Head aggregation strategies for combining branch margins into one logit."""

from __future__ import annotations

import torch

from src.models.common.solvers import fast_context_auroc


def linear_aggregation(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw):
    total_margin = config.weight_cv * m_cv
    if m_dd is not None:
        total_margin = total_margin + config.weight_dd * m_dd
    if m_ct is not None:
        total_margin = total_margin + config.weight_ct * m_ct
    if m_bm is not None:
        total_margin = total_margin + config.weight_bm * m_bm
    if m_bd is not None:
        total_margin = total_margin + config.weight_bd * m_bd
    if m_qa is not None:
        total_margin = total_margin + config.weight_qa * m_qa
    if m_ds is not None:
        total_margin = total_margin + config.weight_ds * m_ds
    if m_lr is not None:
        total_margin = total_margin + config.weight_lr * m_lr
    if m_de is not None:
        total_margin = total_margin + config.weight_de * m_de
    if m_sw is not None:
        total_margin = total_margin + config.weight_sw * m_sw
    return total_margin


def soft_voting(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw):
    active_pairs = []
    if config.weight_cv != 0.0:
        active_pairs.append((config.weight_cv, m_cv))
    if m_dd is not None and config.weight_dd != 0.0:
        active_pairs.append((config.weight_dd, m_dd))
    if m_ct is not None and config.weight_ct != 0.0:
        active_pairs.append((config.weight_ct, m_ct))
    if m_bm is not None and config.weight_bm != 0.0:
        active_pairs.append((config.weight_bm, m_bm))
    if m_bd is not None and config.weight_bd != 0.0:
        active_pairs.append((config.weight_bd, m_bd))
    if m_qa is not None and config.weight_qa != 0.0:
        active_pairs.append((config.weight_qa, m_qa))
    if m_ds is not None and config.weight_ds != 0.0:
        active_pairs.append((config.weight_ds, m_ds))
    if m_lr is not None and config.weight_lr != 0.0:
        active_pairs.append((config.weight_lr, m_lr))
    if m_de is not None and config.weight_de != 0.0:
        active_pairs.append((config.weight_de, m_de))
    if m_sw is not None and config.weight_sw != 0.0:
        active_pairs.append((config.weight_sw, m_sw))

    if not active_pairs:
        return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

    total_weight = sum(w for w, _ in active_pairs)
    avg_prob = sum(w * torch.sigmoid(m) for w, m in active_pairs) / total_weight
    clamped = avg_prob.clamp(1e-7, 1.0 - 1e-7)
    return torch.log(clamped / (1.0 - clamped))


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
):
    branch_pool = []
    if config.weight_cv != 0.0:
        branch_pool.append((m_cv, loo_cv))
    if m_dd is not None and config.weight_dd != 0.0:
        branch_pool.append((m_dd, loo_dd))
    if m_ct is not None and config.weight_ct != 0.0:
        branch_pool.append((m_ct, loo_ct))
    if m_bm is not None and config.weight_bm != 0.0:
        branch_pool.append((m_bm, loo_bm))
    if m_bd is not None and config.weight_bd != 0.0:
        branch_pool.append((m_bd, loo_bd))
    if m_qa is not None and config.weight_qa != 0.0:
        branch_pool.append((m_qa, loo_qa))
    if m_ds is not None and config.weight_ds != 0.0:
        branch_pool.append((m_ds, loo_ds))
    if m_de is not None and config.weight_de != 0.0:
        branch_pool.append((m_de, loo_de))
    if m_sw is not None and config.weight_sw != 0.0:
        branch_pool.append((m_sw, loo_sw))

    if not branch_pool:
        return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

    r_list = []
    for q_m, l_m in branch_pool:
        if l_m is not None:
            r = fast_context_auroc(l_m, context_labels)
        else:
            r = 0.50
        r_list.append(r)

    gamma = getattr(config, "loo_gamma", 2.0)
    floor = getattr(config, "loo_floor", 0.50)
    q_list = [max(0.0, r - floor) ** gamma for r in r_list]
    sum_q = sum(q_list)
    if sum_q > 0:
        weights = [q / sum_q for q in q_list]
    else:
        weights = [1.0 / len(branch_pool)] * len(branch_pool)

    avg_prob = sum(w * torch.sigmoid(q_m) for w, (q_m, _) in zip(weights, branch_pool))
    clamped = avg_prob.clamp(1e-7, 1.0 - 1e-7)
    return torch.log(clamped / (1.0 - clamped))


def trimmed_mean(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw):
    active_probs = []
    if config.weight_cv != 0.0:
        active_probs.append(torch.sigmoid(m_cv))
    if m_dd is not None and config.weight_dd != 0.0:
        active_probs.append(torch.sigmoid(m_dd))
    if m_ct is not None and config.weight_ct != 0.0:
        active_probs.append(torch.sigmoid(m_ct))
    if m_bm is not None and config.weight_bm != 0.0:
        active_probs.append(torch.sigmoid(m_bm))
    if m_bd is not None and config.weight_bd != 0.0:
        active_probs.append(torch.sigmoid(m_bd))
    if m_qa is not None and config.weight_qa != 0.0:
        active_probs.append(torch.sigmoid(m_qa))
    if m_ds is not None and config.weight_ds != 0.0:
        active_probs.append(torch.sigmoid(m_ds))
    if m_lr is not None and config.weight_lr != 0.0:
        active_probs.append(torch.sigmoid(m_lr))
    if m_de is not None and config.weight_de != 0.0:
        active_probs.append(torch.sigmoid(m_de))
    if m_sw is not None and config.weight_sw != 0.0:
        active_probs.append(torch.sigmoid(m_sw))

    if not active_probs:
        return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

    if len(active_probs) >= 3:
        stacked = torch.stack(active_probs, dim=-1)
        sum_p = torch.sum(stacked, dim=-1)
        min_p = torch.min(stacked, dim=-1).values
        max_p = torch.max(stacked, dim=-1).values
        trimmed_avg = (sum_p - min_p - max_p) / float(len(active_probs) - 2)
        clamped = trimmed_avg.clamp(1e-7, 1.0 - 1e-7)
        return torch.log(clamped / (1.0 - clamped))
    else:
        stacked = torch.stack(active_probs, dim=-1)
        avg_p = torch.mean(stacked, dim=-1)
        clamped = avg_p.clamp(1e-7, 1.0 - 1e-7)
        return torch.log(clamped / (1.0 - clamped))


def hard_gated(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw):
    active_probs = []
    if config.weight_cv != 0.0:
        active_probs.append(torch.sigmoid(m_cv))
    if m_dd is not None and config.weight_dd != 0.0:
        active_probs.append(torch.sigmoid(m_dd))
    if m_ct is not None and config.weight_ct != 0.0:
        active_probs.append(torch.sigmoid(m_ct))
    if m_bm is not None and config.weight_bm != 0.0:
        active_probs.append(torch.sigmoid(m_bm))
    if m_bd is not None and config.weight_bd != 0.0:
        active_probs.append(torch.sigmoid(m_bd))
    if m_qa is not None and config.weight_qa != 0.0:
        active_probs.append(torch.sigmoid(m_qa))
    if m_ds is not None and config.weight_ds != 0.0:
        active_probs.append(torch.sigmoid(m_ds))
    if m_lr is not None and config.weight_lr != 0.0:
        active_probs.append(torch.sigmoid(m_lr))
    if m_de is not None and config.weight_de != 0.0:
        active_probs.append(torch.sigmoid(m_de))
    if m_sw is not None and config.weight_sw != 0.0:
        active_probs.append(torch.sigmoid(m_sw))

    if not active_probs:
        return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

    stacked = torch.stack(active_probs, dim=-1)  # [N, B]
    tau = getattr(config, "gated_tau", 0.05)
    c = (stacked - 0.5).abs()
    mask = (c >= tau).float()
    has_active = (mask.sum(dim=-1, keepdim=True) > 0)
    weights = torch.where(has_active, mask, torch.ones_like(mask))
    avg_p = (weights * stacked).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    clamped = avg_p.clamp(1e-7, 1.0 - 1e-7)
    return torch.log(clamped / (1.0 - clamped))


def adaptive_trimmed(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw):
    active_probs = []
    if config.weight_cv != 0.0:
        active_probs.append(torch.sigmoid(m_cv))
    if m_dd is not None and config.weight_dd != 0.0:
        active_probs.append(torch.sigmoid(m_dd))
    if m_ct is not None and config.weight_ct != 0.0:
        active_probs.append(torch.sigmoid(m_ct))
    if m_bm is not None and config.weight_bm != 0.0:
        active_probs.append(torch.sigmoid(m_bm))
    if m_bd is not None and config.weight_bd != 0.0:
        active_probs.append(torch.sigmoid(m_bd))
    if m_qa is not None and config.weight_qa != 0.0:
        active_probs.append(torch.sigmoid(m_qa))
    if m_ds is not None and config.weight_ds != 0.0:
        active_probs.append(torch.sigmoid(m_ds))
    if m_lr is not None and config.weight_lr != 0.0:
        active_probs.append(torch.sigmoid(m_lr))
    if m_de is not None and config.weight_de != 0.0:
        active_probs.append(torch.sigmoid(m_de))
    if m_sw is not None and config.weight_sw != 0.0:
        active_probs.append(torch.sigmoid(m_sw))

    if not active_probs:
        return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

    stacked = torch.stack(active_probs, dim=-1)  # [N, B]
    B = stacked.shape[-1]
    if B < 3:
        avg_p = stacked.mean(dim=-1)
        clamped = avg_p.clamp(1e-7, 1.0 - 1e-7)
        return torch.log(clamped / (1.0 - clamped))

    sorted_p, _ = torch.sort(stacked, dim=-1)
    c = (stacked - 0.5).abs()
    c_med = torch.median(c, dim=-1).values
    min_p = sorted_p[:, 0]
    max_p = sorted_p[:, -1]
    c_min = (min_p - 0.5).abs()
    c_max = (max_p - 0.5).abs()

    tau = getattr(config, "adaptive_tau", 0.08)
    ratio = getattr(config, "adaptive_ratio", 1.5)

    drop_min = (c_min <= ratio * c_med) | (c_min <= tau)
    drop_max = (c_max <= ratio * c_med) | (c_max <= tau)

    sum_all = sorted_p.sum(dim=-1)
    count_all = torch.full_like(sum_all, float(B))
    sum_trimmed = sum_all - torch.where(drop_min, min_p, torch.zeros_like(min_p)) - torch.where(drop_max, max_p, torch.zeros_like(max_p))
    count_trimmed = count_all - drop_min.float() - drop_max.float()
    avg_p = sum_trimmed / count_trimmed.clamp_min(1.0)
    clamped = avg_p.clamp(1e-7, 1.0 - 1e-7)
    return torch.log(clamped / (1.0 - clamped))

