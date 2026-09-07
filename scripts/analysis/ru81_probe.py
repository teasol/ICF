"""RU-81: replay existing solvers on identical features, changing only lambda.

No production model is changed. Hooks exist only in the diagnostic worker.
The three penalties depend on context Gram geometry, never query performance.
"""
from __future__ import annotations

import torch

FACTORS = (0.1, 1.0, 10.0)
BRANCHES = ("cv", "bm", "bd", "qa", "ds")


def geometry(context, labels):
    with torch.autocast(device_type=context.device.type, enabled=False):
        x = context.float()
        counts = torch.bincount(labels.long(), minlength=2)
        if (counts == 0).any():
            raise ValueError("Both context classes required")
        w = counts.float().reciprocal()[labels.long()]
        mu = (w[:, None] * x).sum(0) / w.sum()
        design = (x - mu) * w.sqrt()[:, None]
        eigen = torch.linalg.eigvalsh((design @ design.T).double()).clamp_min(0)
        # Gram was formed in float32; double eigendecomposition does not undo
        # its rounding noise. Rank uses the precision of that original Gram.
        tol = eigen.max() * design.shape[0] * torch.finfo(torch.float32).eps
        positive = eigen[eigen > tol]
        if positive.numel() == 0:
            raise ValueError("Zero-rank context Gram")
        scale = positive.mean().item()
        penalties = [min(1e4, max(1e-4, scale * f)) for f in FACTORS]
        return {
            "eigenvalues": eigen.cpu(), "rank": positive.numel(),
            "dimension": x.shape[1], "n_context": x.shape[0],
            "scale": scale, "rank_tolerance": tol.item(), "penalties": penalties,
            "df_baseline": (eigen / (eigen + 1.0)).sum().item(),
            "df_probe": [(eigen / (eigen + p)).sum().item() for p in penalties],
        }


def ensemble(margins):
    p = torch.stack([torch.sigmoid(margins[b].float()) for b in BRANCHES])
    return p.sort(dim=0).values[1:-1].mean(dim=0)


def check_replay(actual, replay, tolerance=1e-5):
    if actual.shape != replay.shape or not torch.isfinite(replay).all():
        raise ValueError("Invalid replay shape or nonfinite output")
    error = (actual.float() - replay.float()).abs().max().item()
    if error > tolerance:
        raise ValueError(f"Baseline replay failed: {error} > {tolerance}")
    return error


def install(evaluator):
    from src.models.set_transformer_ridge import SetTransformerRidgeModel

    original_cv = SetTransformerRidgeModel._ridge_logits
    original_krr = evaluator._solve_kernel_ridge
    original_trial = evaluator.evaluate_trial
    state = {}
    rows = []

    def cv(self, context, labels, query):
        base = original_cv(self, context, labels, query)
        if "cv" in state:
            raise ValueError("Unexpected multiple CV calls")
        with torch.autocast(device_type=context.device.type, enabled=False):
            normalized, _ = self._normalize_descriptors(context.float(), query.float())
        g = geometry(normalized, labels)
        saved = self.ridge_log_lambda.detach().clone()
        try:
            replay = original_cv(self, context, labels, query)
            g["replay_error"] = check_replay(base, replay)
            g["margins"] = []
            for penalty in g["penalties"]:
                self.ridge_log_lambda.fill_(penalty)
                self.ridge_log_lambda.log_()
                logits = original_cv(self, context, labels, query)
                g["margins"].append((logits[:, 1] - logits[:, 0]).float().cpu())
        finally:
            self.ridge_log_lambda.copy_(saved)
        g["baseline"] = (base[:, 1] - base[:, 0]).float().cpu()
        state["cv"] = g
        return base

    def krr(context, labels, query, **kwargs):
        names = [n for n in ("bm", "qa", "ds") if n not in state]
        if not names or kwargs.get("kernel", "linear") != "linear" or kwargs.get("return_loo"):
            raise ValueError("Unexpected diagnostic solver path")
        name = names[0]
        base = original_krr(context, labels, query, **kwargs)
        with torch.autocast(device_type=context.device.type, enabled=False):
            centre = context.mean(dim=0, keepdim=True)
            scale = (context - centre).square().mean(0).sqrt().clamp_min(1e-6)
            normalized = (context - centre) / scale
        g = geometry(normalized, labels)
        replay = original_krr(context, labels, query, **kwargs)
        g["replay_error"] = check_replay(base, replay)
        g["baseline"] = base.float().cpu()
        g["margins"] = [original_krr(context, labels, query,
            **{**kwargs, "reg_lambda": p}).float().cpu() for p in g["penalties"]]
        state[name] = g
        return base

    def trial(**kwargs):
        state.clear()
        result = original_trial(**kwargs)
        if set(state) != {"cv", "bm", "qa", "ds"}:
            raise ValueError(f"Missing branches: {state.keys()}")
        margins = {b: result[f"m_{b}"].float().cpu() for b in BRANCHES}
        for b, g in state.items():
            g["path_error"] = check_replay(margins[b], g["baseline"])
            if not all(torch.isfinite(m).all() for m in g["margins"]):
                raise ValueError("Nonfinite probe")
        replay_error = check_replay(result["probability"].cpu(), ensemble(margins), 1e-6)
        rows.append({"slide_id": result["queried_ids"], "label": result["target"].cpu(),
                     "baseline": margins, "probe": dict(state), "ensemble_replay_error": replay_error})
        return result

    SetTransformerRidgeModel._ridge_logits = cv
    evaluator._solve_kernel_ridge = krr
    evaluator.evaluate_trial = trial
    return rows
