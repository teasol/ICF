"""Multi-head Latent Attention (MLA) layer — position-free, no noise.

DeepSeek-style low-rank latent attention. Queries/keys/values are produced
through compact latent projections so the KV cache stores only ``c_kv``
(d_c << n_heads * d_h). In inference the up-projections are absorbed into the
query/output weights, so the d_h expansion is never materialized.

Two forward modes:

Training (``use_cache=False``, standard attention, no caching):
    c_kv = W_DKV(h)                   [B, S, d_c]
    K, V = W_UK(c_kv), W_UV(c_kv)     [B, N, S, d_h]
    c_q  = W_DQ(h)                    [B, S, d_cq]
    Q    = W_UQ(c_q)                  [B, N, S, d_h]
    O    = softmax(Q K^T / sqrt(d_h)) V
    out  = W_O(flatten heads)         [B, S, d_model]

Inference (``use_cache=True``, KV cache stores ONLY ``c_kv`` + matrix
absorption, no d_h expansion):
    cache holds c_kv only             [B, S_total, d_c]
    W_Q_abs = W_UQ^T W_UK             [N, d_cq, d_c]
    W_O_abs = W_UV^T W_O              [N, d_c, d_model]
    scores  = (c_q @ W_Q_abs) @ c_kv^T / sqrt(d_h)
    out     = softmax(scores) @ c_kv @ W_O_abs

No positional embedding and no injected noise (per spec). ``is_causal`` applies
a standard left-to-right mask in both modes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiheadLatentAttention(nn.Module):
    """Multi-head latent attention with KV-cache / matrix-absorption inference.

    Args:
        d_model: input/output dimension.
        n_heads: number of attention heads.
        d_h: per-head dimension (n_heads * d_h is the expanded attention dim).
        d_c: KV compression latent dim (d_c << n_heads * d_h).
        d_cq: query compression latent dim.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_h: int,
        d_c: int,
        d_cq: int,
    ) -> None:
        super().__init__()
        assert d_c <= n_heads * d_h, "d_c should compress the KV (d_c << n_heads*d_h)"
        assert d_cq >= 1 and d_c >= 1
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_h
        self.d_c = d_c
        self.d_cq = d_cq
        self.scale = 1.0 / math.sqrt(d_h)

        # 1. Query down/up projection.
        self.w_dq = nn.Linear(d_model, d_cq, bias=False)          # [d_cq, d_model]
        self.w_uq = nn.Linear(d_cq, n_heads * d_h, bias=False)    # [n_heads*d_h, d_cq]
        # 2. KV down/up projection.
        self.w_dkv = nn.Linear(d_model, d_c, bias=False)          # [d_c, d_model]
        self.w_uk = nn.Linear(d_c, n_heads * d_h, bias=False)     # [n_heads*d_h, d_c]
        self.w_uv = nn.Linear(d_c, n_heads * d_h, bias=False)     # [n_heads*d_h, d_c]
        # 3. Output projection.
        self.w_o = nn.Linear(n_heads * d_h, d_model, bias=False)  # [d_model, n_heads*d_h]

    def _causal_add(self, scores: torch.Tensor, s_total: int, s: int) -> torch.Tensor:
        """Mask keys beyond the current position (works for both modes).

        Query row ``i`` (of the current S tokens) may attend to cache keys
        ``j <= (S_total - S) + i``; everything above is set to -inf.
        """
        diagonal = s_total - s + 1
        mask = torch.triu(torch.full_like(scores, float("-inf")), diagonal=diagonal)
        return scores + mask

    def forward(
        self,
        h: torch.Tensor,
        is_causal: bool = False,
        use_cache: bool = False,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            h: input tokens [B, S, d_model].
            is_causal: apply a left-to-right causal mask.
            use_cache: inference mode with KV-cache + matrix absorption.
            cache: previous ``c_kv`` [B, S_prev, d_c] (or None).

        Returns:
            (output [B, S, d_model], cache). In training mode the returned
            cache is the current ``c_kv`` (convenience); in inference mode it is
            the concatenated cache [B, S_prev + S, d_c].
        """
        b, s, _ = h.shape
        c_kv = self.w_dkv(h)   # [B, S, d_c]
        c_q = self.w_dq(h)     # [B, S, d_cq]

        if not use_cache:
            # Training: expand to per-head Q/K/V and run standard attention.
            q = self.w_uq(c_q).view(b, s, self.n_heads, self.d_h).transpose(1, 2)
            k = self.w_uk(c_kv).view(b, s, self.n_heads, self.d_h).transpose(1, 2)
            v = self.w_uv(c_kv).view(b, s, self.n_heads, self.d_h).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,N,S,S]
            if is_causal:
                scores = self._causal_add(scores, s, s)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)  # [B,N,S,d_h]
            out = out.transpose(1, 2).reshape(b, s, self.n_heads * self.d_h)
            out = self.w_o(out)          # [B,S,d_model]
            return out, c_kv

        # Inference: append only c_kv to the cache.
        if cache is None:
            cache = c_kv
        else:
            cache = torch.cat([cache, c_kv], dim=1)  # [B, S_total, d_c]
        s_total = cache.shape[1]

        # Matrix absorption (weights never expanded to d_h).
        w_uq = self.w_uq.weight.view(self.n_heads, self.d_h, self.d_cq)   # [N, d_h, d_cq]
        w_uk = self.w_uk.weight.view(self.n_heads, self.d_h, self.d_c)    # [N, d_h, d_c]
        w_uv = self.w_uv.weight.view(self.n_heads, self.d_h, self.d_c)    # [N, d_h, d_c]
        w_o = self.w_o.weight.view(self.d_model, self.n_heads, self.d_h)  # [d_model, N, d_h]

        # W_Q_abs[n] = W_UQ[n]^T @ W_UK[n] : [d_cq, d_c]
        w_q_abs = torch.einsum("nhq,nhc->nqc", w_uq, w_uk)
        # W_O_abs[n] = W_UV[n]^T @ W_O[:, n, :] : [d_c, d_model]
        w_o_abs = torch.einsum("nhc,dnh->ncd", w_uv, w_o)

        # Query in the KV latent space: [B, N, S, d_c]
        q_latent = torch.einsum("bsq,nqc->bnsc", c_q, w_q_abs)
        scores = torch.einsum("bnsc,btc->bnst", q_latent, cache) * self.scale
        if is_causal:
            scores = self._causal_add(scores, s_total, s)
        attn = torch.softmax(scores, dim=-1)                      # [B,N,S,S_total]
        out = torch.einsum("bnst,btc,ncd->bsd", attn, cache, w_o_abs)  # [B,S,d_model]
        return out, cache


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)  # tight equivalence check

    d_model, n_heads, d_h, d_c, d_cq = 4096, 128, 128, 512, 1536
    b, s = 2, 9
    layer = MultiheadLatentAttention(d_model, n_heads, d_h, d_c, d_cq)
    h = torch.randn(b, s, d_model)

    # 1) Non-causal: full-epoch training == one-shot inference (cache=None).
    train_out, train_cache = layer(h, is_causal=False, use_cache=False)
    assert train_cache.shape == (b, s, d_c), f"cache shape {train_cache.shape}"
    inf_out, inf_cache = layer(h, is_causal=False, use_cache=True, cache=None)
    assert inf_cache.shape == (b, s, d_c)
    diff = (train_out - inf_out).abs().max().item()
    print(f"[non-causal] train vs absorbed-inference max|diff| = {diff:.3e}")
    assert diff < 1e-8, "non-causal absorption mismatch"

    # 2) Causal: token-by-token inference must match full-sequence training.
    train_out, _ = layer(h, is_causal=True, use_cache=False)
    cache = None
    outs = []
    for t in range(s):
        out, cache = layer(h[:, t : t + 1], is_causal=True, use_cache=True, cache=cache)
        outs.append(out)
    causal_inf = torch.cat(outs, dim=1)
    assert cache.shape == (b, s, d_c), f"causal cache shape {cache.shape}"
    diff = (train_out - causal_inf).abs().max().item()
    print(f"[causal]    train vs token-by-token inference max|diff| = {diff:.3e}")
    assert diff < 1e-8, "causal cache mismatch"

    # 3) Parameter count / cache size check.
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"[params]    total = {n_params:,}")
    print(f"[cache]     full KV = {b * s * n_heads * d_h * 2} floats vs MLA c_kv = {b * s * d_c} floats")
    print("MLA implementation OK (training == inference, causal cache correct).")
