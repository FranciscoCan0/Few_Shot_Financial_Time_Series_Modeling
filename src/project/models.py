"""X-Trend (Wood et al., 2023) and the baseline neural forecaster, in PyTorch.

Architecture from Section 3, losses from Eqs. 8 and 20-23, fixed settings and search grid
from Appendix C (Tables 3-4). Training, the random search and ensembling live elsewhere;
this module only defines the networks and their losses.

    model = build_model("xtrend", n_inputs=8, n_tickers=49, loss="qre", context="final")
    t = vectorization_batching.as_torch(batch)       # one Batcher batch as tensors
    out = model(t["x"], t["ticker_id"], t["context_x"], t["context_mask"],
                t["context_ticker_id"], context_y=t["context_y"])
    loss = model_loss(out, t["y"], loss="qre", mask=t["score_mask"])

Inputs follow project.vectorization_batching.Batcher (front-padded contexts with a mask):
  target_x       (B, lt, n_inputs)         float, the feature_columns(cfg) inputs
  target_ticker  (B,)                      long, or None for zero-shot targets
  context_x      (B, C, lc, n_inputs)      float, the contexts' inputs
  context_y      (B, C, lc)                float, the contexts' vol-scaled next-day return
                                           (or leave it out and pass context_x with it as
                                           an extra last column, n_inputs + 1)
  context_mask   (B, C, lc)                bool, True on real steps
  context_ticker (B, C)                    long
  target         (B, lt)                   the `target` (y) of the target sequence
Target sequences are assumed to be full length (no padding).

The fixed settings, search grid and quantiles are read from configs/model.json.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from project.vectorization_batching import load_model_config

MODEL_CFG = load_model_config()
_s, _t, _l = MODEL_CFG["sequences"], MODEL_CFG["training"], MODEL_CFG["losses"]
# Table 4: fixed for every architecture
FIXED = {"lr": _t["learning_rate"], "warmup": _s["warmup_length"],
         "target_length": _s["target_length"], "patience": _t["early_stopping_patience"],
         "max_epochs": _t["max_epochs"], "n_heads": MODEL_CFG["architecture"]["attention_heads"],
         "alpha": {"gauss": _l["gaussian"]["alpha"], "qre": _l["quantile"]["alpha"]}}
# Table 3: random grid for the outer hyperparameter search
SEARCH_GRID = {"dropout": MODEL_CFG["search_space"]["dropout"],
               "hidden": MODEL_CFG["search_space"]["hidden_size"],
               "batch_size": MODEL_CFG["search_space"]["batch_size"],
               "max_grad_norm": MODEL_CFG["search_space"]["max_grad_norm"]}
# Section 3.4: quantiles for X-Trend-Q, including both tails
QUANTILES = tuple(_l["quantile"]["quantiles"])
# sharpe: X-Trend / Baseline (Eq. 8); gauss: X-Trend-G (Eq. 21); qre: X-Trend-Q (Eq. 23)
LOSSES = ("sharpe", "gauss", "qre")
# final: target attends to each context's final hidden state (F and C contexts);
# time: target step k attends to step k of every context (T contexts)
CONTEXTS = ("final", "time")


# --------------------------------------------------------------------------
# building blocks (Section 3.1)
# --------------------------------------------------------------------------

class FFN(nn.Module):
    """Eq. 12a: Linear3 ∘ ELU(Linear1(h) + Linear2(Embedding(s))).

    The embedding term only exists with static=True and is skipped when emb is None (a
    zero-shot target). emb is one vector per sequence, broadcast over the time steps.
    Dropout sits after the ELU; the paper does not say where it applies dropout.
    """

    def __init__(self, d_in, d_hidden, d_out, static=False, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_hidden, bias=False) if static else None
        self.lin3 = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, emb=None):
        z = self.lin1(h)
        if self.lin2 is not None and emb is not None:
            e = self.lin2(emb)
            while e.dim() < z.dim():          # (N, d) -> (N, 1, d) for sequences
                e = e.unsqueeze(-2)
            z = z + e
        return self.lin3(self.drop(F.elu(z)))


class VSN(nn.Module):
    """Eq. 13: each input j gets its own FFN_j: R -> R^dh, weighted by Softmax(FFN(x_t, s))."""

    def __init__(self, n_inputs, d_h, dropout=0.0):
        super().__init__()
        self.weights = FFN(n_inputs, d_h, n_inputs, static=True, dropout=dropout)
        self.inputs = nn.ModuleList(FFN(1, d_h, d_h, dropout=dropout) for _ in range(n_inputs))

    def forward(self, x, emb=None):
        w = torch.softmax(self.weights(x, emb), dim=-1)                          # (..., n)
        v = torch.stack([f(x[..., j:j + 1]) for j, f in enumerate(self.inputs)], dim=-2)
        return (w.unsqueeze(-1) * v).sum(-2)                                     # (..., dh)


class SequenceEncoder(nn.Module):
    """Ξ (Eq. 14): VSN -> LSTM -> LayerNorm(h + x') -> LayerNorm(FFN(a, s) + a).

    With d_extra > 0 this is the decoder Ξ_Dec (Eq. 19): the encoder output y_t is
    concatenated with VSN(x_t) and fused by an FFN and a LayerNorm before the LSTM.
    The LSTM's initial state is (FFN3(emb), FFN4(emb)), specific to each contract; with
    no embedding (zero-shot target) it starts from zeros.
    """

    def __init__(self, n_inputs, d_h, dropout=0.0, d_extra=0):
        super().__init__()
        self.vsn = VSN(n_inputs, d_h, dropout)
        self.fuse = FFN(d_h + d_extra, d_h, d_h, dropout=dropout) if d_extra else None
        self.fuse_norm = nn.LayerNorm(d_h) if d_extra else None
        self.lstm = nn.LSTM(d_h, d_h, batch_first=True)
        self.init_h = FFN(d_h, d_h, d_h)
        self.init_c = FFN(d_h, d_h, d_h)
        self.norm1 = nn.LayerNorm(d_h)
        self.ffn = FFN(d_h, d_h, d_h, static=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_h)

    def forward(self, x, emb=None, lengths=None, extra=None):
        """x (N, L, n_inputs), left-aligned when `lengths` is given: steps past each
        sequence's length are padding and the LSTM never reads them. Returns (N, L, dh)."""
        xp = self.vsn(x, emb)
        if self.fuse is not None:
            xp = self.fuse_norm(self.fuse(torch.cat([xp, extra], dim=-1)))

        if emb is not None:
            state = (self.init_h(emb).unsqueeze(0).contiguous(),
                     self.init_c(emb).unsqueeze(0).contiguous())
        else:
            zeros = xp.new_zeros(1, x.shape[0], xp.shape[-1])
            state = (zeros, zeros)

        if lengths is None or bool((lengths == x.shape[1]).all()):
            h, _ = self.lstm(xp, state)
        else:
            packed = pack_padded_sequence(xp, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            h, _ = self.lstm(packed, state)
            h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])

        a = self.norm1(h + xp)
        return self.norm2(self.ffn(a, emb) + a)


def _left_align(x, mask):
    """ContextSampler pads at the front; move each sequence's real steps to the start
    so packing can skip the padding. x (N, L, d), mask (N, L). Returns (x, lengths)."""
    lengths = mask.sum(-1)
    L = x.shape[1]
    idx = (torch.arange(L, device=x.device) + (L - lengths).unsqueeze(-1)).clamp(max=L - 1)
    return x.gather(1, idx.unsqueeze(-1).expand_as(x)), lengths


# --------------------------------------------------------------------------
# cross attention over the context set (Sections 2.4, 3.3)
# --------------------------------------------------------------------------

class ContextAttention(nn.Module):
    """Eqs. 17-18: self-attention over the context values (V' = FFN ∘ Att(V, V, V)), then
    the target query cross-attends to the context keys: y = LayerNorm ∘ FFN ∘ Att(q, K, V').

    Uses torch's multi-head attention with d_att = dh split across the heads. The returned
    weights are averaged over heads, which is what Fig. 9 interprets.
    """

    def __init__(self, d_h, n_heads, dropout=0.0):
        super().__init__()
        self.self_att = nn.MultiheadAttention(d_h, n_heads, dropout=dropout, batch_first=True)
        self.self_ffn = FFN(d_h, d_h, d_h, dropout=dropout)
        self.cross_att = nn.MultiheadAttention(d_h, n_heads, dropout=dropout, batch_first=True)
        self.cross_ffn = FFN(d_h, d_h, d_h, dropout=dropout)
        self.norm = nn.LayerNorm(d_h)

    def forward(self, q, k, v):
        """q (M, Tq, dh); k, v (M, C, dh). Returns y (M, Tq, dh), weights (M, Tq, C)."""
        v = self.self_ffn(self.self_att(v, v, v, need_weights=False)[0])
        y, w = self.cross_att(q, k, v)
        return self.norm(self.cross_ffn(y)), w


# --------------------------------------------------------------------------
# output heads (Sections 2.3, 3.4)
# --------------------------------------------------------------------------

class PositionHead(nn.Module):
    """Turns the decoder output into a position z_t in (-1, 1).

    sharpe: tanh ∘ Linear (Eq. 7), trained on the Sharpe loss alone.
    gauss:  forecast (mu, sigma) of the vol-scaled next-day return, then PTP_G: FFN + tanh.
    qre:    13 forecast quantiles, then PTP_Q: FFN + tanh.
    The Sharpe loss backpropagates through PTP into the forecast head (joint training).
    """

    def __init__(self, d_h, loss):
        super().__init__()
        self.loss = loss
        if loss == "sharpe":
            self.out = nn.Linear(d_h, 1)
        else:
            n_out = 2 if loss == "gauss" else len(QUANTILES)
            self.forecast = nn.Linear(d_h, n_out)
            self.ptp = FFN(n_out, d_h, 1)

    def forward(self, h):
        if self.loss == "sharpe":
            return {"position": torch.tanh(self.out(h)).squeeze(-1)}
        f = self.forecast(h)
        if self.loss == "gauss":
            mu, sigma = f[..., 0], F.softplus(f[..., 1]) + 1e-6
            out, f = {"mu": mu, "sigma": sigma}, torch.stack([mu, sigma], dim=-1)
        else:
            out = {"quantiles": f}
        out["position"] = torch.tanh(self.ptp(f)).squeeze(-1)
        return out


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

class XTrend(nn.Module):
    """X-Trend (Fig. 4), or the Baseline neural forecaster when context is None.

    Baseline: the decoder Ξ on the target sequence alone, then the head.
    X-Trend:  Ξ_key encodes each context without its next-day returns, Ξ_value with them,
              Ξ_query encodes the target; the target cross-attends to the contexts and the
              decoder Ξ_Dec fuses that output y_t with the target sequence.
    One ticker embedding is shared by every part. Zero-shot: pass target_ticker=None; the
    contexts keep their embeddings, as in Section 3.4.

    forward returns a dict with `position` (B, lt), the forecast outputs for joint losses
    (`mu`, `sigma` or `quantiles`) and, with contexts, `attention` (B, lt, C).
    """

    def __init__(self, n_inputs, n_tickers, hidden=64, dropout=0.3, loss="sharpe",
                 context="final", n_heads=FIXED["n_heads"]):
        super().__init__()
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
        if context is not None and context not in CONTEXTS:
            raise ValueError(f"context must be one of {CONTEXTS} or None, got {context!r}")
        if hidden % n_heads:
            raise ValueError(f"hidden ({hidden}) must be divisible by n_heads ({n_heads})")
        self.loss, self.context = loss, context
        self.embedding = nn.Embedding(n_tickers, hidden)
        if context is not None:
            self.key_enc = SequenceEncoder(n_inputs, hidden, dropout)
            self.value_enc = SequenceEncoder(n_inputs + 1, hidden, dropout)
            self.query_enc = SequenceEncoder(n_inputs, hidden, dropout)
            self.attention = ContextAttention(hidden, n_heads, dropout)
        self.decoder = SequenceEncoder(n_inputs, hidden, dropout,
                                       d_extra=hidden if context is not None else 0)
        self.head = PositionHead(hidden, loss)

    def forward(self, target_x, target_ticker=None, context_x=None, context_mask=None,
                context_ticker=None, context_y=None):
        emb_t = self.embedding(target_ticker) if target_ticker is not None else None
        if self.context is None:
            return self.head(self.decoder(target_x, emb_t))
        if context_x is None or context_mask is None or context_ticker is None:
            raise ValueError("X-Trend needs context_x, context_mask and context_ticker")
        if context_y is not None:          # Batcher format: append the next-day return
            context_x = torch.cat([context_x, context_y.unsqueeze(-1)], dim=-1)
        if context_x.shape[-1] != target_x.shape[-1] + 1:
            raise ValueError(f"context_x has {context_x.shape[-1]} columns, the target "
                             f"{target_x.shape[-1]}: pass context_y, or context_x with the "
                             "next-day return as an extra last column")

        B, C, lc, d = context_x.shape
        lt = target_x.shape[1]
        cx, lengths = _left_align(context_x.reshape(B * C, lc, d),
                                  context_mask.reshape(B * C, lc))
        emb_c = self.embedding(context_ticker.reshape(B * C))
        keys = self.key_enc(cx[..., :-1], emb_c, lengths)     # (B*C, lc, dh), Eq. 15
        values = self.value_enc(cx, emb_c, lengths)
        query = self.query_enc(target_x, emb_t)               # (B, lt, dh), Eq. 16

        if self.context == "final":   # every target step attends to the final states
            rows, last = torch.arange(B * C, device=cx.device), lengths - 1
            k, v = keys[rows, last].view(B, C, -1), values[rows, last].view(B, C, -1)
            y, w = self.attention(query, k, v)                # (B, lt, dh), (B, lt, C)
        else:                         # target step k attends to step k of each context
            if lc != lt or not bool(context_mask.all()):
                raise ValueError(f"time-equivalent contexts must be unpadded and as long as "
                                 f"the target ({lt} steps), got length {lc}")
            k = keys.view(B, C, lt, -1).transpose(1, 2).reshape(B * lt, C, -1)
            v = values.view(B, C, lt, -1).transpose(1, 2).reshape(B * lt, C, -1)
            y, w = self.attention(query.reshape(B * lt, 1, -1), k, v)
            y, w = y.view(B, lt, -1), w.view(B, lt, C)

        out = self.head(self.decoder(target_x, emb_t, extra=y))
        out["attention"] = w
        return out


def build_model(kind, n_inputs, n_tickers, loss="sharpe", context="final", hidden=64,
                dropout=0.3, n_heads=FIXED["n_heads"]):
    """kind: 'baseline' (no context set) or 'xtrend'. With kind='xtrend', loss picks the
    variant (sharpe: X-Trend, gauss: X-Trend-G, qre: X-Trend-Q) and context the attention
    ('final' for F and CPD contexts, 'time' for T). hidden and dropout come from Table 3.
    n_tickers = panel.ticker_id.max() + 1 so every ticker_id has an embedding row."""
    if kind not in ("baseline", "xtrend"):
        raise ValueError(f"kind must be 'baseline' or 'xtrend', got {kind!r}")
    return XTrend(n_inputs, n_tickers, hidden, dropout, loss,
                  context if kind == "xtrend" else None, n_heads)


# --------------------------------------------------------------------------
# losses (Eqs. 8, 20-23)
# --------------------------------------------------------------------------

def _omega(x, warmup, mask):
    """The pairs Ω the loss is taken over: every (sequence, step) after the first `warmup`
    steps (Section 2.3), and only real steps if a mask is given. Keeps trailing dims."""
    x = x[:, warmup:]
    if mask is not None:
        return x[mask[:, warmup:]]
    return x.reshape(-1, *x.shape[2:])


def sharpe_loss(position, target, warmup=FIXED["warmup"], mask=None):
    """Eq. 8: minus the annualised Sharpe ratio of z_t * (vol-scaled r_{t+1}), pooled over
    every asset and day in the batch."""
    r = _omega(position * target, warmup, mask)
    return -np.sqrt(252) * r.mean() / (r.std() + 1e-9)


def gaussian_nll(mu, sigma, target, warmup=FIXED["warmup"], mask=None):
    """Eq. 20: Gaussian negative log-likelihood of the vol-scaled next-day return."""
    mu, sigma, y = (_omega(t, warmup, mask) for t in (mu, sigma, target))
    return (torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + 0.5 * np.log(2 * np.pi)).mean()


def quantile_loss(quantiles, target, warmup=FIXED["warmup"], mask=None):
    """Eq. 22: pinball loss averaged over Ω and the quantiles in QUANTILES."""
    q, y = _omega(quantiles, warmup, mask), _omega(target, warmup, mask).unsqueeze(-1)
    eta = torch.tensor(QUANTILES, dtype=q.dtype, device=q.device)
    err = y - q
    return torch.maximum(eta * err, (eta - 1) * err).mean()


def model_loss(out, target, loss="sharpe", alpha=None, warmup=FIXED["warmup"], mask=None):
    """The training loss for a variant: Sharpe alone, or alpha * forecast loss + Sharpe
    (Eqs. 21, 23) with alpha = 1 for gauss and 5 for qre unless given."""
    sharpe = sharpe_loss(out["position"], target, warmup, mask)
    if loss == "sharpe":
        return sharpe
    if alpha is None:
        alpha = FIXED["alpha"][loss]
    if loss == "gauss":
        return alpha * gaussian_nll(out["mu"], out["sigma"], target, warmup, mask) + sharpe
    if loss == "qre":
        return alpha * quantile_loss(out["quantiles"], target, warmup, mask) + sharpe
    raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
