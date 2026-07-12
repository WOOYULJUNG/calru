"""Full recurrent block scaffold for PAN-Block experiments.

This module separates the shared sequence-model scaffold from the recurrent
module. The intended comparison is:

    LRU-Block, near-one LRU-Block, real-diag LRU-Block, P-LRU-Block, PAN-Block,
    PAN-NW-Block, PAN-RNW-Block, RG-LRU-Block, GRU-Block

All variants share encoder, block wrapper, GLU path, skip path, normalization,
dropout, depth, width, and output head. The recurrent module is the intervention.
"""

import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from plru_regularizers import DEFAULT_C, DEFAULT_TAU, canonical_q_loss_from_lam


BLOCK_VARIANTS = (
    "LRU-Block",
    "near-one LRU-Block",
    "real-diag LRU-Block",
    "P-LRU-Block",
    "PAN-Block",
    "PAN-NW-Block",
    "PAN-RNW-Block",
    "RG-LRU-Block",
    "GRU-Block",
)


def normalize_block_variant(name):
    key = re.sub(r"[\s_]+", "-", name.strip().lower())
    aliases = {
        "lru": "LRU-Block",
        "lru-block": "LRU-Block",
        "near-one-lru": "near-one LRU-Block",
        "near-one-lru-block": "near-one LRU-Block",
        "nearone-lru": "near-one LRU-Block",
        "nearone-lru-block": "near-one LRU-Block",
        "real-diag-lru": "real-diag LRU-Block",
        "real-diag-lru-block": "real-diag LRU-Block",
        "realdiag-lru": "real-diag LRU-Block",
        "realdiag-lru-block": "real-diag LRU-Block",
        "real-diagonal-lru": "real-diag LRU-Block",
        "real-diagonal-lru-block": "real-diag LRU-Block",
        "p-lru": "P-LRU-Block",
        "plru": "P-LRU-Block",
        "p-lru-block": "P-LRU-Block",
        "plru-block": "P-LRU-Block",
        "pan": "PAN-Block",
        "pan-block": "PAN-Block",
        "pan-nw": "PAN-NW-Block",
        "pan-nw-block": "PAN-NW-Block",
        "pan-nonlinear-writer": "PAN-NW-Block",
        "pan-nonlinear-writer-block": "PAN-NW-Block",
        "pan-rnw": "PAN-RNW-Block",
        "pan-rnw-block": "PAN-RNW-Block",
        "pan-recurrent-writer": "PAN-RNW-Block",
        "pan-recurrent-writer-block": "PAN-RNW-Block",
        "rg-lru": "RG-LRU-Block",
        "rglru": "RG-LRU-Block",
        "rg-lru-block": "RG-LRU-Block",
        "rglru-block": "RG-LRU-Block",
        "griffin-rg-lru": "RG-LRU-Block",
        "griffin-rg-lru-block": "RG-LRU-Block",
        "gru-block": "GRU-Block",
        "scaffold-gru": "GRU-Block",
        "scaffold-gru-block": "GRU-Block",
    }
    if key not in aliases:
        valid = ", ".join(BLOCK_VARIANTS)
        raise ValueError(f"unknown block variant {name!r}; expected one of: {valid}")
    return aliases[key]


def _lambda_to_theta(lam):
    q = lam.square().clamp(1e-8, 1.0 - 1e-8)
    return torch.logit(q)


class ComplexLRURec(nn.Module):
    """Complex diagonal LRU recurrence stored as real and imaginary state."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        r_min=0.90,
        r_max=0.999,
        max_phase=2.0 * math.pi,
        gamma_mode="tied",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.gamma_mode = gamma_mode

        u = torch.linspace(float(r_max), float(r_min), self.hidden_dim)
        u = u.clamp(1e-4, 0.9999)
        self.nu = nn.Parameter(torch.log(-torch.log(u)))
        self.phase = nn.Parameter(torch.rand(self.hidden_dim) * float(max_phase))

        scale = 1.0 / math.sqrt(2.0 * self.input_dim)
        self.B_re = nn.Parameter(torch.randn(self.hidden_dim, self.input_dim) * scale)
        self.B_im = nn.Parameter(torch.randn(self.hidden_dim, self.input_dim) * scale)
        if gamma_mode == "free":
            self.gamma_raw = nn.Parameter(torch.zeros(self.hidden_dim))
        self.out_proj = nn.Linear(2 * self.hidden_dim, self.output_dim)

    @property
    def state_size(self):
        return 2 * self.hidden_dim

    def lam_mag(self):
        return torch.exp(-torch.exp(self.nu)).clamp(0.0, 0.9999)

    def gamma(self):
        mag = self.lam_mag()
        if self.gamma_mode == "tied":
            return torch.sqrt(torch.clamp(1.0 - mag.square(), min=1e-6))
        if self.gamma_mode == "free":
            return torch.sigmoid(self.gamma_raw)
        return torch.ones_like(mag)

    def step(self, u_t, state):
        hr, hi = state.chunk(2, dim=-1)
        mag = self.lam_mag()
        a = mag * torch.cos(self.phase)
        b = mag * torch.sin(self.phase)
        gamma = self.gamma()
        wr = u_t @ self.B_re.t()
        wi = u_t @ self.B_im.t()
        hr_next = a * hr - b * hi + gamma * wr
        hi_next = b * hr + a * hi + gamma * wi
        return torch.cat([hr_next, hi_next], dim=-1)

    def output(self, state):
        return self.out_proj(state)


class RealDiagLRURec(nn.Module):
    """Real diagonal recurrence used for phase-zero P-LRU and PAN substrates."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        r_min=0.0,
        r_max=1.0,
        gamma_free=True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        u = torch.rand(self.hidden_dim) * (float(r_max) - float(r_min)) + float(r_min)
        u = u.clamp(1e-4, 0.9999)
        self.nu = nn.Parameter(torch.log(-torch.log(u)))

        scale = 1.0 / math.sqrt(self.input_dim)
        self.B = nn.Parameter(torch.randn(self.hidden_dim, self.input_dim) * scale)
        self.gamma_raw = nn.Parameter(torch.zeros(self.hidden_dim)) if gamma_free else None
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

    @property
    def state_size(self):
        return self.hidden_dim

    def lam_mag(self):
        return torch.exp(-torch.exp(self.nu)).clamp(0.0, 0.9999)

    def gamma(self):
        if self.gamma_raw is None:
            return torch.ones_like(self.lam_mag())
        return torch.sigmoid(self.gamma_raw)

    def step(self, u_t, state):
        return self.lam_mag() * state + self.gamma() * (u_t @ self.B.t())

    def output(self, state):
        return self.out_proj(state)


class PLRURec(RealDiagLRURec):
    """Phase-zero real diagonal recurrence with canonical P-LRU regularization."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        tau=DEFAULT_TAU,
        c=DEFAULT_C,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            r_min=0.0,
            r_max=1.0,
            gamma_free=True,
        )
        self.tau = float(tau)
        self.c = float(c)

    def regularization_loss(self):
        return canonical_q_loss_from_lam(self.lam_mag(), tau=self.tau, c=self.c)


class PANRec(nn.Module):
    """PAN recurrence with direct score update for theta.

    The update itself is performed outside autograd:

        theta_j <- theta_j + eta_lambda * s_j
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        lambda_min=0.90,
        lambda_max=0.999,
        gamma_free=True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        lam = torch.linspace(float(lambda_max), float(lambda_min), self.hidden_dim)
        lam = lam.clamp(1e-4, 0.9999)
        self.theta = nn.Parameter(_lambda_to_theta(lam), requires_grad=False)

        scale = 1.0 / math.sqrt(self.input_dim)
        self.B = nn.Parameter(torch.randn(self.hidden_dim, self.input_dim) * scale)
        self.gamma_raw = nn.Parameter(torch.zeros(self.hidden_dim)) if gamma_free else None
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

    @property
    def state_size(self):
        return self.hidden_dim

    def lam_mag(self):
        q = torch.sigmoid(self.theta).clamp(1e-8, 1.0 - 1e-8)
        return torch.sqrt(q)

    def gamma(self):
        if self.gamma_raw is None:
            return torch.ones_like(self.lam_mag())
        return torch.sigmoid(self.gamma_raw)

    def step(self, u_t, state):
        return self.lam_mag() * state + self.gamma() * (u_t @ self.B.t())

    def output(self, state):
        return self.out_proj(state)

    @torch.no_grad()
    def update_theta(self, scores, eta_lambda):
        delta = torch.as_tensor(scores, dtype=self.theta.dtype, device=self.theta.device)
        if delta.shape != self.theta.shape:
            raise ValueError(f"PAN score shape {tuple(delta.shape)} does not match theta {tuple(self.theta.shape)}")
        self.theta.add_(float(eta_lambda) * delta)
        self.theta.clamp_(-18.0, 18.0)


class PANNonlinearWriterRec(nn.Module):
    """PAN recurrence with a zero-drive nonlinear writer.

    writer_mode="input" uses g(u_t) with bias-free layers, so g(0)=0.
    writer_mode="recurrent" uses g(h_{t-1}, u_t) - g(h_{t-1}, 0), so
    the write is exactly zero when the input drive is zero.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        lambda_min=0.90,
        lambda_max=0.999,
        gamma_free=True,
        writer_mode="input",
        writer_hidden=None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.writer_mode = str(writer_mode)
        width = int(writer_hidden or max(self.input_dim, self.hidden_dim))

        lam = torch.linspace(float(lambda_max), float(lambda_min), self.hidden_dim)
        lam = lam.clamp(1e-4, 0.9999)
        self.theta = nn.Parameter(_lambda_to_theta(lam), requires_grad=False)
        self.gamma_raw = nn.Parameter(torch.zeros(self.hidden_dim)) if gamma_free else None

        if self.writer_mode == "input":
            self.writer = nn.Sequential(
                nn.Linear(self.input_dim, width, bias=False),
                nn.GELU(),
                nn.Linear(width, self.hidden_dim, bias=False),
            )
        elif self.writer_mode == "recurrent":
            self.writer = nn.Sequential(
                nn.Linear(self.hidden_dim + self.input_dim, width),
                nn.GELU(),
                nn.Linear(width, self.hidden_dim),
            )
        else:
            raise ValueError(f"unknown writer_mode: {writer_mode}")
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

    @property
    def state_size(self):
        return self.hidden_dim

    def lam_mag(self):
        q = torch.sigmoid(self.theta).clamp(1e-8, 1.0 - 1e-8)
        return torch.sqrt(q)

    def gamma(self):
        if self.gamma_raw is None:
            return torch.ones_like(self.lam_mag())
        return torch.sigmoid(self.gamma_raw)

    def write(self, u_t, state):
        if self.writer_mode == "input":
            return self.writer(u_t)
        z = torch.cat([state, u_t], dim=-1)
        zero_u = torch.zeros_like(u_t)
        z0 = torch.cat([state, zero_u], dim=-1)
        return self.writer(z) - self.writer(z0)

    def step(self, u_t, state):
        return self.lam_mag() * state + self.gamma() * self.write(u_t, state)

    def output(self, state):
        return self.out_proj(state)

    @torch.no_grad()
    def update_theta(self, scores, eta_lambda):
        delta = torch.as_tensor(scores, dtype=self.theta.dtype, device=self.theta.device)
        if delta.shape != self.theta.shape:
            raise ValueError(f"PAN score shape {tuple(delta.shape)} does not match theta {tuple(self.theta.shape)}")
        self.theta.add_(float(eta_lambda) * delta)
        self.theta.clamp_(-18.0, 18.0)


class BlockDiagonalLinear(nn.Module):
    """Square block-diagonal affine map used by the Griffin RG-LRU gates.

    This is intentionally local to the recurrence implementation so the
    surrounding :class:`RecurrentBlock` remains identical across models.
    ``width`` must be divisible by ``num_blocks``.
    """

    def __init__(self, width, num_blocks, variance_scale=1.0):
        super().__init__()
        self.width = int(width)
        self.num_blocks = int(num_blocks)
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        if self.width % self.num_blocks != 0:
            raise ValueError(
                f"width ({self.width}) must be divisible by num_blocks ({self.num_blocks})"
            )
        self.block_width = self.width // self.num_blocks
        self.variance_scale = float(variance_scale)
        self.weight = nn.Parameter(
            torch.empty(self.num_blocks, self.block_width, self.block_width)
        )
        self.bias = nn.Parameter(torch.empty(self.num_blocks, self.block_width))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(self.variance_scale / self.block_width)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if x.shape[-1] != self.width:
            raise ValueError(
                f"expected final dimension {self.width}, got {x.shape[-1]}"
            )
        blocked = x.reshape(*x.shape[:-1], self.num_blocks, self.block_width)
        out = torch.einsum("...bi,bij->...bj", blocked, self.weight)
        out = out + self.bias
        return out.reshape(*x.shape[:-1], self.width)


class _SqrtBoundDerivative(torch.autograd.Function):
    """Square root whose backward derivative is capped for stability.

    Griffin's reference implementation caps the derivative at 1000 to avoid
    numerical failures when the dynamic recurrence approaches one.
    """

    max_gradient = 1000.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        clipped_x_times_four = torch.clamp(
            4.0 * x, min=1.0 / (_SqrtBoundDerivative.max_gradient**2)
        )
        return grad_output / torch.sqrt(clipped_x_times_four)


def _inverse_softplus(x):
    """Numerically stable inverse of softplus for strictly positive ``x``."""

    return x + torch.log(-torch.expm1(-x))


class RGLRURec(nn.Module):
    """Real-Gated LRU adapted to the shared CA-LRU block interface.

    The recurrence follows De et al., *Griffin: Mixing Gated Linear
    Recurrences with Local Attention for Efficient Language Models* (2024),
    and Google DeepMind's reference RecurrentGemma implementation
    (``google-deepmind/recurrentgemma`` commit ``2efa84d``,
    ``recurrentgemma/torch/layers.py``)::

        i_t = sigmoid(W_i x_t + b_i)
        r_t = sigmoid(W_r x_t + b_r)
        a_t = exp(-c * r_t * softplus(a_param))
        h_t = a_t * h_{t-1} + sqrt(1 - a_t**2) * (i_t * x_t)

    A learned input projection permits ``input_dim != hidden_dim`` while the
    recurrence itself remains diagonal. Gate maps are block diagonal, as in
    Griffin. The Conv1D and parallel branch from the language-model block are
    deliberately omitted because the experiment holds the outer block
    scaffold fixed and changes only its recurrent module.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_blocks=16,
        recurrence_scale=8.0,
        min_radius=0.90,
        max_radius=0.999,
        gate_variance_scale=1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_blocks = int(num_blocks)
        self.recurrence_scale = float(recurrence_scale)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        if self.input_dim <= 0 or self.hidden_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        if self.recurrence_scale <= 0.0:
            raise ValueError("recurrence_scale must be positive")
        if not (0.0 < self.min_radius < self.max_radius < 1.0):
            raise ValueError(
                "RG-LRU radii must satisfy 0 < min_radius < max_radius < 1"
            )

        # The reference recurrent block zero-initializes the projection bias.
        # A bias-free adapter preserves the stronger blank-input identity:
        # u_t=0 implies that the RG-LRU input contribution is exactly zero.
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.input_gate = BlockDiagonalLinear(
            self.hidden_dim,
            self.num_blocks,
            variance_scale=gate_variance_scale,
        )
        self.recurrence_gate = BlockDiagonalLinear(
            self.hidden_dim,
            self.num_blocks,
            variance_scale=gate_variance_scale,
        )
        self.a_param = nn.Parameter(torch.empty(self.hidden_dim))
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self.reset_parameters()

    @property
    def state_size(self):
        return self.hidden_dim

    def reset_parameters(self):
        nn.init.normal_(
            self.input_proj.weight,
            mean=0.0,
            std=math.sqrt(1.0 / self.input_dim),
        )
        self.input_gate.reset_parameters()
        self.recurrence_gate.reset_parameters()
        nn.init.normal_(
            self.out_proj.weight,
            mean=0.0,
            std=math.sqrt(1.0 / self.hidden_dim),
        )
        nn.init.zeros_(self.out_proj.bias)
        with torch.no_grad():
            # Sampling radius squared uniformly matches the ring-area
            # initialization used by the official implementation.
            radius_sq = torch.empty_like(self.a_param).uniform_(
                self.min_radius**2,
                self.max_radius**2,
            )
            radius = torch.sqrt(radius_sq)
            self.a_param.copy_(_inverse_softplus(-torch.log(radius)))

    def lam_mag(self):
        """Return the learned base radius before input-dependent gating."""

        return torch.exp(-F.softplus(self.a_param))

    def projected_input_and_gates(self, u_t):
        x_t = self.input_proj(u_t)
        input_gate = torch.sigmoid(self.input_gate(x_t))
        recurrence_gate = torch.sigmoid(self.recurrence_gate(x_t))
        return x_t, input_gate, recurrence_gate

    def step(self, u_t, state):
        x_t, input_gate, recurrence_gate = self.projected_input_and_gates(u_t)
        log_a = -self.recurrence_scale * recurrence_gate * F.softplus(self.a_param)
        a_t = torch.exp(log_a)
        input_scale = _SqrtBoundDerivative.apply(
            torch.clamp(1.0 - torch.exp(2.0 * log_a), min=0.0)
        )
        return a_t * state + input_scale * input_gate * x_t

    def output(self, state):
        return self.out_proj(state)


class GRURec(nn.Module):
    """Standard GRU recurrence behind the shared full-block scaffold."""

    def __init__(self, input_dim, hidden_dim, output_dim, keep_bias_init=0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        if self.input_dim <= 0 or self.hidden_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        self.cell = nn.GRUCell(self.input_dim, self.hidden_dim)
        with torch.no_grad():
            # PyTorch orders gates as reset, update, new. A positive update
            # bias increases retention under its h'=(1-z)n+z h convention.
            update = slice(self.hidden_dim, 2 * self.hidden_dim)
            self.cell.bias_ih[update].fill_(float(keep_bias_init))
            self.cell.bias_hh[update].zero_()
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

    @property
    def state_size(self):
        return self.hidden_dim

    def step(self, u_t, state):
        return self.cell(u_t, state)

    def output(self, state):
        return self.out_proj(state)


class RecurrentBlock(nn.Module):
    """Shared block wrapper around an interchangeable recurrent module."""

    def __init__(
        self,
        d_model,
        rec,
        dropout=0.0,
        use_norm_in=True,
        norm_in_affine=True,
        use_norm_out=True,
        norm_out_affine=True,
        update_mode="glu",
        use_residual=True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.rec = rec
        self.update_mode = str(update_mode)
        self.use_residual = bool(use_residual)
        self.norm_in = (
            nn.LayerNorm(self.d_model, elementwise_affine=bool(norm_in_affine))
            if use_norm_in
            else nn.Identity()
        )
        self.glu_proj = nn.Linear(self.d_model, 2 * self.d_model) if self.update_mode == "glu" else None
        self.dropout = nn.Dropout(float(dropout))
        self.norm_out = (
            nn.LayerNorm(self.d_model, elementwise_affine=bool(norm_out_affine))
            if use_norm_out
            else nn.Identity()
        )

    @property
    def state_size(self):
        return self.rec.state_size

    def step(self, stream, state):
        u_t = self.norm_in(stream)
        next_state = self.rec.step(u_t, state)
        rec_out = self.rec.output(next_state)
        if self.update_mode == "glu":
            gate_input = F.gelu(rec_out)
            update = F.glu(self.glu_proj(gate_input), dim=-1)
        elif self.update_mode == "gelu":
            update = F.gelu(rec_out)
        elif self.update_mode == "linear":
            update = rec_out
        else:
            raise ValueError(f"unknown update_mode: {self.update_mode}")
        stream_base = stream + self.dropout(update) if self.use_residual else self.dropout(update)
        next_stream = self.norm_out(stream_base)
        return next_stream, next_state

    def regularization_loss(self):
        if hasattr(self.rec, "regularization_loss"):
            return self.rec.regularization_loss()
        return None


class FullBlockSequenceModel(nn.Module):
    """Encoder -> stacked recurrent blocks -> output head.

    The public step/decode interface is compatible with the earlier StepModel
    diagnostics. The flat state contains all recurrent states plus the latest
    output stream, so decode(state) matches the most recent block output.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        variant,
        d_model=64,
        rec_dim=64,
        num_layers=2,
        dropout=0.0,
        plru_tau=DEFAULT_TAU,
        plru_c=DEFAULT_C,
        pan_lambda_min=0.90,
        pan_lambda_max=0.999,
        encoder_bias=True,
        use_norm_in=True,
        norm_in_affine=True,
        use_norm_out=True,
        norm_out_affine=True,
        update_mode="glu",
        use_residual=True,
        decode_mode="stream",
        carry_stream=False,
        rglru_num_blocks=16,
        rglru_recurrence_scale=8.0,
        rglru_min_radius=0.90,
        rglru_max_radius=0.999,
        gru_keep_bias_init=0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.variant = normalize_block_variant(variant)
        self.d_model = int(d_model)
        self.rec_dim = int(rec_dim)
        self.num_layers = int(num_layers)
        self.decode_mode = str(decode_mode)
        self.carry_stream = bool(carry_stream)
        self.rglru_num_blocks = int(rglru_num_blocks)
        self.rglru_recurrence_scale = float(rglru_recurrence_scale)
        self.rglru_min_radius = float(rglru_min_radius)
        self.rglru_max_radius = float(rglru_max_radius)
        self.gru_keep_bias_init = float(gru_keep_bias_init)

        self.encoder = nn.Linear(self.input_dim, self.d_model, bias=bool(encoder_bias))
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            rec = self._make_rec(
                plru_tau=float(plru_tau),
                plru_c=float(plru_c),
                pan_lambda_min=float(pan_lambda_min),
                pan_lambda_max=float(pan_lambda_max),
            )
            self.blocks.append(
                RecurrentBlock(
                    self.d_model,
                    rec,
                    dropout=dropout,
                    use_norm_in=use_norm_in,
                    norm_in_affine=norm_in_affine,
                    use_norm_out=use_norm_out,
                    norm_out_affine=norm_out_affine,
                    update_mode=update_mode,
                    use_residual=use_residual,
                )
            )
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.output_dim),
        )

        offset = 0
        self._rec_slices = []
        for block in self.blocks:
            next_offset = offset + block.state_size
            self._rec_slices.append(slice(offset, next_offset))
            offset = next_offset
        self.recurrent_state_size = offset
        self.readout_slice = slice(self.recurrent_state_size, self.recurrent_state_size + self.d_model)

    def _make_rec(self, plru_tau, plru_c, pan_lambda_min, pan_lambda_max):
        if self.variant == "LRU-Block":
            return ComplexLRURec(
                input_dim=self.d_model,
                hidden_dim=self.rec_dim,
                output_dim=self.d_model,
                r_min=0.90,
                r_max=0.999,
            )
        if self.variant == "near-one LRU-Block":
            return ComplexLRURec(
                input_dim=self.d_model,
                hidden_dim=self.rec_dim,
                output_dim=self.d_model,
                r_min=0.99,
                r_max=0.9999,
            )
        if self.variant == "real-diag LRU-Block":
            return RealDiagLRURec(
                input_dim=self.d_model,
                hidden_dim=self.rec_dim,
                output_dim=self.d_model,
                r_min=0.0,
                r_max=1.0,
                gamma_free=True,
            )
        if self.variant == "P-LRU-Block":
            return PLRURec(
                input_dim=self.d_model,
                hidden_dim=self.rec_dim,
                output_dim=self.d_model,
                tau=plru_tau,
                c=plru_c,
            )
        if self.variant == "PAN-Block":
            return PANRec(
                input_dim=self.d_model,
                hidden_dim=self.rec_dim,
                output_dim=self.d_model,
                lambda_min=pan_lambda_min,
                lambda_max=pan_lambda_max,
            )
        if self.variant == "PAN-NW-Block":
            return PANNonlinearWriterRec(
                input_dim=self.d_model,
                hidden_dim=self.rec_dim,
                output_dim=self.d_model,
                lambda_min=pan_lambda_min,
                lambda_max=pan_lambda_max,
                writer_mode="input",
            )
        if self.variant == "PAN-RNW-Block":
            return PANNonlinearWriterRec(
                input_dim=self.d_model,
                hidden_dim=self.rec_dim,
                output_dim=self.d_model,
                lambda_min=pan_lambda_min,
                lambda_max=pan_lambda_max,
                writer_mode="recurrent",
            )
        if self.variant == "RG-LRU-Block":
            return RGLRURec(
                input_dim=self.d_model,
                hidden_dim=self.rec_dim,
                output_dim=self.d_model,
                num_blocks=self.rglru_num_blocks,
                recurrence_scale=self.rglru_recurrence_scale,
                min_radius=self.rglru_min_radius,
                max_radius=self.rglru_max_radius,
            )
        if self.variant == "GRU-Block":
            return GRURec(
                input_dim=self.d_model,
                hidden_dim=self.rec_dim,
                output_dim=self.d_model,
                keep_bias_init=self.gru_keep_bias_init,
            )
        raise AssertionError(self.variant)

    @property
    def state_size(self):
        return self.recurrent_state_size + self.d_model

    def init_state(self, batch, device):
        return torch.zeros(int(batch), self.state_size, device=device)

    def split_recurrent_state(self, state):
        return [state[:, sl] for sl in self._rec_slices]

    def merge_state(self, rec_states, stream):
        return torch.cat(list(rec_states) + [stream], dim=-1)

    def step(self, x_t, state):
        stream = self.encoder(x_t)
        if self.carry_stream:
            stream = stream + state[:, self.readout_slice]
        rec_states = self.split_recurrent_state(state)
        next_rec_states = []
        for block, rec_state in zip(self.blocks, rec_states):
            stream, next_rec_state = block.step(stream, rec_state)
            next_rec_states.append(next_rec_state)
        return self.merge_state(next_rec_states, stream)

    def decode(self, state):
        if self.decode_mode == "stream":
            return self.head(state[:, self.readout_slice])
        if self.decode_mode == "last_rec":
            last_state = state[:, self._rec_slices[-1]]
            return self.head(self.blocks[-1].rec.output(last_state))
        raise ValueError(f"unknown decode_mode: {self.decode_mode}")

    def forward(self, x_seq, return_states=False):
        state = self.init_state(x_seq.shape[1], x_seq.device)
        outs = []
        states = []
        for x_t in x_seq:
            state = self.step(x_t, state)
            outs.append(self.decode(state))
            if return_states:
                states.append(state)
        out = torch.stack(outs, dim=0)
        if return_states:
            return out, torch.stack(states, dim=0)
        return out

    def regularization_loss(self):
        losses = [block.regularization_loss() for block in self.blocks]
        losses = [loss for loss in losses if loss is not None]
        if not losses:
            return None
        return sum(losses)

    def lam_mag(self):
        lams = []
        for block in self.blocks:
            if hasattr(block.rec, "lam_mag"):
                lams.append(block.rec.lam_mag())
        if not lams:
            return torch.empty(0, device=next(self.parameters()).device)
        return torch.cat(lams, dim=0)

    def pan_recs_with_slices(self):
        pairs = []
        for block, rec_slice in zip(self.blocks, self._rec_slices):
            if isinstance(block.rec, (PANRec, PANNonlinearWriterRec)):
                pairs.append((block.rec, rec_slice))
        return pairs


def build_block_model(
    variant,
    input_dim,
    output_dim,
    d_model=64,
    rec_dim=64,
    num_layers=2,
    dropout=0.0,
    plru_tau=DEFAULT_TAU,
    plru_c=DEFAULT_C,
    pan_lambda_min=0.90,
    pan_lambda_max=0.999,
    encoder_bias=True,
    use_norm_in=True,
    norm_in_affine=True,
    use_norm_out=True,
    norm_out_affine=True,
    update_mode="glu",
    use_residual=True,
    decode_mode="stream",
    carry_stream=False,
    rglru_num_blocks=16,
    rglru_recurrence_scale=8.0,
    rglru_min_radius=0.90,
    rglru_max_radius=0.999,
    gru_keep_bias_init=0.0,
):
    return FullBlockSequenceModel(
        input_dim=input_dim,
        output_dim=output_dim,
        variant=variant,
        d_model=d_model,
        rec_dim=rec_dim,
        num_layers=num_layers,
        dropout=dropout,
        plru_tau=plru_tau,
        plru_c=plru_c,
        pan_lambda_min=pan_lambda_min,
        pan_lambda_max=pan_lambda_max,
        encoder_bias=encoder_bias,
        use_norm_in=use_norm_in,
        norm_in_affine=norm_in_affine,
        use_norm_out=use_norm_out,
        norm_out_affine=norm_out_affine,
        update_mode=update_mode,
        use_residual=use_residual,
        decode_mode=decode_mode,
        carry_stream=carry_stream,
        rglru_num_blocks=rglru_num_blocks,
        rglru_recurrence_scale=rglru_recurrence_scale,
        rglru_min_radius=rglru_min_radius,
        rglru_max_radius=rglru_max_radius,
        gru_keep_bias_init=gru_keep_bias_init,
    )


@dataclass(frozen=True)
class ParameterMatch:
    """Result of a recurrent-width parameter matching search."""

    target_params: int
    candidate_params: int
    candidate_rec_dim: int
    relative_error: float
    target_trainable_params: int
    candidate_trainable_params: int


def count_trainable_parameters(model):
    """Count trainable scalar parameters without counting frozen RP state."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_model_parameters(model):
    """Count every stored parameter, including RP-updated frozen theta values."""

    return sum(parameter.numel() for parameter in model.parameters())


def find_parameter_matched_rec_dim(reference_model, candidate_factory, candidate_rec_dims):
    """Select the candidate recurrent width closest to ``reference_model``.

    Args:
        reference_model: Instantiated target model (normally CA-LRU).
        candidate_factory: Callable taking an integer recurrent width and
            returning an instantiated candidate model.
        candidate_rec_dims: Finite iterable of positive widths to search. For
            RG-LRU these should be divisible by the requested gate-block count.

    Returns:
        :class:`ParameterMatch`, with ties resolved toward the smaller width.

    Candidate construction is isolated with ``torch.random.fork_rng`` so this
    bookkeeping helper does not perturb experiment initialization.
    """

    target_params = count_model_parameters(reference_model)
    target_trainable_params = count_trainable_parameters(reference_model)
    if target_params <= 0:
        raise ValueError("reference_model has no trainable parameters")
    widths = sorted(set(int(width) for width in candidate_rec_dims))
    if not widths or widths[0] <= 0:
        raise ValueError("candidate_rec_dims must contain positive integers")

    best = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for width in widths:
            candidate = candidate_factory(width)
            candidate_params = count_model_parameters(candidate)
            candidate_trainable_params = count_trainable_parameters(candidate)
            error = abs(candidate_params - target_params) / target_params
            match = ParameterMatch(
                target_params=target_params,
                candidate_params=candidate_params,
                candidate_rec_dim=width,
                relative_error=float(error),
                target_trainable_params=target_trainable_params,
                candidate_trainable_params=candidate_trainable_params,
            )
            if best is None or (match.relative_error, width) < (
                best.relative_error,
                best.candidate_rec_dim,
            ):
                best = match
            del candidate
    return best
