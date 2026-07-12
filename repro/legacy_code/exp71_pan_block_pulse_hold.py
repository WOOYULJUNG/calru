"""Exp71: full-block pulse-hold scaffold for PAN-Block.

This is the first architecture-level track after the Exp69 simplified
recurrence diagnostics. The scaffold is fixed:

    Encoder -> stacked recurrent blocks -> OutputHead

and the recurrent module selects:

    LRU-Block, near-one LRU-Block, P-LRU-Block, PAN-Block
"""

import argparse
import contextlib
import csv
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pan_block import BLOCK_VARIANTS, build_block_model, normalize_block_variant
from plru_regularizers import DEFAULT_C, DEFAULT_TAU, DEFAULT_WARMUP_FRAC, canonical_q_loss_from_lam


HOLD_HORIZONS = [50, 100, 200, 500, 1000, 2000]
UPDATE_COUNTS = [1, 3, 5, 10]
RECOVERY_STEPS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
TANGENT_STEPS = [0, 20, 100, 500]
MODEL_COLORS = {
    "RNN": "#9CA3AF",
    "Leaky RNN": "#6B7280",
    "RNN-PAN": "#0EA5E9",
    "PAN-RNN": "#0EA5E9",
    "GRU": "#4C78A8",
    "GRU keep1": "#7CA7D9",
    "GRU keep2": "#5B8FD1",
    "GRU-PAN": "#1D4ED8",
    "PAN-GRU": "#1D4ED8",
    "LSTM": "#A855F7",
    "LSTM forget1": "#C084FC",
    "LSTM forget2": "#A855F7",
    "LSTM-PAN": "#7E22CE",
    "SSM": "#54A24B",
    "LRU unit": "#F59E0B",
    "LRU full": "#D97706",
    "real-diag unit": "#EF4444",
    "rank-matched LRU unit": "#92400E",
    "real-diag full": "#B91C1C",
    "rank-matched real-diag full": "#7F1D1D",
    "rm-real full no-enc-bias": "#991B1B",
    "rm-real full no-ln": "#BE123C",
    "rm-real full no-glu": "#E11D48",
    "rm-real full no-residual": "#F43F5E",
    "rm-real full rec-decode": "#FB7185",
    "rm-real full zero-drive": "#DC2626",
    "rm-real full zero-input-ln": "#B45309",
    "rm-real full minimal": "#881337",
    "LRU-Block": "#F58518",
    "near-one LRU-Block": "#6B7280",
    "rank-matched LRU-Block": "#111827",
    "P-LRU unit": "#2DD4BF",
    "P-LRU-Block": "#14B8A6",
    "PAN-unit": "#60A5FA",
    "PAN-full": "#2563EB",
    "PAN-NW-full": "#38BDF8",
    "PAN-RNW-full": "#0F766E",
}
MODEL_DISPLAY_NAMES = {
    "Leaky RNN": "Leaky RNN",
    "RNN-PAN": "RNN-PAN",
    "PAN-RNN": "RNN-PAN",
    "GRU-PAN": "GRU-PAN",
    "PAN-GRU": "GRU-PAN",
    "GRU keep1": "GRU keep-bias",
    "GRU keep2": "GRU keep-bias",
    "LSTM-PAN": "LSTM-PAN",
    "LSTM forget1": "LSTM forget-bias",
    "LSTM forget2": "LSTM forget-bias",
    "LRU unit": "LRU-pure unit",
    "LRU-Block": "LRU-pure full",
    "P-LRU-Block": "P-LRU full",
    "rm-real full zero-drive": "rank-matched LRU full",
    "PAN-NW-full": "AM-LRU-NW",
    "PAN-RNW-full": "AM-LRU-RNW",
}
BASELINE_VARIANTS = (
    "RNN",
    "Leaky RNN",
    "RNN-PAN",
    "GRU",
    "GRU keep1",
    "GRU keep2",
    "GRU-PAN",
    "LSTM",
    "LSTM forget1",
    "LSTM forget2",
    "LSTM-PAN",
    "SSM",
)
MODEL_VARIANTS = BASELINE_VARIANTS + (
    "LRU unit",
    "LRU full",
    "real-diag unit",
    "rank-matched LRU unit",
    "real-diag full",
    "rank-matched real-diag full",
    "rm-real full no-enc-bias",
    "rm-real full no-ln",
    "rm-real full no-glu",
    "rm-real full no-residual",
    "rm-real full rec-decode",
    "rm-real full zero-drive",
    "rm-real full zero-input-ln",
    "rm-real full minimal",
    "LRU-Block",
    "near-one LRU-Block",
    "rank-matched LRU-Block",
    "P-LRU unit",
    "P-LRU-Block",
    "PAN-unit",
    "PAN-full",
    "PAN-NW-full",
    "PAN-RNW-full",
)
_PLT = None
_PLOT_IMPORT_FAILED = False

RM_REAL_FULL_WRAPPER_CONFIGS = {
    "rank-matched real-diag full": {},
    "rm-real full no-enc-bias": {"encoder_bias": False},
    "rm-real full no-ln": {"use_norm_in": False, "use_norm_out": False},
    "rm-real full no-glu": {"update_mode": "gelu"},
    "rm-real full no-residual": {"use_residual": False},
    "rm-real full rec-decode": {"decode_mode": "last_rec"},
    "rm-real full zero-drive": {"encoder_bias": False, "use_norm_in": False},
    "rm-real full zero-input-ln": {"encoder_bias": False, "norm_in_affine": False},
    "rm-real full minimal": {
        "encoder_bias": False,
        "use_norm_in": False,
        "norm_in_affine": True,
        "use_norm_out": False,
        "update_mode": "linear",
        "use_residual": False,
        "decode_mode": "last_rec",
    },
}
LRU_FULL_WRAPPER_CONFIGS = {
    "LRU full": {"encoder_bias": False, "use_norm_in": False},
}
PAN_WRAPPER_CONFIGS = {
    "PAN-full": {"encoder_bias": False, "use_norm_in": False},
    "PAN-NW-full": {"encoder_bias": False, "use_norm_in": False},
    "PAN-RNW-full": {"encoder_bias": False, "use_norm_in": False},
}
WRAPPER_CONFIGS = {**RM_REAL_FULL_WRAPPER_CONFIGS, **LRU_FULL_WRAPPER_CONFIGS, **PAN_WRAPPER_CONFIGS}


def is_pan_variant(variant):
    return variant in (
        "PAN-unit",
        "PAN-full",
        "PAN-NW-full",
        "PAN-RNW-full",
        "PAN-Block",
        "PAN-NW-Block",
        "PAN-RNW-Block",
        "RNN-PAN",
        "PAN-RNN",
        "GRU-PAN",
        "PAN-GRU",
        "LSTM-PAN",
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_pyplot():
    global _PLT, _PLOT_IMPORT_FAILED
    if _PLT is not None:
        return _PLT
    if _PLOT_IMPORT_FAILED:
        return None
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

        _PLT = plt
        return _PLT
    except Exception as exc:
        _PLOT_IMPORT_FAILED = True
        print(f"plotting disabled: matplotlib import failed: {exc}", flush=True)
        return None


def slugify(text):
    text = text.replace("P-LRU", "PLRU")
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def float_slug(value):
    text = f"{float(value):.4g}".replace("-", "m").replace(".", "p")
    return text


def normalize_model_variant(name):
    key = re.sub(r"[\s_]+", "-", name.strip().lower())
    aliases = {
        "rnn": "RNN",
        "vanilla-rnn": "RNN",
        "leaky-rnn": "Leaky RNN",
        "leakyrnn": "Leaky RNN",
        "rnn-pan": "RNN-PAN",
        "pan-rnn": "RNN-PAN",
        "pan-leaky-rnn": "RNN-PAN",
        "pan-leakyrnn": "RNN-PAN",
        "gru": "GRU",
        "gru-keep1": "GRU keep1",
        "gru-keep2": "GRU keep2",
        "gru-high-keep": "GRU keep1",
        "gru-keep-bias": "GRU keep1",
        "gru-pan": "GRU-PAN",
        "pan-gru": "GRU-PAN",
        "pan-gated-gru": "GRU-PAN",
        "lstm": "LSTM",
        "lstm-forget1": "LSTM forget1",
        "lstm-forget2": "LSTM forget2",
        "lstm-high-forget": "LSTM forget1",
        "lstm-forget-bias": "LSTM forget1",
        "lstm-pan": "LSTM-PAN",
        "pan-lstm": "LSTM-PAN",
        "ssm": "SSM",
        "lru-unit": "LRU unit",
        "lru-pure-unit": "LRU unit",
        "pure-lru-unit": "LRU unit",
        "unit-lru": "LRU unit",
        "lru-full": "LRU full",
        "full-lru": "LRU full",
        "corrected-lru-full": "LRU full",
        "zero-drive-lru-full": "LRU full",
        "real-diag-unit": "real-diag unit",
        "realdiag-unit": "real-diag unit",
        "real-diagonal-unit": "real-diag unit",
        "real-lru-unit": "real-diag unit",
        "real-diag-full": "real-diag full",
        "realdiag-full": "real-diag full",
        "real-diagonal-full": "real-diag full",
        "real-lru-full": "real-diag full",
        "rank-matched-real-diag-full": "rank-matched real-diag full",
        "rankmatched-real-diag-full": "rank-matched real-diag full",
        "rank-matched-realdiag-full": "rank-matched real-diag full",
        "rankmatched-realdiag-full": "rank-matched real-diag full",
        "rm-real-diag-full": "rank-matched real-diag full",
        "real-rm-full": "rank-matched real-diag full",
        "rm-real-full": "rank-matched real-diag full",
        "rm-real-full-no-enc-bias": "rm-real full no-enc-bias",
        "rm-real-full-no-encoder-bias": "rm-real full no-enc-bias",
        "rm-real-full-no-ln": "rm-real full no-ln",
        "rm-real-full-no-layernorm": "rm-real full no-ln",
        "rm-real-full-no-glu": "rm-real full no-glu",
        "rm-real-full-no-residual": "rm-real full no-residual",
        "rm-real-full-rec-decode": "rm-real full rec-decode",
        "rm-real-full-recurrent-decode": "rm-real full rec-decode",
        "rm-real-full-zero-drive": "rm-real full zero-drive",
        "rm-real-full-zero-input-ln": "rm-real full zero-input-ln",
        "rm-real-full-zero-input-layernorm": "rm-real full zero-input-ln",
        "rm-real-full-minimal": "rm-real full minimal",
        "pan": "PAN-full",
        "pan-block": "PAN-full",
        "pan-full": "PAN-full",
        "pan-nw": "PAN-NW-full",
        "pan-nw-full": "PAN-NW-full",
        "am-lru-nw": "PAN-NW-full",
        "amlru-nw": "PAN-NW-full",
        "pan-nonlinear-writer": "PAN-NW-full",
        "pan-nonlinear-writer-full": "PAN-NW-full",
        "pan-rnw": "PAN-RNW-full",
        "pan-rnw-full": "PAN-RNW-full",
        "am-lru-rnw": "PAN-RNW-full",
        "amlru-rnw": "PAN-RNW-full",
        "pan-recurrent-writer": "PAN-RNW-full",
        "pan-recurrent-writer-full": "PAN-RNW-full",
        "pan-unit": "PAN-unit",
        "rank-matched-lru-unit": "rank-matched LRU unit",
        "rankmatched-lru-unit": "rank-matched LRU unit",
        "rm-lru-unit": "rank-matched LRU unit",
        "rank-matched-real-lru-unit": "rank-matched LRU unit",
        "rankmatched-real-lru-unit": "rank-matched LRU unit",
        "rm-real-lru-unit": "rank-matched LRU unit",
        "real-rm-lru-unit": "rank-matched LRU unit",
        "lru-block-pure": "LRU-Block",
        "lru-pure-full": "LRU-Block",
        "pure-lru-full": "LRU-Block",
        "rank-matched-lru": "rank-matched LRU-Block",
        "rank-matched-lru-block": "rank-matched LRU-Block",
        "rank-matched-lru-full": "rank-matched LRU-Block",
        "rankmatched-lru": "rank-matched LRU-Block",
        "rankmatched-lru-block": "rank-matched LRU-Block",
        "rankmatched-lru-full": "rank-matched LRU-Block",
        "rm-lru": "rank-matched LRU-Block",
        "rm-lru-block": "rank-matched LRU-Block",
        "rm-lru-full": "rank-matched LRU-Block",
        "p-lru-unit": "P-LRU unit",
        "plru-unit": "P-LRU unit",
        "p-lru-full": "P-LRU-Block",
        "plru-full": "P-LRU-Block",
    }
    if key in aliases:
        return aliases[key]
    return normalize_block_variant(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class StepBaseline(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    @property
    def state_size(self):
        raise NotImplementedError

    def init_state(self, batch, device):
        return torch.zeros(int(batch), self.state_size, device=device)

    def step(self, x_t, state):
        raise NotImplementedError

    def decode(self, state):
        raise NotImplementedError

    def forward(self, x_seq, return_states=False):
        state = self.init_state(x_seq.shape[1], x_seq.device)
        outs, states = [], []
        for x_t in x_seq:
            state = self.step(x_t, state)
            outs.append(self.decode(state))
            if return_states:
                states.append(state)
        out = torch.stack(outs, dim=0)
        if return_states:
            return out, torch.stack(states, dim=0)
        return out


class MLPReadout(nn.Module):
    def __init__(self, state_size, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, state):
        return self.net(state)


class RNNBaseline(StepBaseline):
    def __init__(self, input_dim, output_dim, hidden):
        super().__init__(input_dim, output_dim)
        self.hidden = int(hidden)
        self.cell = nn.RNNCell(input_dim, self.hidden, nonlinearity="tanh")
        self.readout = MLPReadout(self.hidden, output_dim)

    @property
    def state_size(self):
        return self.hidden

    def step(self, x_t, state):
        return self.cell(x_t, state)

    def decode(self, state):
        return self.readout(state)


class LeakyRNNBaseline(StepBaseline):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden,
        lambda_min=0.90,
        lambda_max=0.999,
        pan_theta=False,
        decoupled_write=False,
        gamma_init=0.3,
    ):
        super().__init__(input_dim, output_dim)
        self.hidden = int(hidden)
        self.decoupled_write = bool(decoupled_write)
        self.in_proj = nn.Linear(input_dim, self.hidden, bias=True)
        self.rec_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        lam = torch.linspace(float(lambda_max), float(lambda_min), self.hidden).clamp(1e-4, 0.9999)
        self.theta = nn.Parameter(_pan_lambda_to_theta(lam), requires_grad=not bool(pan_theta))
        if self.decoupled_write:
            init = torch.full((self.hidden,), float(gamma_init)).clamp(1e-4, 1.0 - 1e-4)
            self.gamma_raw = nn.Parameter(torch.logit(init))
        else:
            self.gamma_raw = None
        self.readout = MLPReadout(self.hidden, output_dim)

    @property
    def state_size(self):
        return self.hidden

    def lam_mag(self):
        q = torch.sigmoid(self.theta).clamp(1e-8, 1.0 - 1e-8)
        return torch.sqrt(q)

    def step(self, x_t, state):
        candidate = torch.tanh(self.in_proj(x_t) + self.rec_proj(state))
        lam = self.lam_mag()
        if self.gamma_raw is not None:
            return lam * state + torch.sigmoid(self.gamma_raw).unsqueeze(0) * candidate
        return lam * state + (1.0 - lam) * candidate

    def decode(self, state):
        return self.readout(state)


class PANRNNBaseline(LeakyRNNBaseline):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden,
        lambda_min=0.90,
        lambda_max=0.999,
        decoupled_write=False,
        gamma_init=0.3,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden=hidden,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            pan_theta=True,
            decoupled_write=decoupled_write,
            gamma_init=gamma_init,
        )

    def pan_recs_with_slices(self):
        return [(self, slice(0, self.hidden))]

    @torch.no_grad()
    def update_theta(self, scores, eta_lambda):
        delta = torch.as_tensor(scores, dtype=self.theta.dtype, device=self.theta.device)
        if delta.shape != self.theta.shape:
            raise ValueError(f"PAN score shape {tuple(delta.shape)} does not match theta {tuple(self.theta.shape)}")
        self.theta.add_(float(eta_lambda) * delta)
        self.theta.clamp_(-18.0, 18.0)


class GRUBaseline(StepBaseline):
    def __init__(self, input_dim, output_dim, hidden, keep_bias_init=0.0):
        super().__init__(input_dim, output_dim)
        self.hidden = int(hidden)
        self.cell = nn.GRUCell(input_dim, self.hidden)
        with torch.no_grad():
            self.cell.bias_ih[self.hidden : 2 * self.hidden].fill_(float(keep_bias_init))
            self.cell.bias_hh[self.hidden : 2 * self.hidden].zero_()
        self.readout = MLPReadout(self.hidden, output_dim)

    @property
    def state_size(self):
        return self.hidden

    def step(self, x_t, state):
        return self.cell(x_t, state)

    def decode(self, state):
        return self.readout(state)


class PANGRUBaseline(StepBaseline):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden,
        lambda_min=0.90,
        lambda_max=0.999,
        alpha_form="upper_cap",
        keep_bias_init=0.0,
        reset_bias_init=0.0,
    ):
        super().__init__(input_dim, output_dim)
        self.hidden = int(hidden)
        self.alpha_form = str(alpha_form)
        self.input_proj = nn.Linear(input_dim, 3 * self.hidden)
        self.hidden_proj = nn.Linear(self.hidden, 3 * self.hidden, bias=False)
        with torch.no_grad():
            self.input_proj.bias[self.hidden : 2 * self.hidden].fill_(float(keep_bias_init))
            self.input_proj.bias[: self.hidden].fill_(float(reset_bias_init))
        lam = torch.linspace(float(lambda_max), float(lambda_min), self.hidden).clamp(1e-4, 1.0 - 1e-4)
        self.theta = nn.Parameter(_pan_lambda_to_theta(lam), requires_grad=False)
        self.readout = MLPReadout(self.hidden, output_dim)

    @property
    def state_size(self):
        return self.hidden

    def lam_mag(self):
        q = torch.sigmoid(self.theta).clamp(1e-8, 1.0 - 1e-8)
        return torch.sqrt(q)

    def step(self, x_t, state):
        xi_r, xi_z, xi_n = self.input_proj(x_t).chunk(3, dim=-1)
        hh_r, hh_z, hh_n = self.hidden_proj(state).chunk(3, dim=-1)
        r = torch.sigmoid(xi_r + hh_r)
        k_logits = xi_z + hh_z
        k = torch.sigmoid(k_logits)
        n = torch.tanh(xi_n + r * hh_n)
        lam = self.lam_mag().unsqueeze(0)
        if self.alpha_form in ("upper_cap", "lambda*k", "lambda_times_gate"):
            alpha = lam * k
        elif self.alpha_form in ("lower_bound", "lambda+(1-lambda)*k", "lambda_plus"):
            alpha = lam + (1.0 - lam) * k
        elif self.alpha_form in ("gate_bias", "bias", "lambda_bias"):
            lam_logit = torch.logit(lam.clamp(1e-6, 1.0 - 1e-6))
            alpha = torch.sigmoid(k_logits + lam_logit)
        else:
            raise ValueError(f"unknown GRU-PAN alpha form: {self.alpha_form}")
        return alpha * state + (1.0 - alpha) * n

    def decode(self, state):
        return self.readout(state)

    def pan_recs_with_slices(self):
        return [(self, slice(0, self.hidden))]

    @torch.no_grad()
    def update_theta(self, scores, eta_lambda):
        delta = torch.as_tensor(scores, dtype=self.theta.dtype, device=self.theta.device)
        if delta.shape != self.theta.shape:
            raise ValueError(f"PAN score shape {tuple(delta.shape)} does not match theta {tuple(self.theta.shape)}")
        self.theta.add_(float(eta_lambda) * delta)
        self.theta.clamp_(-18.0, 18.0)


class LSTMBaseline(StepBaseline):
    def __init__(self, input_dim, output_dim, hidden, forget_bias_init=0.0):
        super().__init__(input_dim, output_dim)
        self.hidden = int(hidden)
        self.cell = nn.LSTMCell(input_dim, self.hidden)
        with torch.no_grad():
            self.cell.bias_ih[self.hidden : 2 * self.hidden].fill_(float(forget_bias_init))
            self.cell.bias_hh[self.hidden : 2 * self.hidden].zero_()
        self.readout = MLPReadout(self.hidden, output_dim)

    @property
    def state_size(self):
        return 2 * self.hidden

    def step(self, x_t, state):
        h, c = state.split(self.hidden, dim=-1)
        h, c = self.cell(x_t, (h, c))
        return torch.cat([h, c], dim=-1)

    def decode(self, state):
        return self.readout(state[:, : self.hidden])


class PANLSTMBaseline(StepBaseline):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden,
        lambda_min=0.90,
        lambda_max=0.999,
        alpha_form="upper_cap",
        forget_bias_init=1.0,
        input_bias_init=0.0,
        ablation_state="c_only",
    ):
        super().__init__(input_dim, output_dim)
        self.hidden = int(hidden)
        self.alpha_form = str(alpha_form)
        self.ablation_state = str(ablation_state)
        self.input_proj = nn.Linear(input_dim, 4 * self.hidden)
        self.hidden_proj = nn.Linear(self.hidden, 4 * self.hidden, bias=False)
        with torch.no_grad():
            self.input_proj.bias[: self.hidden].fill_(float(forget_bias_init))
            self.input_proj.bias[self.hidden : 2 * self.hidden].fill_(float(input_bias_init))
        lam = torch.linspace(float(lambda_max), float(lambda_min), self.hidden).clamp(1e-4, 1.0 - 1e-4)
        self.theta = nn.Parameter(_pan_lambda_to_theta(lam), requires_grad=False)
        self.readout = MLPReadout(self.hidden, output_dim)

    @property
    def state_size(self):
        return 2 * self.hidden

    def lam_mag(self):
        q = torch.sigmoid(self.theta).clamp(1e-8, 1.0 - 1e-8)
        return torch.sqrt(q)

    def step(self, x_t, state):
        h, c = state.split(self.hidden, dim=-1)
        xi_f, xi_i, xi_o, xi_g = self.input_proj(x_t).chunk(4, dim=-1)
        hh_f, hh_i, hh_o, hh_g = self.hidden_proj(h).chunk(4, dim=-1)
        f_logits = xi_f + hh_f
        f = torch.sigmoid(f_logits)
        i = torch.sigmoid(xi_i + hh_i)
        o = torch.sigmoid(xi_o + hh_o)
        g = torch.tanh(xi_g + hh_g)
        lam = self.lam_mag().unsqueeze(0)
        if self.alpha_form in ("upper_cap", "lambda*f", "lambda_times_gate"):
            alpha = lam * f
        elif self.alpha_form in ("lower_bound", "lambda+(1-lambda)*f", "lambda_plus"):
            alpha = lam + (1.0 - lam) * f
        elif self.alpha_form in ("gate_bias", "forget_bias", "bias", "lambda_bias"):
            lam_logit = torch.logit(lam.clamp(1e-6, 1.0 - 1e-6))
            alpha = torch.sigmoid(f_logits + lam_logit)
        else:
            raise ValueError(f"unknown LSTM-PAN alpha form: {self.alpha_form}")
        c_new = alpha * c + i * g
        h_new = o * torch.tanh(c_new)
        return torch.cat([h_new, c_new], dim=-1)

    def decode(self, state):
        return self.readout(state[:, : self.hidden])

    def pan_recs_with_slices(self):
        return [(self, slice(self.hidden, 2 * self.hidden))]

    def ablate_pan_coordinates(self, state, rec_slice):
        batch, total_state = state.shape
        ablated = state.unsqueeze(0).expand(self.hidden, batch, total_state).clone()
        idx = torch.arange(self.hidden, device=state.device)
        if self.ablation_state in ("h_and_c", "hc", "both"):
            ablated[idx, :, idx] = 0.0
        elif self.ablation_state not in ("c_only", "c"):
            raise ValueError(f"unknown LSTM-PAN ablation state: {self.ablation_state}")
        ablated[idx, :, rec_slice.start + idx] = 0.0
        return ablated

    @torch.no_grad()
    def update_theta(self, scores, eta_lambda):
        delta = torch.as_tensor(scores, dtype=self.theta.dtype, device=self.theta.device)
        if delta.shape != self.theta.shape:
            raise ValueError(f"PAN score shape {tuple(delta.shape)} does not match theta {tuple(self.theta.shape)}")
        self.theta.add_(float(eta_lambda) * delta)
        self.theta.clamp_(-18.0, 18.0)


class DiagSSMBaseline(StepBaseline):
    def __init__(self, input_dim, output_dim, hidden):
        super().__init__(input_dim, output_dim)
        self.hidden = int(hidden)
        self.a_raw = nn.Parameter(torch.full((self.hidden,), 2.2))
        self.B = nn.Linear(input_dim, self.hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.hidden))
        self.readout = MLPReadout(self.hidden, output_dim)

    @property
    def state_size(self):
        return self.hidden

    def lam_mag(self):
        return torch.sigmoid(self.a_raw)

    def step(self, x_t, state):
        a = self.lam_mag()
        return a * state + self.B(x_t) + self.bias

    def decode(self, state):
        return self.readout(state)


class ComplexLRUUnitBaseline(StepBaseline):
    def __init__(
        self,
        input_dim,
        output_dim,
        complex_dim,
        r_min=0.90,
        r_max=0.999,
        max_phase=2.0 * math.pi,
        gamma_mode="tied",
    ):
        super().__init__(input_dim, output_dim)
        self.n = int(complex_dim)
        self.gamma_mode = gamma_mode
        u = torch.linspace(float(r_max), float(r_min), self.n)
        u = u.clamp(1e-4, 0.9999)
        self.nu = nn.Parameter(torch.log(-torch.log(u)))
        self.phase = nn.Parameter(torch.rand(self.n) * float(max_phase))
        scale = 1.0 / math.sqrt(2.0 * self.input_dim)
        self.B_re = nn.Parameter(torch.randn(self.n, self.input_dim) * scale)
        self.B_im = nn.Parameter(torch.randn(self.n, self.input_dim) * scale)
        if gamma_mode == "free":
            self.gamma_raw = nn.Parameter(torch.zeros(self.n))
        self.readout = MLPReadout(2 * self.n, output_dim)

    @property
    def state_size(self):
        return 2 * self.n

    def lam_mag(self):
        return torch.exp(-torch.exp(self.nu)).clamp(0.0, 0.9999)

    def gamma(self):
        mag = self.lam_mag()
        if self.gamma_mode == "tied":
            return torch.sqrt(torch.clamp(1.0 - mag.square(), min=1e-6))
        if self.gamma_mode == "free":
            return torch.sigmoid(self.gamma_raw)
        return torch.ones_like(mag)

    def step(self, x_t, state):
        hr, hi = state.chunk(2, dim=-1)
        mag = self.lam_mag()
        a = mag * torch.cos(self.phase)
        b = mag * torch.sin(self.phase)
        gamma = self.gamma()
        wr = x_t @ self.B_re.t()
        wi = x_t @ self.B_im.t()
        hr_next = a * hr - b * hi + gamma * wr
        hi_next = b * hr + a * hi + gamma * wi
        return torch.cat([hr_next, hi_next], dim=-1)

    def decode(self, state):
        return self.readout(state)


class RealDiagLRUUnitBaseline(StepBaseline):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden,
        r_min=0.0,
        r_max=1.0,
        gamma_free=True,
    ):
        super().__init__(input_dim, output_dim)
        self.hidden = int(hidden)
        u = torch.rand(self.hidden) * (float(r_max) - float(r_min)) + float(r_min)
        u = u.clamp(1e-4, 0.9999)
        self.nu = nn.Parameter(torch.log(-torch.log(u)))
        scale = 1.0 / math.sqrt(self.input_dim)
        self.B = nn.Parameter(torch.randn(self.hidden, self.input_dim) * scale)
        self.gamma_raw = nn.Parameter(torch.zeros(self.hidden)) if gamma_free else None
        self.readout = MLPReadout(self.hidden, output_dim)

    @property
    def state_size(self):
        return self.hidden

    def lam_mag(self):
        return torch.exp(-torch.exp(self.nu)).clamp(0.0, 0.9999)

    def gamma(self):
        if self.gamma_raw is None:
            return torch.ones_like(self.lam_mag())
        return torch.sigmoid(self.gamma_raw)

    def step(self, x_t, state):
        return self.lam_mag() * state + self.gamma() * (x_t @ self.B.t())

    def decode(self, state):
        return self.readout(state)


class PLRUUnitBaseline(RealDiagLRUUnitBaseline):
    """Phase-zero real diagonal unit with canonical P-LRU regularization."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden,
        tau=DEFAULT_TAU,
        c=DEFAULT_C,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden=hidden,
            r_min=0.0,
            r_max=1.0,
            gamma_free=True,
        )
        self.tau = float(tau)
        self.c = float(c)

    def regularization_loss(self):
        return canonical_q_loss_from_lam(self.lam_mag(), tau=self.tau, c=self.c)


def _pan_lambda_to_theta(lam):
    q = lam.square().clamp(1e-8, 1.0 - 1e-8)
    return torch.logit(q)


class PANUnitBaseline(StepBaseline):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden,
        lambda_min=0.90,
        lambda_max=0.999,
        gamma_free=True,
    ):
        super().__init__(input_dim, output_dim)
        self.hidden = int(hidden)
        lam = torch.linspace(float(lambda_max), float(lambda_min), self.hidden).clamp(1e-4, 0.9999)
        self.theta = nn.Parameter(_pan_lambda_to_theta(lam), requires_grad=False)
        scale = 1.0 / math.sqrt(self.input_dim)
        self.B = nn.Parameter(torch.randn(self.hidden, self.input_dim) * scale)
        self.gamma_raw = nn.Parameter(torch.zeros(self.hidden)) if gamma_free else None
        self.readout = MLPReadout(self.hidden, output_dim)

    @property
    def state_size(self):
        return self.hidden

    def lam_mag(self):
        q = torch.sigmoid(self.theta).clamp(1e-8, 1.0 - 1e-8)
        return torch.sqrt(q)

    def gamma(self):
        if self.gamma_raw is None:
            return torch.ones_like(self.lam_mag())
        return torch.sigmoid(self.gamma_raw)

    def step(self, x_t, state):
        return self.lam_mag() * state + self.gamma() * (x_t @ self.B.t())

    def decode(self, state):
        return self.readout(state)

    def pan_recs_with_slices(self):
        return [(self, slice(0, self.hidden))]

    @torch.no_grad()
    def update_theta(self, scores, eta_lambda):
        delta = torch.as_tensor(scores, dtype=self.theta.dtype, device=self.theta.device)
        if delta.shape != self.theta.shape:
            raise ValueError(f"PAN score shape {tuple(delta.shape)} does not match theta {tuple(self.theta.shape)}")
        self.theta.add_(float(eta_lambda) * delta)
        self.theta.clamp_(-18.0, 18.0)


def set_rank_matched_lambdas(model, rank, lambda_high=0.999, lambda_low=0.0, freeze=True):
    if hasattr(model, "blocks"):
        recs = [block.rec for block in model.blocks if hasattr(block.rec, "nu")]
    elif hasattr(model, "nu"):
        recs = [model]
    else:
        return model
    total = sum(int(rec.nu.numel()) for rec in recs)
    keep = max(0, min(int(rank), total))
    values = torch.full((total,), float(lambda_low))
    if keep > 0:
        values[:keep] = float(lambda_high)
    values = values.clamp(1e-4, 0.9999)
    theta = torch.log(-torch.log(values))
    offset = 0
    for rec in recs:
        width = int(rec.nu.numel())
        with torch.no_grad():
            rec.nu.copy_(theta[offset : offset + width].to(rec.nu.device, dtype=rec.nu.dtype))
        if freeze:
            rec.nu.requires_grad_(False)
        offset += width
    return model


def _lambda_to_theta(lam):
    q = lam.square().clamp(1e-8, 1.0 - 1e-8)
    return torch.logit(q)


def _slow_lambda_recs(model):
    if hasattr(model, "blocks"):
        return [block.rec for block in model.blocks if hasattr(block.rec, "nu") or hasattr(block.rec, "theta")]
    if hasattr(model, "nu") or hasattr(model, "theta"):
        return [model]
    return []


def apply_slow_lambda_init(
    model,
    mode="default",
    lambda_min=0.90,
    lambda_max=0.999,
    lambda_fixed=0.999,
    lambda_mean=0.5,
    lambda_std=0.2,
    lambda_low_mean=0.05,
    lambda_high_mean=0.95,
    lambda_high_prob=0.5,
    shuffle=False,
    seed=-1,
):
    mode = str(mode)
    if mode == "default":
        return model
    recs = _slow_lambda_recs(model)
    total = 0
    for rec in recs:
        param = rec.nu if hasattr(rec, "nu") else rec.theta
        total += int(param.numel())
    if total <= 0:
        return model
    lo = float(lambda_min)
    hi = float(lambda_max)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if mode == "linspace":
        values = torch.linspace(hi, lo, total, device=device, dtype=dtype)
    elif mode == "random":
        if int(seed) >= 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            values = torch.rand(total, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
        else:
            values = torch.rand(total, device=device, dtype=dtype)
        values = values * (hi - lo) + lo
    elif mode == "gaussian":
        if int(seed) >= 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            values = torch.randn(total, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
        else:
            values = torch.randn(total, device=device, dtype=dtype)
        values = values * float(lambda_std) + float(lambda_mean)
    elif mode == "bimodal":
        if int(seed) >= 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            choose_high = torch.rand(total, generator=gen, dtype=torch.float32) < float(lambda_high_prob)
            noise = torch.randn(total, generator=gen, dtype=torch.float32)
            values = torch.where(
                choose_high,
                torch.full((total,), float(lambda_high_mean)),
                torch.full((total,), float(lambda_low_mean)),
            )
            values = (values + noise * float(lambda_std)).to(device=device, dtype=dtype)
        else:
            choose_high = torch.rand(total, device=device, dtype=dtype) < float(lambda_high_prob)
            noise = torch.randn(total, device=device, dtype=dtype)
            values = torch.where(
                choose_high,
                torch.full((total,), float(lambda_high_mean), device=device, dtype=dtype),
                torch.full((total,), float(lambda_low_mean), device=device, dtype=dtype),
            )
            values = values + noise * float(lambda_std)
    elif mode == "fixed":
        values = torch.full((total,), float(lambda_fixed), device=device, dtype=dtype)
    else:
        raise ValueError(f"unknown slow lambda init mode: {mode}")
    values = values.clamp(1e-4, 0.9999)
    if shuffle and total > 1:
        if int(seed) >= 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed) + 1000003)
            order = torch.randperm(total, generator=gen).to(device)
        else:
            order = torch.randperm(total, device=device)
        values = values[order]

    offset = 0
    with torch.no_grad():
        for rec in recs:
            param = rec.nu if hasattr(rec, "nu") else rec.theta
            width = int(param.numel())
            chunk = values[offset : offset + width].to(param.device, dtype=param.dtype)
            if hasattr(rec, "nu"):
                rec.nu.copy_(torch.log(-torch.log(chunk)))
            elif hasattr(rec, "theta"):
                rec.theta.copy_(_lambda_to_theta(chunk))
            offset += width
    return model


def build_model_variant(
    variant,
    input_dim,
    output_dim,
    rank,
    d_model,
    rec_dim,
    layers,
    dropout,
    plru_tau,
    plru_c,
    pan_lambda_min,
    pan_lambda_max,
    rank_matched_lambda_high,
    rank_matched_lambda_low,
    pan_rnn_form="leaky",
    pan_rnn_gamma_init=0.3,
    pan_gru_alpha_form="upper_cap",
    pan_gru_keep_bias_init=0.0,
    pan_gru_reset_bias_init=0.0,
    pan_lstm_alpha_form="upper_cap",
    pan_lstm_forget_bias_init=1.0,
    pan_lstm_input_bias_init=0.0,
    pan_lstm_ablation_state="c_only",
):
    if variant == "RNN":
        return RNNBaseline(input_dim, output_dim, hidden=rec_dim)
    if variant == "Leaky RNN":
        return LeakyRNNBaseline(
            input_dim,
            output_dim,
            hidden=rec_dim,
            lambda_min=pan_lambda_min,
            lambda_max=pan_lambda_max,
            pan_theta=False,
        )
    if variant in ("RNN-PAN", "PAN-RNN"):
        return PANRNNBaseline(
            input_dim,
            output_dim,
            hidden=rec_dim,
            lambda_min=pan_lambda_min,
            lambda_max=pan_lambda_max,
            decoupled_write=str(pan_rnn_form) in ("decoupled", "decoupled_write", "gamma"),
            gamma_init=pan_rnn_gamma_init,
        )
    if variant == "GRU":
        return GRUBaseline(input_dim, output_dim, hidden=rec_dim)
    if variant == "GRU keep1":
        return GRUBaseline(input_dim, output_dim, hidden=rec_dim, keep_bias_init=1.0)
    if variant == "GRU keep2":
        return GRUBaseline(input_dim, output_dim, hidden=rec_dim, keep_bias_init=2.0)
    if variant in ("GRU-PAN", "PAN-GRU"):
        return PANGRUBaseline(
            input_dim,
            output_dim,
            hidden=rec_dim,
            lambda_min=pan_lambda_min,
            lambda_max=pan_lambda_max,
            alpha_form=pan_gru_alpha_form,
            keep_bias_init=pan_gru_keep_bias_init,
            reset_bias_init=pan_gru_reset_bias_init,
        )
    if variant == "LSTM":
        return LSTMBaseline(input_dim, output_dim, hidden=rec_dim)
    if variant == "LSTM forget1":
        return LSTMBaseline(input_dim, output_dim, hidden=rec_dim, forget_bias_init=1.0)
    if variant == "LSTM forget2":
        return LSTMBaseline(input_dim, output_dim, hidden=rec_dim, forget_bias_init=2.0)
    if variant == "LSTM-PAN":
        return PANLSTMBaseline(
            input_dim,
            output_dim,
            hidden=rec_dim,
            lambda_min=pan_lambda_min,
            lambda_max=pan_lambda_max,
            alpha_form=pan_lstm_alpha_form,
            forget_bias_init=pan_lstm_forget_bias_init,
            input_bias_init=pan_lstm_input_bias_init,
            ablation_state=pan_lstm_ablation_state,
        )
    if variant == "SSM":
        return DiagSSMBaseline(input_dim, output_dim, hidden=rec_dim)
    if variant == "LRU unit":
        return ComplexLRUUnitBaseline(
            input_dim,
            output_dim,
            complex_dim=int(rec_dim) * int(layers),
        )
    if variant in LRU_FULL_WRAPPER_CONFIGS:
        wrapper_kwargs = LRU_FULL_WRAPPER_CONFIGS[variant]
        return build_block_model(
            variant="LRU-Block",
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            rec_dim=rec_dim,
            num_layers=layers,
            dropout=dropout,
            plru_tau=plru_tau,
            plru_c=plru_c,
            pan_lambda_min=pan_lambda_min,
            pan_lambda_max=pan_lambda_max,
            **wrapper_kwargs,
        )
    if variant == "real-diag unit":
        return RealDiagLRUUnitBaseline(
            input_dim,
            output_dim,
            hidden=int(rec_dim) * int(layers),
            r_min=0.0,
            r_max=1.0,
            gamma_free=True,
        )
    if variant == "rank-matched LRU unit":
        model = RealDiagLRUUnitBaseline(
            input_dim,
            output_dim,
            hidden=int(rec_dim) * int(layers),
            gamma_free=True,
        )
        return set_rank_matched_lambdas(
            model,
            rank=rank,
            lambda_high=rank_matched_lambda_high,
            lambda_low=rank_matched_lambda_low,
            freeze=True,
        )
    if variant == "P-LRU unit":
        return PLRUUnitBaseline(
            input_dim,
            output_dim,
            hidden=int(rec_dim) * int(layers),
            tau=plru_tau,
            c=plru_c,
        )
    if variant == "PAN-unit":
        return PANUnitBaseline(
            input_dim,
            output_dim,
            hidden=int(rec_dim) * int(layers),
            lambda_min=pan_lambda_min,
            lambda_max=pan_lambda_max,
            gamma_free=True,
        )
    if variant == "real-diag full":
        return build_block_model(
            variant="real-diag LRU-Block",
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            rec_dim=rec_dim,
            num_layers=layers,
            dropout=dropout,
            plru_tau=plru_tau,
            plru_c=plru_c,
            pan_lambda_min=pan_lambda_min,
            pan_lambda_max=pan_lambda_max,
        )
    if variant in RM_REAL_FULL_WRAPPER_CONFIGS:
        wrapper_kwargs = RM_REAL_FULL_WRAPPER_CONFIGS[variant]
        model = build_block_model(
            variant="real-diag LRU-Block",
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            rec_dim=rec_dim,
            num_layers=layers,
            dropout=dropout,
            plru_tau=plru_tau,
            plru_c=plru_c,
            pan_lambda_min=pan_lambda_min,
            pan_lambda_max=pan_lambda_max,
            **wrapper_kwargs,
        )
        return set_rank_matched_lambdas(
            model,
            rank=rank,
            lambda_high=rank_matched_lambda_high,
            lambda_low=rank_matched_lambda_low,
            freeze=True,
        )
    if variant in PAN_WRAPPER_CONFIGS:
        wrapper_kwargs = PAN_WRAPPER_CONFIGS[variant]
        block_variant = {
            "PAN-full": "PAN-Block",
            "PAN-NW-full": "PAN-NW-Block",
            "PAN-RNW-full": "PAN-RNW-Block",
        }[variant]
        return build_block_model(
            variant=block_variant,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            rec_dim=rec_dim,
            num_layers=layers,
            dropout=dropout,
            plru_tau=plru_tau,
            plru_c=plru_c,
            pan_lambda_min=pan_lambda_min,
            pan_lambda_max=pan_lambda_max,
            **wrapper_kwargs,
        )
    if variant == "rank-matched LRU-Block":
        model = build_block_model(
            variant="LRU-Block",
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            rec_dim=rec_dim,
            num_layers=layers,
            dropout=dropout,
            plru_tau=plru_tau,
            plru_c=plru_c,
            pan_lambda_min=pan_lambda_min,
            pan_lambda_max=pan_lambda_max,
        )
        return set_rank_matched_lambdas(
            model,
            rank=rank,
            lambda_high=rank_matched_lambda_high,
            lambda_low=rank_matched_lambda_low,
            freeze=True,
        )
    return build_block_model(
        variant=variant,
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=d_model,
        rec_dim=rec_dim,
        num_layers=layers,
        dropout=dropout,
        plru_tau=plru_tau,
        plru_c=plru_c,
        pan_lambda_min=pan_lambda_min,
        pan_lambda_max=pan_lambda_max,
    )


def rmse(pred, target):
    return float(torch.sqrt(F.mse_loss(pred, target)).item())


def vec_rmse(pred, target):
    return float(torch.sqrt(((pred - target) ** 2).sum(dim=-1).mean()).item())


def mean_sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(vals.size))


def pca_stats(states):
    x = states - states.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    total = float(var.sum()) + 1e-12
    ratio = var / total
    pr = float(total ** 2 / (np.square(var).sum() + 1e-12))
    csum = np.cumsum(ratio)
    return {
        "pca_pr": pr,
        "pca_dim90": int(np.searchsorted(csum, 0.90) + 1),
        "pca_dim95": int(np.searchsorted(csum, 0.95) + 1),
        "pca_var1": float(ratio[0]) if ratio.size else 0.0,
        "pca_var2": float(ratio[:2].sum()) if ratio.size >= 2 else float(ratio.sum()),
    }


def participation_ratio(weights):
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    total = float(w.sum())
    if total <= 1e-12:
        return 0.0
    return float(total * total / (np.square(w).sum() + 1e-12))


def n_for_fraction(weights, frac=0.90):
    w = np.sort(np.clip(np.asarray(weights, dtype=np.float64), 0.0, None))[::-1]
    total = float(w.sum())
    if total <= 1e-12:
        return 0
    return int(np.searchsorted(np.cumsum(w), frac * total) + 1)


def pairwise_distance_corr(states_np, target_np, max_points=160):
    n = min(max_points, states_np.shape[0])
    x = states_np[:n]
    y = target_np[:n]
    dx = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1))
    dy = np.sqrt(((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1))
    tri = np.triu_indices(n, k=1)
    a = dx[tri]
    b = dy[tri]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def make_basis(ambient_dim, rank, device, seed=7101):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    a = torch.randn(ambient_dim, ambient_dim, generator=gen, device=device)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q[:, :rank].contiguous()


def embed(memory, basis):
    return memory @ basis.t()


def sample_delta(memory, scale, bound):
    lo = torch.maximum(
        torch.full_like(memory, -scale),
        torch.full_like(memory, -bound) - memory,
    )
    hi = torch.minimum(
        torch.full_like(memory, scale),
        torch.full_like(memory, bound) - memory,
    )
    return lo + torch.rand_like(memory) * (hi - lo).clamp_min(1e-6)


def make_batch(
    batch,
    rank,
    ambient_dim,
    basis,
    n_updates,
    hold_steps,
    device,
    update_scale,
    memory_bound,
):
    input_dim = ambient_dim + 1
    memory = torch.zeros(batch, rank, device=device)
    xs, ys, update_indices = [], [], []
    for _ in range(n_updates):
        delta = sample_delta(memory, update_scale, memory_bound)
        memory = memory + delta
        x = torch.zeros(batch, input_dim, device=device)
        x[:, :ambient_dim] = embed(delta, basis)
        x[:, ambient_dim] = 1.0
        y = embed(memory, basis)
        update_indices.append(len(xs))
        xs.append(x)
        ys.append(y)
        for _ in range(hold_steps):
            xs.append(torch.zeros(batch, input_dim, device=device))
            ys.append(y)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), update_indices, memory


def make_final_hold_batch(
    batch,
    rank,
    ambient_dim,
    basis,
    n_updates,
    inter_hold,
    final_hold,
    device,
    update_scale,
    memory_bound,
):
    input_dim = ambient_dim + 1
    memory = torch.zeros(batch, rank, device=device)
    xs, ys, update_indices = [], [], []
    for update_idx in range(n_updates):
        delta = sample_delta(memory, update_scale, memory_bound)
        memory = memory + delta
        x = torch.zeros(batch, input_dim, device=device)
        x[:, :ambient_dim] = embed(delta, basis)
        x[:, ambient_dim] = 1.0
        y = embed(memory, basis)
        update_indices.append(len(xs))
        xs.append(x)
        ys.append(y)
        hold = final_hold if update_idx == n_updates - 1 else inter_hold
        for _ in range(hold):
            xs.append(torch.zeros(batch, input_dim, device=device))
            ys.append(y)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), update_indices, memory


@torch.no_grad()
def roll_blank(model, state, input_dim, steps):
    blank = torch.zeros(state.shape[0], input_dim, device=state.device)
    cur = state
    for _ in range(int(steps)):
        cur = model.step(blank, cur)
    return cur


@torch.no_grad()
def run_to_last_write_state(model, xs, update_indices):
    state = model.init_state(xs.shape[1], xs.device)
    for t in range(update_indices[-1] + 1):
        state = model.step(xs[t], state)
    return state


@torch.no_grad()
def state_after_write(model, memory, basis, ambient_dim, hold_steps):
    batch = memory.shape[0]
    input_dim = ambient_dim + 1
    state = model.init_state(batch, memory.device)
    x = torch.zeros(batch, input_dim, device=memory.device)
    x[:, :ambient_dim] = embed(memory, basis)
    x[:, ambient_dim] = 1.0
    state = model.step(x, state)
    state = roll_blank(model, state, input_dim, hold_steps)
    return state, embed(memory, basis)


@torch.no_grad()
def state_after_delta_path(model, deltas, basis, ambient_dim, inter_hold):
    batch = deltas[0].shape[0]
    input_dim = ambient_dim + 1
    state = model.init_state(batch, deltas[0].device)
    for idx, delta in enumerate(deltas):
        x = torch.zeros(batch, input_dim, device=delta.device)
        x[:, :ambient_dim] = embed(delta, basis)
        x[:, ambient_dim] = 1.0
        state = model.step(x, state)
        if idx != len(deltas) - 1 and inter_hold > 0:
            state = roll_blank(model, state, input_dim, inter_hold)
    return state


@torch.no_grad()
def local_jacobian(model, memory, basis, ambient_dim, hold_steps, eps=1e-3):
    states = []
    for j in range(memory.shape[1]):
        dm = torch.zeros_like(memory)
        dm[:, j] = eps
        sp, _ = state_after_write(model, memory + dm, basis, ambient_dim, hold_steps)
        sm, _ = state_after_write(model, memory - dm, basis, ambient_dim, hold_steps)
        states.append((sp - sm) / (2.0 * eps))
    return torch.stack(states, dim=1)


def tangent_basis_from_jacobian(jac):
    bases = []
    for i in range(jac.shape[0]):
        q, _ = torch.linalg.qr(jac[i].t(), mode="reduced")
        bases.append(q)
    return torch.stack(bases, dim=0)


@torch.no_grad()
def compute_pan_scores(model, xs, update_indices, target, h_probe):
    state = run_to_last_write_state(model, xs, update_indices)
    clean_final = roll_blank(model, state, model.input_dim, h_probe)
    clean_pred = model.decode(clean_final)
    clean_energy = ((clean_pred - target) ** 2).sum(dim=-1).mean()

    score_payload = []
    batch = state.shape[0]
    total_state = state.shape[1]
    for rec, rec_slice in model.pan_recs_with_slices():
        if hasattr(rec, "ablate_pan_coordinates"):
            ablated = rec.ablate_pan_coordinates(state, rec_slice)
            hidden = ablated.shape[0]
        else:
            hidden = rec_slice.stop - rec_slice.start
            ablated = state.unsqueeze(0).expand(hidden, batch, total_state).clone()
            idx = torch.arange(hidden, device=state.device)
            ablated[idx, :, rec_slice.start + idx] = 0.0
        final = roll_blank(
            model,
            ablated.reshape(hidden * batch, total_state),
            model.input_dim,
            h_probe,
        )
        pred = model.decode(final).reshape(hidden, batch, model.output_dim)
        energy = ((pred - target.unsqueeze(0)) ** 2).sum(dim=-1).mean(dim=1)
        scores = energy - clean_energy
        score_payload.append((rec, scores.detach()))
    return score_payload, float(torch.sqrt(clean_energy / model.output_dim).item())


@torch.no_grad()
def apply_pan_update(score_payload, eta_lambda, score_eps=0.0):
    for rec, scores in score_payload:
        rec.update_theta(scores - float(score_eps), eta_lambda)


@torch.no_grad()
def evaluate_model(model, rank, ambient_dim, basis, args, device):
    metrics = {}
    model.eval()
    for horizon in args.hold_horizons:
        x, y, update_idx, _ = make_final_hold_batch(
            args.eval_batch,
            rank,
            ambient_dim,
            basis,
            args.eval_updates,
            args.eval_inter_hold,
            int(horizon),
            device,
            args.update_scale,
            args.memory_bound,
        )
        out = model(x)
        metrics[f"hold_rmse_H{horizon}"] = rmse(out[-1], y[-1])
        metrics[f"hold_vec_rmse_H{horizon}"] = vec_rmse(out[-1], y[-1])
        metrics[f"update_step_rmse_H{horizon}"] = rmse(out[update_idx], y[update_idx])

    for count in args.update_counts:
        x, y, update_idx, _ = make_batch(
            args.eval_batch,
            rank,
            ambient_dim,
            basis,
            int(count),
            args.eval_inter_hold,
            device,
            args.update_scale,
            args.memory_bound,
        )
        out = model(x)
        metrics[f"updates_rmse_K{count}"] = rmse(out[-1], y[-1])
        metrics[f"updates_vec_rmse_K{count}"] = vec_rmse(out[-1], y[-1])
        metrics[f"updates_step_rmse_K{count}"] = rmse(out[update_idx], y[update_idx])

    geom_batch = min(args.analysis_batch, args.eval_batch)
    memory = (torch.rand(geom_batch, rank, device=device) * 2.0 - 1.0) * args.geometry_memory_scale
    base, target = state_after_write(model, memory, basis, ambient_dim, args.train_hold_max)
    pred = model.decode(base)
    state_np = base.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    stats = pca_stats(state_np)
    metrics.update({f"latent_{k}": v for k, v in stats.items()})
    metrics["manifold_rmse"] = rmse(pred, target)
    metrics["manifold_vec_rmse"] = vec_rmse(pred, target)
    metrics["latent_pr_over_rank"] = stats["pca_pr"] / rank
    metrics["latent_target_dist_corr"] = pairwise_distance_corr(state_np, target_np)

    jac_batch = min(args.jacobian_batch, geom_batch)
    jac_memory = memory[:jac_batch]
    jac = local_jacobian(model, jac_memory, basis, ambient_dim, args.train_hold_max)
    q = tangent_basis_from_jacobian(jac)
    base_j = base[:jac_batch]
    target_j = target[:jac_batch]
    noise = torch.randn_like(base_j)
    coeff = torch.einsum("bsd,bs->bd", q, noise)
    tangent = torch.einsum("bsd,bd->bs", q, coeff)
    normal = noise - tangent
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    state_rms = base_j.std(dim=0).pow(2).mean().sqrt()
    radius = torch.clamp(args.normal_radius * state_rms, min=1e-3)
    pert_normal0 = base_j + radius * normal
    initial_normal_dev = (pert_normal0 - base_j).norm(dim=-1).mean().item() + 1e-8

    tangent_delta = sample_delta(jac_memory, args.tangent_memory_scale, args.memory_bound)
    target_tangent = embed(jac_memory + tangent_delta, basis)
    tangent_state_delta = torch.einsum("bds,bd->bs", jac, tangent_delta)
    pert_tangent0 = base_j + tangent_state_delta

    for step in args.recovery_steps:
        clean = roll_blank(model, base_j, model.input_dim, step)
        pert = roll_blank(model, pert_normal0, model.input_dim, step)
        normal_pred = model.decode(pert)
        metrics[f"normal_state_dev_ratio_R{step}"] = (
            (pert - clean).norm(dim=-1).mean().item() / initial_normal_dev
        )
        metrics[f"normal_rmse_R{step}"] = rmse(normal_pred, target_j)
        metrics[f"normal_vec_rmse_R{step}"] = vec_rmse(normal_pred, target_j)

    for step in args.tangent_steps:
        pert = roll_blank(model, pert_tangent0, model.input_dim, step)
        tangent_pred = model.decode(pert)
        metrics[f"tangent_new_rmse_R{step}"] = rmse(tangent_pred, target_tangent)
        metrics[f"tangent_old_rmse_R{step}"] = rmse(tangent_pred, target_j)
        metrics[f"tangent_new_vec_rmse_R{step}"] = vec_rmse(tangent_pred, target_tangent)

    basin_step = int(args.basin_recovery_step)
    clean_basin = roll_blank(model, base_j, model.input_dim, basin_step)
    for radius_scale in args.basin_radii:
        radius_i = torch.clamp(float(radius_scale) * state_rms, min=1e-3)
        pert0 = base_j + radius_i * normal
        pert = roll_blank(model, pert0, model.input_dim, basin_step)
        pred = model.decode(pert)
        key = float_slug(radius_scale)
        metrics[f"basin_normal_rmse_rad{key}_R{basin_step}"] = rmse(pred, target_j)
        metrics[f"basin_state_dev_ratio_rad{key}_R{basin_step}"] = (
            (pert - clean_basin).norm(dim=-1).mean().item()
            / ((pert0 - base_j).norm(dim=-1).mean().item() + 1e-8)
        )

    same_batch = min(args.same_memory_batch, geom_batch)
    same_memory = (torch.rand(same_batch, rank, device=device) * 2.0 - 1.0) * args.geometry_memory_scale
    state_a = state_after_delta_path(model, [same_memory], basis, ambient_dim, args.eval_inter_hold)
    half = 0.5 * same_memory
    state_b = state_after_delta_path(model, [half, half], basis, ambient_dim, args.eval_inter_hold)
    state_a = roll_blank(model, state_a, model.input_dim, args.train_hold_max)
    state_b = roll_blank(model, state_b, model.input_dim, args.train_hold_max)
    target_same = embed(same_memory, basis)
    for step in args.same_memory_steps:
        roll_a = roll_blank(model, state_a, model.input_dim, step)
        roll_b = roll_blank(model, state_b, model.input_dim, step)
        pred_a = model.decode(roll_a)
        pred_b = model.decode(roll_b)
        metrics[f"same_memory_output_gap_R{step}"] = rmse(pred_a, pred_b)
        metrics[f"same_memory_target_rmse_R{step}"] = rmse(0.5 * (pred_a + pred_b), target_same)
        metrics[f"same_memory_state_gap_R{step}"] = float((roll_a - roll_b).norm(dim=-1).mean().item())

    lam = model.lam_mag().detach() if hasattr(model, "lam_mag") else torch.empty(0, device=device)
    if lam.numel() > 0:
        metrics.update({
            "lambda_sum": float(lam.sum().item()),
            "lambda_max": float(lam.max().item()),
            "lambda_gt_0p9": int((lam > 0.9).sum().item()),
            "lambda_gt_0p95": int((lam > 0.95).sum().item()),
            "lambda_gt_0p99": int((lam > 0.99).sum().item()),
        })

    if hasattr(model, "pan_recs_with_slices") and model.pan_recs_with_slices():
        x, _, update_idx, memory = make_batch(
            args.pan_probe_batch,
            rank,
            ambient_dim,
            basis,
            args.eval_updates,
            args.eval_inter_hold,
            device,
            args.update_scale,
            args.memory_bound,
        )
        target = embed(memory, basis)
        score_payload, pan_probe_rmse = compute_pan_scores(model, x, update_idx, target, args.pan_h_probe)
        scores = torch.cat([score.detach().flatten() for _, score in score_payload]).cpu().numpy()
        scores_pos = np.clip(scores, 0.0, None)
        pan_lams = torch.cat([rec.lam_mag().detach().flatten() for rec, _ in score_payload]).cpu().numpy()
        if np.std(scores) > 1e-12 and np.std(pan_lams) > 1e-12:
            score_lambda_corr = float(np.corrcoef(scores, pan_lams)[0, 1])
        else:
            score_lambda_corr = 0.0
        metrics.update({
            "pan_eval_probe_rmse": pan_probe_rmse,
            "pan_score_mean": float(scores.mean()) if scores.size else 0.0,
            "pan_score_max": float(scores.max()) if scores.size else 0.0,
            "pan_score_pos_sum": float(scores_pos.sum()),
            "pan_score_pr": participation_ratio(scores_pos),
            "pan_score_n90": n_for_fraction(scores_pos, 0.90),
            "pan_score_lambda_corr": score_lambda_corr,
        })
    return metrics


def train_eval_one(args, rank, seed, model_name, device):
    variant = normalize_model_variant(model_name)
    set_seed(seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.ckpt_dir)

    ambient_dim = int(args.ambient_dim)
    input_dim = ambient_dim + 1
    output_dim = ambient_dim
    basis = make_basis(ambient_dim, rank, device, seed=int(args.basis_seed))

    model_slug = slugify(variant)
    json_path = os.path.join(args.out_dir, f"pulsehold_block_p{ambient_dim}_d{rank}_{model_slug}_seed{seed}.json")
    ckpt_path = os.path.join(args.ckpt_dir, f"exp71_block_p{ambient_dim}_d{rank}_{model_slug}_seed{seed}.pt")
    if os.path.exists(json_path) and os.path.exists(ckpt_path) and not args.force:
        return {"status": "skipped", "path": json_path}

    model = build_model_variant(
        variant=variant,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
        d_model=args.d_model,
        rec_dim=args.rec_dim,
        layers=args.layers,
        dropout=args.dropout,
        plru_tau=args.plru_tau,
        plru_c=args.plru_c,
        pan_lambda_min=args.pan_lambda_min,
        pan_lambda_max=args.pan_lambda_max,
        rank_matched_lambda_high=args.rank_matched_lambda_high,
        rank_matched_lambda_low=args.rank_matched_lambda_low,
    ).to(device)
    apply_slow_lambda_init(
        model,
        mode=args.slow_lambda_init_mode,
        lambda_min=args.slow_lambda_min,
        lambda_max=args.slow_lambda_max,
        lambda_fixed=args.slow_lambda_fixed,
        lambda_mean=args.slow_lambda_mean,
        lambda_std=args.slow_lambda_std,
        lambda_low_mean=args.slow_lambda_low_mean,
        lambda_high_mean=args.slow_lambda_high_mean,
        lambda_high_prob=args.slow_lambda_high_prob,
        shuffle=args.slow_lambda_shuffle,
        seed=args.slow_lambda_init_seed,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)

    losses, task_losses, reg_losses = [], [], []
    pan_probe_rmse = float("nan")
    pan_last_score_mean = float("nan")
    pan_last_score_max = float("nan")
    loss_trace = None
    if int(args.loss_log_every) > 0 or int(args.eval_log_every) > 0:
        loss_trace = {
            "steps": [],
            "train_loss": [],
            "train_task_loss": [],
            "train_reg_loss": [],
            "eval_mse": [],
            "eval_rmse": [],
            "eval_horizon": int(args.eval_log_horizon),
        }
    lambda_trace = None
    if bool(args.log_lambda_trajectory) and hasattr(model, "lam_mag"):
        lambda_trace = {
            "steps": [],
            "lambdas": [],
            "lambda_gt_0p9": [],
            "lambda_gt_0p95": [],
            "lambda_gt_0p99": [],
            "pan_score_mean": [],
            "pan_score_max": [],
            "pan_probe_rmse": [],
        }
        lam0 = model.lam_mag().detach().cpu().numpy().astype(np.float32)
        lambda_trace["steps"].append(0)
        lambda_trace["lambdas"].append(lam0)
        lambda_trace["lambda_gt_0p9"].append(int((lam0 > 0.9).sum()))
        lambda_trace["lambda_gt_0p95"].append(int((lam0 > 0.95).sum()))
        lambda_trace["lambda_gt_0p99"].append(int((lam0 > 0.99).sum()))
        lambda_trace["pan_score_mean"].append(float("nan"))
        lambda_trace["pan_score_max"].append(float("nan"))
        lambda_trace["pan_probe_rmse"].append(float("nan"))
    warmup_steps = int(round(args.steps * args.pan_warmup_frac)) if is_pan_variant(variant) else 0
    t0 = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        hold = random.randint(args.train_hold_min, args.train_hold_max)
        n_updates = random.randint(args.train_updates_min, args.train_updates_max)
        x, y, update_idx, memory = make_batch(
            args.batch,
            rank,
            ambient_dim,
            basis,
            n_updates,
            hold,
            device,
            args.update_scale,
            args.memory_bound,
        )
        out = model(x)
        task_loss = F.mse_loss(out, y)
        reg = model.regularization_loss() if hasattr(model, "regularization_loss") else None
        reg_value = 0.0
        loss = task_loss
        if reg is not None:
            reg_weight = 1.0
            if variant in ("P-LRU unit", "P-LRU-Block"):
                plru_warmup = max(0, int(round(args.steps * args.plru_warmup_frac)))
                reg_weight = 1.0 if plru_warmup == 0 else min(1.0, step / plru_warmup)
            loss = loss + reg_weight * reg
            reg_value = float(reg.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if is_pan_variant(variant) and step % args.pan_probe_every == 0:
            model.eval()
            probe_x, _, probe_update_idx, probe_memory = make_batch(
                args.pan_probe_batch,
                rank,
                ambient_dim,
                basis,
                args.eval_updates,
                args.eval_inter_hold,
                device,
                args.update_scale,
                args.memory_bound,
            )
            target = embed(probe_memory, basis)
            score_payload, pan_probe_rmse = compute_pan_scores(
                model,
                probe_x,
                probe_update_idx,
                target,
                args.pan_h_probe,
            )
            all_scores = torch.cat([scores.detach().flatten() for _, scores in score_payload])
            pan_last_score_mean = float(all_scores.mean().item())
            pan_last_score_max = float(all_scores.max().item())
            if step > warmup_steps:
                apply_pan_update(score_payload, args.pan_eta_lambda, args.pan_score_eps)
            model.train()

        losses.append(float(loss.item()))
        task_losses.append(float(task_loss.item()))
        reg_losses.append(reg_value)

        should_log_loss = loss_trace is not None and (
            step == 1
            or step == args.steps
            or (int(args.loss_log_every) > 0 and step % int(args.loss_log_every) == 0)
            or (int(args.eval_log_every) > 0 and step % int(args.eval_log_every) == 0)
        )
        if should_log_loss:
            eval_mse = float("nan")
            eval_rmse = float("nan")
            if int(args.eval_log_every) > 0 and (step == 1 or step == args.steps or step % int(args.eval_log_every) == 0):
                model.eval()
                with torch.no_grad():
                    eval_x, eval_y, _, _ = make_final_hold_batch(
                        int(args.eval_log_batch),
                        rank,
                        ambient_dim,
                        basis,
                        int(args.eval_updates),
                        int(args.eval_inter_hold),
                        int(args.eval_log_horizon),
                        device,
                        float(args.update_scale),
                        float(args.memory_bound),
                    )
                    eval_out = model(eval_x)
                    eval_mse = float(F.mse_loss(eval_out[-1], eval_y[-1]).item())
                    eval_rmse = float(torch.sqrt(F.mse_loss(eval_out[-1], eval_y[-1])).item())
                model.train()
            loss_trace["steps"].append(step)
            loss_trace["train_loss"].append(float(loss.item()))
            loss_trace["train_task_loss"].append(float(task_loss.item()))
            loss_trace["train_reg_loss"].append(float(reg_value))
            loss_trace["eval_mse"].append(eval_mse)
            loss_trace["eval_rmse"].append(eval_rmse)

        if step == 1 or step % max(1, args.steps // 4) == 0:
            lam = model.lam_mag().detach() if hasattr(model, "lam_mag") else torch.empty(0, device=device)
            lam_msg = f" sum_lambda={lam.sum().item():.1f} n>.99={(lam > .99).sum().item()}" if lam.numel() else ""
            print(
                f"[Exp71 {variant:18s} d={rank:<2d} seed={seed}] "
                f"{step:5d}/{args.steps} task={task_loss.item():.5f} reg={reg_value:.4f}{lam_msg}",
                flush=True,
            )
        if lambda_trace is not None and (step == 1 or step == args.steps or step % args.lambda_log_every == 0):
            lam = model.lam_mag().detach().cpu().numpy().astype(np.float32)
            lambda_trace["steps"].append(step)
            lambda_trace["lambdas"].append(lam)
            lambda_trace["lambda_gt_0p9"].append(int((lam > 0.9).sum()))
            lambda_trace["lambda_gt_0p95"].append(int((lam > 0.95).sum()))
            lambda_trace["lambda_gt_0p99"].append(int((lam > 0.99).sum()))
            lambda_trace["pan_score_mean"].append(float(pan_last_score_mean))
            lambda_trace["pan_score_max"].append(float(pan_last_score_max))
            lambda_trace["pan_probe_rmse"].append(float(pan_probe_rmse))

    metrics = evaluate_model(model, rank, ambient_dim, basis, args, device)
    result = {
        "task": "pulsehold_block",
        "ambient_dim": ambient_dim,
        "rank": int(rank),
        "model": variant,
        "seed": int(seed),
        "device": str(device),
        "params": count_params(model),
        "layers": int(args.layers),
        "d_model": int(args.d_model),
        "rec_dim": int(args.rec_dim),
        "train_steps": int(args.steps),
        "train_loss_final": float(np.mean(losses[-min(50, len(losses)):])),
        "train_task_loss_final": float(np.mean(task_losses[-min(50, len(task_losses)):])),
        "train_reg_loss_final": float(np.mean(reg_losses[-min(50, len(reg_losses)):])),
        "seconds": time.time() - t0,
        "plru_tau": args.plru_tau if variant in ("P-LRU unit", "P-LRU-Block") else "",
        "plru_c": args.plru_c if variant in ("P-LRU unit", "P-LRU-Block") else "",
        "plru_warmup_frac": args.plru_warmup_frac if variant in ("P-LRU unit", "P-LRU-Block") else "",
        "rank_matched_lambda_high": args.rank_matched_lambda_high
        if variant in ("rank-matched LRU-Block", "rank-matched LRU unit") or variant in RM_REAL_FULL_WRAPPER_CONFIGS
        else "",
        "rank_matched_lambda_low": args.rank_matched_lambda_low
        if variant in ("rank-matched LRU-Block", "rank-matched LRU unit") or variant in RM_REAL_FULL_WRAPPER_CONFIGS
        else "",
        "wrapper_encoder_bias": WRAPPER_CONFIGS.get(variant, {}).get("encoder_bias", True),
        "wrapper_use_norm_in": WRAPPER_CONFIGS.get(variant, {}).get("use_norm_in", True),
        "wrapper_norm_in_affine": WRAPPER_CONFIGS.get(variant, {}).get("norm_in_affine", True),
        "wrapper_use_norm_out": WRAPPER_CONFIGS.get(variant, {}).get("use_norm_out", True),
        "wrapper_norm_out_affine": WRAPPER_CONFIGS.get(variant, {}).get("norm_out_affine", True),
        "wrapper_update_mode": WRAPPER_CONFIGS.get(variant, {}).get("update_mode", "glu"),
        "wrapper_use_residual": WRAPPER_CONFIGS.get(variant, {}).get("use_residual", True),
        "wrapper_decode_mode": WRAPPER_CONFIGS.get(variant, {}).get("decode_mode", "stream"),
        "pan_lambda_min": args.pan_lambda_min if is_pan_variant(variant) else "",
        "pan_lambda_max": args.pan_lambda_max if is_pan_variant(variant) else "",
        "pan_eta_lambda": args.pan_eta_lambda if is_pan_variant(variant) else "",
        "pan_score_eps": args.pan_score_eps if is_pan_variant(variant) else "",
        "pan_h_probe": args.pan_h_probe if is_pan_variant(variant) else "",
        "pan_probe_every": args.pan_probe_every if is_pan_variant(variant) else "",
        "pan_probe_rmse": pan_probe_rmse if is_pan_variant(variant) else "",
        "pan_last_score_mean": pan_last_score_mean if is_pan_variant(variant) else "",
        "pan_last_score_max": pan_last_score_max if is_pan_variant(variant) else "",
        "slow_lambda_init_mode": args.slow_lambda_init_mode,
        "slow_lambda_min": args.slow_lambda_min if args.slow_lambda_init_mode != "default" else "",
        "slow_lambda_max": args.slow_lambda_max if args.slow_lambda_init_mode != "default" else "",
        "slow_lambda_fixed": args.slow_lambda_fixed if args.slow_lambda_init_mode == "fixed" else "",
        "slow_lambda_mean": args.slow_lambda_mean if args.slow_lambda_init_mode == "gaussian" else "",
        "slow_lambda_std": args.slow_lambda_std if args.slow_lambda_init_mode in ("gaussian", "bimodal") else "",
        "slow_lambda_low_mean": args.slow_lambda_low_mean if args.slow_lambda_init_mode == "bimodal" else "",
        "slow_lambda_high_mean": args.slow_lambda_high_mean if args.slow_lambda_init_mode == "bimodal" else "",
        "slow_lambda_high_prob": args.slow_lambda_high_prob if args.slow_lambda_init_mode == "bimodal" else "",
        "slow_lambda_shuffle": bool(args.slow_lambda_shuffle),
        "slow_lambda_init_seed": int(args.slow_lambda_init_seed),
    }
    result.update(metrics)

    torch.save(model.state_dict(), ckpt_path)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    if lambda_trace is not None and lambda_trace["steps"]:
        trace_dir = args.trace_dir or args.out_dir
        ensure_dir(trace_dir)
        trace_path = os.path.join(trace_dir, f"pulsehold_block_p{ambient_dim}_d{rank}_{model_slug}_seed{seed}_lambda_trace.npz")
        np.savez_compressed(
            trace_path,
            steps=np.asarray(lambda_trace["steps"], dtype=np.int64),
            lambdas=np.asarray(lambda_trace["lambdas"], dtype=np.float32),
            lambda_gt_0p9=np.asarray(lambda_trace["lambda_gt_0p9"], dtype=np.int64),
            lambda_gt_0p95=np.asarray(lambda_trace["lambda_gt_0p95"], dtype=np.int64),
            lambda_gt_0p99=np.asarray(lambda_trace["lambda_gt_0p99"], dtype=np.int64),
            pan_score_mean=np.asarray(lambda_trace["pan_score_mean"], dtype=np.float32),
            pan_score_max=np.asarray(lambda_trace["pan_score_max"], dtype=np.float32),
            pan_probe_rmse=np.asarray(lambda_trace["pan_probe_rmse"], dtype=np.float32),
        )
    if loss_trace is not None and loss_trace["steps"]:
        trace_dir = args.trace_dir or args.out_dir
        ensure_dir(trace_dir)
        trace_path = os.path.join(trace_dir, f"pulsehold_block_p{ambient_dim}_d{rank}_{model_slug}_seed{seed}_loss_trace.npz")
        np.savez_compressed(
            trace_path,
            steps=np.asarray(loss_trace["steps"], dtype=np.int64),
            train_loss=np.asarray(loss_trace["train_loss"], dtype=np.float32),
            train_task_loss=np.asarray(loss_trace["train_task_loss"], dtype=np.float32),
            train_reg_loss=np.asarray(loss_trace["train_reg_loss"], dtype=np.float32),
            eval_mse=np.asarray(loss_trace["eval_mse"], dtype=np.float32),
            eval_rmse=np.asarray(loss_trace["eval_rmse"], dtype=np.float32),
            eval_horizon=np.asarray([loss_trace["eval_horizon"]], dtype=np.int64),
        )
    return {"status": "done", "path": json_path}


def train_eval_job(job):
    gpu = int(job.get("gpu", -1))
    requested_device = job.get("device", "auto")
    if requested_device == "auto" and gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.set_num_threads(1)
    args = argparse.Namespace(**job)
    if requested_device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    return train_eval_one(args, int(job["rank"]), int(job["seed"]), job["model"], device)


def load_rows(out_dir):
    rows = []
    if not os.path.isdir(out_dir):
        return rows
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".json") and name.startswith("pulsehold_block_"):
            with open(os.path.join(out_dir, name)) as f:
                rows.append(json.load(f))
    return rows


def write_csv(rows, out_dir):
    if not rows:
        return None
    keys = sorted({key for row in rows for key in row.keys()})
    path = os.path.join(out_dir, "exp71_pan_block_pulsehold_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(path)
    return path


def _available_horizons(rows, prefix):
    values = []
    for row in rows:
        for key in row:
            if key.startswith(prefix):
                try:
                    values.append(int(key.split(prefix)[-1]))
                except ValueError:
                    pass
    return sorted(set(values))


def plot_rank_summary(rows, fig_dir, ranks):
    if not rows:
        return
    plt = get_pyplot()
    if plt is None:
        return
    ensure_dir(fig_dir)
    hold_horizons = _available_horizons(rows, "hold_rmse_H")
    recovery_steps = _available_horizons(rows, "normal_rmse_R")
    tangent_steps = _available_horizons(rows, "tangent_new_rmse_R")
    hold_key = f"hold_rmse_H{max(hold_horizons)}" if hold_horizons else "hold_rmse_H500"
    recovery_key = f"normal_rmse_R{max(recovery_steps)}" if recovery_steps else "normal_rmse_R500"
    tangent_key = f"tangent_new_rmse_R{max(tangent_steps)}" if tangent_steps else "tangent_new_rmse_R500"

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2))
    specs = [
        (hold_key, f"long hold, {hold_key.split('_')[-1]}", "RMSE"),
        (recovery_key, f"normal recovery, {recovery_key.split('_')[-1]}", "RMSE"),
        (tangent_key, f"tangent persistence, {tangent_key.split('_')[-1]}", "RMSE"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, specs):
        for model_name in MODEL_VARIANTS:
            xs, ys, es = [], [], []
            for rank in ranks:
                vals = [
                    row[metric]
                    for row in rows
                    if row.get("rank") == rank and row.get("model") == model_name and metric in row
                ]
                m, s = mean_sem(vals)
                if m is None:
                    continue
                xs.append(rank)
                ys.append(m)
                es.append(s)
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=es,
                    marker="o",
                    lw=1.8,
                    capsize=3,
                    color=MODEL_COLORS.get(model_name),
                    label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(rank) for rank in ranks])
        ax.set_xlabel("intrinsic memory rank d")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Exp71 full-block pulse-hold: controlled memory diagnostics", fontsize=12, fontweight="bold")
    path = os.path.join(fig_dir, "exp71_pan_block_rank_summary.png")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_geometry_summary(rows, fig_dir, ranks):
    if not rows:
        return
    plt = get_pyplot()
    if plt is None:
        return
    ensure_dir(fig_dir)
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2))
    specs = [
        ("latent_pca_dim90", "latent PCA dim90", "dimension"),
        ("lambda_gt_0p99", "lambda count > .99", "count"),
        ("latent_target_dist_corr", "state-target distance correlation", "correlation"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, specs):
        for model_name in MODEL_VARIANTS:
            xs, ys, es = [], [], []
            for rank in ranks:
                vals = [
                    row[metric]
                    for row in rows
                    if row.get("rank") == rank and row.get("model") == model_name and metric in row
                ]
                m, s = mean_sem(vals)
                if m is None:
                    continue
                xs.append(rank)
                ys.append(m)
                es.append(s)
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=es,
                    marker="o",
                    lw=1.8,
                    capsize=3,
                    color=MODEL_COLORS.get(model_name),
                    label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(rank) for rank in ranks])
        ax.set_xlabel("intrinsic memory rank d")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Exp71 full-block pulse-hold: state geometry", fontsize=12, fontweight="bold")
    path = os.path.join(fig_dir, "exp71_pan_block_geometry_summary.png")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_hold_curves(rows, fig_dir, ranks):
    if not rows:
        return
    plt = get_pyplot()
    if plt is None:
        return
    ensure_dir(fig_dir)
    horizons = _available_horizons(rows, "hold_rmse_H")
    if not horizons:
        return
    focus = [rank for rank in [16, 32] if rank in ranks]
    if not focus:
        focus = ranks[-min(2, len(ranks)) :]
    fig, axes = plt.subplots(1, len(focus), figsize=(5.2 * len(focus), 4.2), squeeze=False)
    for ax, rank in zip(axes[0], focus):
        for model_name in MODEL_VARIANTS:
            xs, ys, es = [], [], []
            for horizon in horizons:
                metric = f"hold_rmse_H{horizon}"
                vals = [
                    row[metric]
                    for row in rows
                    if row.get("rank") == rank and row.get("model") == model_name and metric in row
                ]
                m, s = mean_sem(vals)
                if m is None:
                    continue
                xs.append(horizon)
                ys.append(m)
                es.append(s)
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=es,
                    marker="o",
                    lw=1.8,
                    capsize=3,
                    color=MODEL_COLORS.get(model_name),
                    label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
                )
        ax.axvspan(10, 50, color="0.82", alpha=0.28)
        ax.set_xscale("log")
        ax.set_xlabel("final zero-input hold length")
        ax.set_ylabel("RMSE")
        ax.set_title(f"d={rank}")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Exp71 full-block pulse-hold: hold OOD curves", fontsize=12, fontweight="bold")
    path = os.path.join(fig_dir, "exp71_pan_block_hold_curves.png")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(path)


def aggregate_and_plot(out_dir, fig_dir, ranks):
    rows = load_rows(out_dir)
    write_csv(rows, out_dir)
    if rows:
        plot_rank_summary(rows, fig_dir, ranks)
        plot_geometry_summary(rows, fig_dir, ranks)
        plot_hold_curves(rows, fig_dir, ranks)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ambient-dim", type=int, default=32)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--models", nargs="+", default=list(MODEL_VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--analysis-batch", type=int, default=96)
    parser.add_argument("--jacobian-batch", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--rec-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-hold-min", type=int, default=10)
    parser.add_argument("--train-hold-max", type=int, default=50)
    parser.add_argument("--train-updates-min", type=int, default=1)
    parser.add_argument("--train-updates-max", type=int, default=3)
    parser.add_argument("--hold-horizons", type=int, nargs="+", default=HOLD_HORIZONS)
    parser.add_argument("--update-counts", type=int, nargs="+", default=UPDATE_COUNTS)
    parser.add_argument("--recovery-steps", type=int, nargs="+", default=RECOVERY_STEPS)
    parser.add_argument("--tangent-steps", type=int, nargs="+", default=TANGENT_STEPS)
    parser.add_argument("--basin-radii", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--basin-recovery-step", type=int, default=500)
    parser.add_argument("--same-memory-steps", type=int, nargs="+", default=[0, 100, 500])
    parser.add_argument("--same-memory-batch", type=int, default=64)
    parser.add_argument("--eval-updates", type=int, default=3)
    parser.add_argument("--eval-inter-hold", type=int, default=20)
    parser.add_argument("--update-scale", type=float, default=0.20)
    parser.add_argument("--memory-bound", type=float, default=0.90)
    parser.add_argument("--geometry-memory-scale", type=float, default=0.50)
    parser.add_argument("--normal-radius", type=float, default=0.25)
    parser.add_argument("--tangent-memory-scale", type=float, default=0.10)
    parser.add_argument("--basis-seed", type=int, default=7101)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--plru-tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--plru-c", type=float, default=DEFAULT_C)
    parser.add_argument("--plru-warmup-frac", type=float, default=DEFAULT_WARMUP_FRAC)
    parser.add_argument("--rank-matched-lambda-high", type=float, default=0.999)
    parser.add_argument("--rank-matched-lambda-low", type=float, default=0.0)
    parser.add_argument("--slow-lambda-init-mode", choices=["default", "linspace", "random", "gaussian", "bimodal", "fixed"], default="default")
    parser.add_argument("--slow-lambda-min", type=float, default=0.90)
    parser.add_argument("--slow-lambda-max", type=float, default=0.999)
    parser.add_argument("--slow-lambda-fixed", type=float, default=0.999)
    parser.add_argument("--slow-lambda-mean", type=float, default=0.5)
    parser.add_argument("--slow-lambda-std", type=float, default=0.2)
    parser.add_argument("--slow-lambda-low-mean", type=float, default=0.05)
    parser.add_argument("--slow-lambda-high-mean", type=float, default=0.95)
    parser.add_argument("--slow-lambda-high-prob", type=float, default=0.5)
    parser.add_argument("--slow-lambda-shuffle", action="store_true")
    parser.add_argument("--slow-lambda-init-seed", type=int, default=-1)
    parser.add_argument("--pan-lambda-min", type=float, default=0.90)
    parser.add_argument("--pan-lambda-max", type=float, default=0.999)
    parser.add_argument("--pan-eta-lambda", type=float, default=300.0)
    parser.add_argument("--pan-score-eps", type=float, default=0.0)
    parser.add_argument("--pan-warmup-frac", type=float, default=0.3)
    parser.add_argument("--pan-probe-every", type=int, default=100)
    parser.add_argument("--pan-probe-batch", type=int, default=96)
    parser.add_argument("--pan-h-probe", type=int, default=200)
    parser.add_argument("--log-lambda-trajectory", action="store_true")
    parser.add_argument("--lambda-log-every", type=int, default=100)
    parser.add_argument("--loss-log-every", type=int, default=0)
    parser.add_argument("--eval-log-every", type=int, default=0)
    parser.add_argument("--eval-log-horizon", type=int, default=500)
    parser.add_argument("--eval-log-batch", type=int, default=256)
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--out-dir", default="exp71_pan_block_pulsehold_results")
    parser.add_argument("--ckpt-dir", default="checkpoints_exp71_pan_block")
    parser.add_argument("--fig-dir", default="figures_exp71_pan_block")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.ranks = [1]
        args.seeds = [0]
        args.models = list(MODEL_VARIANTS)
        args.steps = 3
        args.batch = 8
        args.eval_batch = 8
        args.analysis_batch = 8
        args.jacobian_batch = 4
        args.d_model = 16
        args.rec_dim = 16
        args.layers = 1
        args.hold_horizons = [2]
        args.update_counts = [1]
        args.recovery_steps = [0, 2]
        args.tangent_steps = [0, 2]
        args.basin_radii = [0.1, 0.25]
        args.basin_recovery_step = 2
        args.same_memory_steps = [0, 2]
        args.same_memory_batch = 4
        args.eval_inter_hold = 2
        args.pan_probe_every = 1
        args.pan_probe_batch = 8
        args.pan_h_probe = 2
        args.max_workers = 1

    ensure_dir(args.out_dir)
    ensure_dir(args.ckpt_dir)
    ensure_dir(args.fig_dir)
    if args.plot_only:
        aggregate_and_plot(args.out_dir, args.fig_dir, args.ranks)
        return

    base_job = vars(args).copy()
    jobs = []
    idx = 0
    use_gpu_pool = args.device == "auto" and bool(args.gpus) and torch.cuda.is_available() and not args.smoke
    for rank in args.ranks:
        for seed in args.seeds:
            for model_name in args.models:
                job = base_job.copy()
                job.update({
                    "rank": int(rank),
                    "seed": int(seed),
                    "model": model_name,
                    "gpu": int(args.gpus[idx % len(args.gpus)]) if use_gpu_pool else -1,
                })
                jobs.append(job)
                idx += 1

    max_workers = args.max_workers
    if max_workers is None:
        max_workers = min(len(args.gpus), len(jobs)) if use_gpu_pool else 1
    max_workers = max(1, min(int(max_workers), len(jobs))) if jobs else 1

    if max_workers == 1:
        for job in jobs:
            job_gpu = int(job.get("gpu", -1))
            if args.device == "auto" and job_gpu >= 0 and torch.cuda.is_available() and not args.smoke:
                device = torch.device(f"cuda:{job_gpu}")
            elif args.device == "auto":
                device = torch.device("cuda:0" if torch.cuda.is_available() and not args.smoke else "cpu")
            else:
                device = torch.device(args.device)
            result = train_eval_one(argparse.Namespace(**job), int(job["rank"]), int(job["seed"]), job["model"], device)
            print(result, flush=True)
    else:
        print(f"Launching {len(jobs)} Exp71 jobs with {max_workers} workers on GPUs {args.gpus}", flush=True)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as ex:
            futures = [ex.submit(train_eval_job, job) for job in jobs]
            for fut in as_completed(futures):
                print(fut.result(), flush=True)

    aggregate_and_plot(args.out_dir, args.fig_dir, args.ranks)


if __name__ == "__main__":
    main()
