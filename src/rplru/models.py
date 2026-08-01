"""Clean recurrent model registry for the RP-LRU experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn


CORE_MODEL_NAMES = ("rnn", "gru", "lstm", "lru", "rp_lru")
EXTENDED_BASELINE_NAMES = ("orthogonal_rnn", "s4d_legs", "chrono_lstm")
MODEL_NAMES = (*CORE_MODEL_NAMES, *EXTENDED_BASELINE_NAMES)
RETENTION_MODES = (
    "rp",
    "rp_grad",
    "frozen",
    "bptt",
    "all_slow",
    "direct_lambda_grad",
    "ste_recall_grad",
    "schedule_bptt",
    "fixed_binary",
    "fixed_unit_bptt",
)


def saturated_sigmoid(
    theta: torch.Tensor,
    tau_sat: float = 16.64,
) -> torch.Tensor:
    """Precision-independent unit branch used by every RP-LRU variant."""

    sigmoid_value = torch.sigmoid(theta)
    return torch.where(
        theta >= float(tau_sat),
        torch.ones_like(sigmoid_value),
        sigmoid_value,
    )


def _sample_lru_ring_radii(
    width: int,
    *,
    radius_min: float,
    radius_max: float,
) -> torch.Tensor:
    """Sample the Orvieto et al. (2023) complex-annulus initialization.

    Uniform density over the area of an annulus requires the squared radius,
    rather than the radius itself, to be uniform.  The LRU stores the sampled
    radius through the stable parameterization ``r = exp(-exp(nu))``.
    """

    lower = float(radius_min)
    upper = float(radius_max)
    if int(width) <= 0:
        raise ValueError("LRU width must be positive")
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("LRU radii require 0 < radius_min < radius_max < 1")
    radius_squared = torch.empty(int(width), dtype=torch.float32).uniform_(
        lower * lower,
        upper * upper,
    )
    return torch.sqrt(radius_squared)


class SequenceModel(nn.Module):
    """Minimal time-major recurrent interface shared by every comparison."""

    input_dim: int
    output_dim: int
    state_size: int

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_size, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def optimizer_parameter_groups(
        self, *, learning_rate: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        return [
            {
                "params": [
                    parameter
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ],
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
                "group_name": "all_trainable_parameters",
            }
        ]

    def architecture_metadata(self) -> dict[str, Any]:
        return {}

    def forward_states(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected time x batch x {self.input_dim} input, got "
                f"{tuple(inputs.shape)}"
            )
        state = self.initial_state(
            inputs.shape[1], device=inputs.device, dtype=inputs.dtype
        )
        states = []
        for x_t in inputs:
            state = self.step(x_t, state)
            states.append(state)
        return torch.stack(states, dim=0)

    def final_states(
        self, inputs: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.forward_states(inputs)
        if lengths.shape != (inputs.shape[1],):
            raise ValueError("length tensor has the wrong shape")
        sample = torch.arange(inputs.shape[1], device=inputs.device)
        final = states[lengths.to(inputs.device) - 1, sample]
        return final, states

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        final, _ = self.final_states(inputs, lengths)
        return self.decode(final)


class GLUReadout(nn.Module):
    """Single-layer deep-LRU readout kept outside the recurrence.

    The residual path surrounds only the position-wise GLU mixing.  Keeping
    this module in ``decode`` ensures that zero-input autonomous dynamics are
    still determined entirely by the recurrent transition.
    """

    def __init__(self, state_features: int, model_dim: int, output_dim: int):
        super().__init__()
        self.pre = nn.Linear(int(state_features), int(model_dim))
        self.value = nn.Linear(int(model_dim), int(model_dim))
        self.gate = nn.Linear(int(model_dim), int(model_dim))
        self.post = nn.Linear(int(model_dim), int(output_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        pre = self.pre(state)
        mixed = self.value(pre) * torch.sigmoid(self.gate(pre))
        return self.post(pre + mixed)


class RNNModel(SequenceModel):
    def __init__(self, input_dim: int, output_dim: int, width: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.state_size = self.width
        self.cell = nn.RNNCell(self.input_dim, self.width, nonlinearity="tanh")
        self.decoder = nn.Linear(self.width, self.output_dim)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.cell(x_t, state)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)


class GRUModel(SequenceModel):
    def __init__(self, input_dim: int, output_dim: int, width: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.state_size = self.width
        self.cell = nn.GRUCell(self.input_dim, self.width)
        self.decoder = nn.Linear(self.width, self.output_dim)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.cell(x_t, state)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)


class LSTMModel(SequenceModel):
    def __init__(self, input_dim: int, output_dim: int, width: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.state_size = 2 * self.width
        self.cell = nn.LSTMCell(self.input_dim, self.width)
        self.decoder = nn.Linear(self.width, self.output_dim)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        hidden, cell = state.split(self.width, dim=-1)
        next_hidden, next_cell = self.cell(x_t, (hidden, cell))
        return torch.cat([next_hidden, next_cell], dim=-1)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state[..., : self.width])


class LRUModel(SequenceModel):
    """Standard complex diagonal LRU with dense input mixing.

    This is deliberately distinct from the real-diagonal RP-LRU.  A
    bias-free projection makes literal task holds literal zero drive, while
    the complex diagonal transition and tied input normalization follow the
    standard LRU design.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        *,
        lambda_min: float = 0.90,
        lambda_max: float = 0.999,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.state_size = 2 * self.width
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.input_projection = nn.Linear(self.input_dim, self.width, bias=False)
        radius = _sample_lru_ring_radii(
            self.width,
            radius_min=self.lambda_min,
            radius_max=self.lambda_max,
        )
        self.nu = nn.Parameter(torch.log(-torch.log(radius)))
        self.phase = nn.Parameter(torch.rand(self.width) * (2.0 * math.pi))
        scale = 1.0 / math.sqrt(float(self.width))
        self.B_re = nn.Parameter(torch.randn(self.width, self.width) * scale)
        self.B_im = nn.Parameter(torch.randn(self.width, self.width) * scale)
        self.decoder = GLUReadout(
            2 * self.width,
            self.width,
            self.output_dim,
        )

    def retention(self) -> torch.Tensor:
        return torch.exp(-torch.exp(self.nu))

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        hidden_re, hidden_im = state.split(self.width, dim=-1)
        drive = self.input_projection(x_t)
        radius = self.retention()
        transition_re = radius * torch.cos(self.phase)
        transition_im = radius * torch.sin(self.phase)
        gamma = torch.sqrt(torch.clamp(1.0 - radius.square(), min=1e-8))
        write_re = drive @ self.B_re.t()
        write_im = drive @ self.B_im.t()
        next_re = (
            transition_re * hidden_re
            - transition_im * hidden_im
            + gamma * write_re
        )
        next_im = (
            transition_im * hidden_re
            + transition_re * hidden_im
            + gamma * write_im
        )
        return torch.cat([next_re, next_im], dim=-1)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "family": "complex_diagonal_lru",
            "reference": "Orvieto et al. (2023)",
            "eigenvalue_initialization": "uniform_area_complex_annulus",
            "radius_squared_distribution": "uniform",
            "radius_min": self.lambda_min,
            "radius_max": self.lambda_max,
            "phase_distribution": "uniform_0_2pi",
            "stable_parameterization": "radius=exp(-exp(nu))",
        }


@dataclass(frozen=True)
class RPUpdateResult:
    horizon: int
    clean_mse: float
    damage_mean: float
    damage_max: float
    normalized_score_mean: float
    normalized_score_max: float
    normalized_score_min: float
    negative_score_fraction: float
    weak_positive_score_fraction: float
    positive_allocation_fraction: float
    lambda_min: float
    lambda_mean: float
    lambda_max: float
    exact_unit_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class RPGradUpdateResult:
    horizon: int
    probe_loss: float
    gradient_l2: float
    gradient_max_abs: float
    theta_step_l2: float
    theta_step_max_abs: float
    lambda_min: float
    lambda_mean: float
    lambda_max: float
    exact_unit_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


class RPLRUModel(SequenceModel):
    """Real diagonal RP-LRU with an exactly zero-drive recurrent writer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        *,
        retention_mode: str = "rp",
        initial_lambda: float = 0.9,
        initial_lambda_low: float | None = None,
        initial_lambda_high: float | None = None,
        all_slow_lambda: float | None = None,
        writer_width: int | None = None,
        tau_sat: float = 16.64,
        fixed_unit_count: int = 0,
        fixed_fast_lambda: float = 0.0,
        fixed_subset_seed: int = 0,
    ):
        super().__init__()
        if retention_mode not in RETENTION_MODES:
            raise ValueError(f"unknown retention mode {retention_mode!r}")
        if not 0 < initial_lambda < 1:
            raise ValueError("initial_lambda must be strictly between zero and one")
        ranged_initialization = (
            initial_lambda_low is not None or initial_lambda_high is not None
        )
        if ranged_initialization and (
            initial_lambda_low is None
            or initial_lambda_high is None
            or not 0 < initial_lambda_low < initial_lambda_high < 1
        ):
            raise ValueError(
                "uniform lambda initialization requires 0 < low < high < 1"
            )
        if retention_mode == "all_slow":
            if all_slow_lambda is None or not 0 < all_slow_lambda <= 1:
                raise ValueError("all-slow mode requires lambda in (0, 1]")
        elif all_slow_lambda is not None:
            raise ValueError("all_slow_lambda is only valid in all-slow mode")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.state_size = self.width
        self.retention_mode = str(retention_mode)
        self.tau_sat = float(tau_sat)
        self.use_hard_saturation = True
        self.all_slow_lambda = (
            None if all_slow_lambda is None else float(all_slow_lambda)
        )
        hidden = int(writer_width or self.width)
        if hidden <= 0:
            raise ValueError("writer width must be positive")
        self.writer_width = hidden
        self.input_projection = nn.Linear(self.input_dim, self.width, bias=False)
        if ranged_initialization:
            initial_retention = torch.empty(self.width, dtype=torch.float32).uniform_(
                float(initial_lambda_low), float(initial_lambda_high)
            )
        else:
            initial_retention = torch.full(
                (self.width,), float(initial_lambda), dtype=torch.float32
            )
        theta_value = torch.logit(initial_retention)
        if not 0 <= int(fixed_unit_count) <= self.width:
            raise ValueError("fixed_unit_count must be in [0, width]")
        if not 0.0 <= float(fixed_fast_lambda) <= 1.0:
            raise ValueError("fixed_fast_lambda must be in [0, 1]")
        generator = torch.Generator().manual_seed(int(fixed_subset_seed))
        permutation = torch.randperm(self.width, generator=generator)
        unit_indices = permutation[: int(fixed_unit_count)]
        unit_mask = torch.zeros(self.width, dtype=torch.bool)
        unit_mask[unit_indices] = True
        free_indices = torch.nonzero(~unit_mask, as_tuple=False).flatten()
        self.register_buffer("fixed_unit_mask", unit_mask)
        self.register_buffer("fixed_free_indices", free_indices)
        self.fixed_unit_count = int(fixed_unit_count)
        self.fixed_fast_lambda = float(fixed_fast_lambda)
        self.fixed_subset_seed = int(fixed_subset_seed)

        if retention_mode == "direct_lambda_grad":
            self.lambda_param = nn.Parameter(initial_retention, requires_grad=False)
        elif retention_mode == "fixed_binary":
            fixed = torch.full((self.width,), float(fixed_fast_lambda))
            fixed[unit_mask] = 1.0
            self.register_buffer("fixed_lambda", fixed)
        elif retention_mode == "fixed_unit_bptt":
            self.theta_free = nn.Parameter(theta_value[free_indices].clone())
        else:
            self.theta = nn.Parameter(
                theta_value,
                requires_grad=retention_mode in {"bptt", "schedule_bptt"},
            )
        self.gamma = nn.Parameter(torch.ones(self.width))
        self.writer = nn.Sequential(
            nn.Linear(2 * self.width, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.width),
        )
        self.decoder = GLUReadout(
            self.width,
            self.width,
            self.output_dim,
        )

    @property
    def rp_enabled(self) -> bool:
        return self.retention_mode == "rp"

    @property
    def rp_grad_enabled(self) -> bool:
        return self.retention_mode in {
            "rp_grad",
            "direct_lambda_grad",
            "ste_recall_grad",
        }

    @property
    def probe_update_enabled(self) -> bool:
        return self.rp_enabled or self.rp_grad_enabled

    def retention(self) -> torch.Tensor:
        if self.retention_mode == "all_slow":
            return torch.full_like(self.theta, float(self.all_slow_lambda))
        if self.retention_mode == "direct_lambda_grad":
            return self.lambda_param
        if self.retention_mode == "fixed_binary":
            return self.fixed_lambda
        if self.retention_mode == "fixed_unit_bptt":
            result = torch.ones(
                self.width,
                dtype=self.theta_free.dtype,
                device=self.theta_free.device,
            )
            result[self.fixed_free_indices] = self._map_theta(self.theta_free)
            return result
        return self._map_theta(self.theta)

    def _map_theta(self, theta: torch.Tensor) -> torch.Tensor:
        if self.use_hard_saturation:
            return saturated_sigmoid(theta, self.tau_sat)
        return torch.sigmoid(theta)

    def retention_logits(self) -> torch.Tensor:
        """Return a full-width diagnostic logit vector for every control."""

        if hasattr(self, "theta"):
            return self.theta
        if self.retention_mode == "fixed_unit_bptt":
            result = torch.full(
                (self.width,),
                float("inf"),
                dtype=self.theta_free.dtype,
                device=self.theta_free.device,
            )
            result[self.fixed_free_indices] = self.theta_free
            return result
        retention = self.retention()
        return torch.logit(retention.clamp(min=1e-12, max=1.0 - 1e-7))

    def schedule_task_gradient_parameter(self) -> nn.Parameter | None:
        if self.retention_mode == "schedule_bptt":
            return self.theta
        if self.retention_mode == "fixed_unit_bptt":
            return self.theta_free
        return None

    def write(self, state: torch.Tensor, projected_input: torch.Tensor) -> torch.Tensor:
        actual = self.writer(torch.cat([state, projected_input], dim=-1))
        zero = self.writer(torch.cat([state, torch.zeros_like(projected_input)], dim=-1))
        return actual - zero

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x_t)
        return self.retention() * state + self.gamma * self.write(state, projected)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)

    def autonomous_rollout(self, state: torch.Tensor, horizon: int) -> torch.Tensor:
        if int(horizon) < 0:
            raise ValueError("autonomous horizon must be non-negative")
        return state * self.retention().pow(int(horizon))

    def retention_timescales(self) -> torch.Tensor:
        retention = self.retention()
        return -1.0 / torch.log(retention)

    @torch.no_grad()
    def rp_update(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        *,
        horizon: int,
        eta_lambda: float,
        retention_threshold: float,
        normalization_epsilon: float,
        return_diagnostics: bool = False,
    ) -> RPUpdateResult | tuple[RPUpdateResult, dict[str, torch.Tensor]]:
        if not self.rp_enabled:
            raise ValueError("RP update requested for a non-RP model")
        if states.ndim != 2 or states.shape[-1] != self.width:
            raise ValueError("RP states have the wrong shape")
        if targets.shape != (states.shape[0], self.output_dim):
            raise ValueError("RP targets have the wrong shape")
        if eta_lambda <= 0 or retention_threshold < 0:
            raise ValueError("invalid RP update hyperparameters")
        if normalization_epsilon <= 0:
            raise ValueError("normalization epsilon must be positive")
        probe_count = states.shape[0]
        theta_before = self.theta.detach().clone()
        lambda_before = self.retention().detach().clone()
        decay = lambda_before.pow(int(horizon))
        clean_prediction = self.decode(states * decay)
        clean_error = (clean_prediction - targets).square().sum(dim=-1)

        ablated = states.unsqueeze(0).expand(
            self.width, probe_count, self.width
        ).clone()
        coordinate = torch.arange(self.width, device=states.device)
        ablated[coordinate, :, coordinate] = 0.0
        prediction = self.decode(ablated * decay)
        ablated_error = (
            prediction - targets.unsqueeze(0)
        ).square().sum(dim=-1)
        damage = (ablated_error - clean_error.unsqueeze(0)).mean(dim=1)
        perturbation_energy = states.square().mean(dim=0)
        normalized_score = damage / (
            perturbation_energy + float(normalization_epsilon)
        )
        allocation = normalized_score - float(retention_threshold)
        self.theta.add_(float(eta_lambda) * allocation)
        if not torch.isfinite(self.theta).all():
            raise FloatingPointError("RP produced non-finite retention logits")
        retention = self.retention()
        result = RPUpdateResult(
            horizon=int(horizon),
            clean_mse=float(clean_error.mean().cpu() / self.output_dim),
            damage_mean=float(damage.mean().cpu()),
            damage_max=float(damage.max().cpu()),
            normalized_score_mean=float(normalized_score.mean().cpu()),
            normalized_score_max=float(normalized_score.max().cpu()),
            normalized_score_min=float(normalized_score.min().cpu()),
            negative_score_fraction=float(
                (normalized_score < 0).float().mean().cpu()
            ),
            weak_positive_score_fraction=float(
                (
                    (normalized_score > 0)
                    & (normalized_score < 0.1)
                ).float().mean().cpu()
            ),
            positive_allocation_fraction=float((allocation > 0).float().mean().cpu()),
            lambda_min=float(retention.min().cpu()),
            lambda_mean=float(retention.mean().cpu()),
            lambda_max=float(retention.max().cpu()),
            exact_unit_count=int((retention == 1.0).sum().cpu()),
        )
        if not return_diagnostics:
            return result
        return result, {
            "theta_before": theta_before,
            "theta_after": self.theta.detach().clone(),
            "lambda_before": lambda_before,
            "lambda_after": retention.detach().clone(),
            "damage": damage.detach().clone(),
            "perturbation_energy": perturbation_energy.detach().clone(),
            "normalized_score": normalized_score.detach().clone(),
            "allocation": allocation.detach().clone(),
        }

    def rp_grad_update(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        *,
        horizon: int,
        eta_gradient: float,
    ) -> tuple[RPGradUpdateResult, dict[str, torch.Tensor]]:
        """Minimize Eq. (7) through theta while keeping theta out of task BPTT."""

        if not self.rp_grad_enabled:
            raise ValueError("RP-Grad update requested for a non-RP-Grad model")
        if states.ndim != 2 or states.shape[-1] != self.width:
            raise ValueError("RP-Grad states have the wrong shape")
        if targets.shape != (states.shape[0], self.output_dim):
            raise ValueError("RP-Grad targets have the wrong shape")
        if int(horizon) <= 0 or float(eta_gradient) <= 0:
            raise ValueError("invalid RP-Grad update hyperparameters")

        states = states.detach()
        targets = targets.detach()
        with torch.enable_grad():
            direct_lambda = self.retention_mode == "direct_lambda_grad"
            ste = self.retention_mode == "ste_recall_grad"
            if direct_lambda:
                theta_before = self.retention_logits().detach().clone()
                lambda_variable = self.lambda_param.detach().clone().requires_grad_(True)
                lambda_before = lambda_variable
            else:
                theta_before = self.theta.detach().clone().requires_grad_(True)
                lambda_hard = saturated_sigmoid(theta_before, self.tau_sat)
                lambda_before = (
                    lambda_hard.detach() + theta_before - theta_before.detach()
                    if ste
                    else lambda_hard
                )
            decay = lambda_before.pow(int(horizon))
            prediction = self.decode(states * decay)
            squared_error = (prediction - targets).square()
            # Eq. (7) is ||D(F_0^K(h))-z||_2^2: average over the
            # probe batch while summing over output coordinates.
            probe_loss = squared_error.sum(dim=-1).mean()
            gradient_lambda = torch.autograd.grad(
                probe_loss, lambda_before, retain_graph=not direct_lambda
            )[0]
            if direct_lambda:
                gradient_theta = gradient_lambda
                sigmoid_derivative = torch.ones_like(lambda_before)
            else:
                gradient_theta = torch.autograd.grad(probe_loss, theta_before)[0]
                sigmoid_derivative = (
                    torch.ones_like(lambda_before)
                    if ste
                    else lambda_hard * (1.0 - lambda_hard)
                )
            chain_rule = gradient_lambda * sigmoid_derivative
            if not torch.allclose(gradient_theta, chain_rule, rtol=2e-5, atol=1e-8):
                maximum_error = float(
                    (gradient_theta - chain_rule).abs().max().detach().cpu()
                )
                raise AssertionError(
                    f"RP-Grad chain-rule audit failed: {maximum_error}"
                )

        if not torch.isfinite(gradient_theta).all():
            raise FloatingPointError("RP-Grad produced non-finite theta gradients")
        theta_step = -float(eta_gradient) * gradient_theta.detach()
        with torch.no_grad():
            if direct_lambda:
                self.lambda_param.add_(theta_step).clamp_(0.0, 1.0)
            else:
                self.theta.add_(theta_step)
                if not torch.isfinite(self.theta).all():
                    raise FloatingPointError("RP-Grad produced non-finite logits")
            lambda_after = self.retention().detach().clone()
            theta_after = self.retention_logits().detach().clone()

        result = RPGradUpdateResult(
            horizon=int(horizon),
            probe_loss=float(probe_loss.detach().cpu()),
            gradient_l2=float(torch.linalg.vector_norm(gradient_theta).detach().cpu()),
            gradient_max_abs=float(gradient_theta.abs().max().detach().cpu()),
            theta_step_l2=float(torch.linalg.vector_norm(theta_step).cpu()),
            theta_step_max_abs=float(theta_step.abs().max().cpu()),
            lambda_min=float(lambda_after.min().cpu()),
            lambda_mean=float(lambda_after.mean().cpu()),
            lambda_max=float(lambda_after.max().cpu()),
            exact_unit_count=int((lambda_after == 1.0).sum().cpu()),
        )
        diagnostics = {
            "theta_before": theta_before.detach(),
            "theta_after": theta_after,
            "lambda_before": lambda_before.detach(),
            "lambda_after": lambda_after,
            "gradient_theta": gradient_theta.detach(),
            "gradient_lambda": gradient_lambda.detach(),
            "sigmoid_derivative": sigmoid_derivative.detach(),
        }
        return result, diagnostics


def build_model(
    name: str,
    *,
    dimension: int,
    width: int,
    retention_mode: str = "rp",
    initial_lambda: float = 0.9,
    initial_lambda_low: float | None = None,
    initial_lambda_high: float | None = None,
    all_slow_lambda: float | None = None,
    chrono_t_max: int = 255,
    tau_sat: float = 16.64,
    fixed_unit_count: int = 0,
    fixed_fast_lambda: float = 0.0,
    fixed_subset_seed: int = 0,
) -> SequenceModel:
    key = str(name).lower()
    if key not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    input_dim = int(dimension) + 2
    output_dim = int(dimension)
    if key == "rnn":
        return RNNModel(input_dim, output_dim, width)
    if key == "gru":
        return GRUModel(input_dim, output_dim, width)
    if key == "lstm":
        return LSTMModel(input_dim, output_dim, width)
    if key == "lru":
        return LRUModel(input_dim, output_dim, width)
    if key in EXTENDED_BASELINE_NAMES:
        from .extended_baselines import (
            ChronoLSTMModel,
            OrthogonalRNNModel,
            S4DLegSModel,
        )

        if key == "orthogonal_rnn":
            return OrthogonalRNNModel(input_dim, output_dim, width)
        if key == "s4d_legs":
            return S4DLegSModel(input_dim, output_dim, width)
        return ChronoLSTMModel(
            input_dim,
            output_dim,
            width,
            chrono_t_max=int(chrono_t_max),
        )
    return RPLRUModel(
        input_dim,
        output_dim,
        width,
        retention_mode=retention_mode,
        initial_lambda=initial_lambda,
        initial_lambda_low=initial_lambda_low,
        initial_lambda_high=initial_lambda_high,
        all_slow_lambda=all_slow_lambda,
        tau_sat=tau_sat,
        fixed_unit_count=fixed_unit_count,
        fixed_fast_lambda=fixed_fast_lambda,
        fixed_subset_seed=fixed_subset_seed,
    )


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def parameter_count_formula(
    name: str,
    *,
    dimension: int,
    width: int,
    trainable_only: bool = False,
    retention_mode: str = "rp",
    fixed_unit_count: int = 0,
) -> int:
    """Exact count for the constructors above without allocating the model."""

    key = str(name).lower()
    d = int(dimension)
    n = int(width)
    if d <= 0 or n <= 0:
        raise ValueError("dimension and width must be positive")
    if key == "rnn":
        return n * n + n * (2 * d + 4) + d
    if key == "gru":
        return 3 * n * n + n * (4 * d + 12) + d
    if key == "lstm":
        return 4 * n * n + n * (5 * d + 16) + d
    if key == "lru":
        return 6 * n * n + n * (2 * d + 7) + d
    if key == "rp_lru":
        total = 6 * n * n + n * (2 * d + 9) + d
        if retention_mode == "fixed_binary":
            return total - n
        if retention_mode == "fixed_unit_bptt":
            if not 0 <= int(fixed_unit_count) <= n:
                raise ValueError("fixed_unit_count must be in [0, width]")
            return total - int(fixed_unit_count)
        if trainable_only and retention_mode in {
            "rp",
            "rp_grad",
            "frozen",
            "all_slow",
            "direct_lambda_grad",
            "ste_recall_grad",
        }:
            total -= n
        return total
    if key in EXTENDED_BASELINE_NAMES:
        from .extended_baselines import extended_parameter_count_formula

        return extended_parameter_count_formula(key, d, n)
    raise ValueError(f"unknown model {name!r}")


@dataclass(frozen=True)
class ParameterMatch:
    model: str
    width: int
    parameter_count: int
    target_count: int
    relative_gap: float
    count_mode: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "width": self.width,
            "parameter_count": self.parameter_count,
            "target_count": self.target_count,
            "relative_gap": self.relative_gap,
            "count_mode": self.count_mode,
        }


def parameter_matched_width(
    name: str,
    *,
    dimension: int,
    target_count: int,
    minimum_width: int = 4,
    maximum_width: int = 16384,
    trainable_only: bool = False,
) -> ParameterMatch:
    """Find the closest integer width under an exact monotone count formula."""

    key = str(name).lower()
    if key == "rp_lru":
        raise ValueError("RP-LRU defines the target and is not width-matched")
    if target_count <= 0:
        raise ValueError("target parameter count must be positive")
    low = int(minimum_width)
    high = int(maximum_width)
    if low <= 0 or low > high:
        raise ValueError("invalid width-search interval")
    while low < high:
        middle = (low + high) // 2
        count = parameter_count_formula(
            key,
            dimension=dimension,
            width=middle,
            trainable_only=trainable_only,
        )
        if count < target_count:
            low = middle + 1
        else:
            high = middle
    candidates = {low, max(int(minimum_width), low - 1)}
    width = min(
        candidates,
        key=lambda value: (
            abs(
                parameter_count_formula(
                    key,
                    dimension=dimension,
                    width=value,
                    trainable_only=trainable_only,
                )
                - target_count
            ),
            value,
        ),
    )
    count = parameter_count_formula(
        key,
        dimension=dimension,
        width=width,
        trainable_only=trainable_only,
    )
    return ParameterMatch(
        model=key,
        width=width,
        parameter_count=count,
        target_count=int(target_count),
        relative_gap=abs(count - target_count) / float(target_count),
        count_mode=(
            "trainable_parameters" if trainable_only else "total_parameters"
        ),
    )
