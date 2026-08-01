"""Additional long-memory baselines for the final IVI experiment.

The implementations are intentionally dependency-free and expose the same
step/decode interface as the original comparison models:

* ``OrthogonalRNNModel`` follows the expRNN construction (matrix exponential
  of a skew-symmetric generator, Henaff initialization, and modReLU).
* ``S4DLegSModel`` is a recurrent diagonal SSM initialized from the official
  S4D-LegS NPLR spectrum.  Its nonlinear mixing remains outside recurrence.
* ``ChronoLSTMModel`` changes only the LSTM gate-bias initialization.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .models import GLUReadout, LSTMModel, SequenceModel


class OrthogonalRNNModel(SequenceModel):
    """Real expRNN-style baseline with an exactly orthogonal kernel.

    Only the independent upper-triangular entries of the skew generator are
    stored, so parameter matching counts degrees of freedom rather than a
    redundant dense carrier matrix.
    """

    recurrent_learning_rate_ratio = 0.1

    def __init__(self, input_dim: int, output_dim: int, width: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.state_size = self.width
        indices = torch.triu_indices(self.width, self.width, offset=1)
        self.register_buffer("skew_rows", indices[0], persistent=False)
        self.register_buffer("skew_cols", indices[1], persistent=False)
        self.skew = nn.Parameter(torch.zeros(indices.shape[1]))
        self.input_projection = nn.Linear(self.input_dim, self.width, bias=False)
        self.modrelu_bias = nn.Parameter(torch.empty(self.width))
        self.decoder = nn.Linear(self.width, self.output_dim)
        self._cached_transition: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="relu")
        nn.init.uniform_(self.modrelu_bias, -0.01, 0.01)
        with torch.no_grad():
            self.skew.zero_()
            # Henaff initialization: independent 2x2 rotation blocks with
            # angles uniformly distributed over the full circle.
            pair_angle = torch.empty(
                self.width // 2,
                device=self.skew.device,
                dtype=self.skew.dtype,
            ).uniform_(-math.pi, math.pi)
            for pair, angle in enumerate(pair_angle):
                row = 2 * pair
                column = row + 1
                match = (self.skew_rows == row) & (self.skew_cols == column)
                self.skew[match] = angle

    def generator(self) -> torch.Tensor:
        matrix = self.skew.new_zeros((self.width, self.width))
        matrix[self.skew_rows, self.skew_cols] = self.skew
        return matrix - matrix.transpose(0, 1)

    def transition_matrix(self) -> torch.Tensor:
        if not self.training and self._cached_transition is not None:
            return self._cached_transition
        transition = torch.matrix_exp(self.generator())
        if not self.training:
            self._cached_transition = transition.detach()
        return transition

    def train(self, mode: bool = True):
        self._cached_transition = None
        return super().train(mode)

    def _step_with_transition(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        transition: torch.Tensor,
    ) -> torch.Tensor:
        preactivation = self.input_projection(x_t) + state @ transition.transpose(0, 1)
        return torch.sign(preactivation) * torch.relu(
            torch.abs(preactivation) + self.modrelu_bias
        )

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self._step_with_transition(x_t, state, self.transition_matrix())

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Materialize the matrix exponential once per training sequence."""

        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError("orthogonal RNN input has the wrong shape")
        if lengths.shape != (inputs.shape[1],):
            raise ValueError("length tensor has the wrong shape")
        transition = self.transition_matrix()
        state = self.initial_state(
            inputs.shape[1], device=inputs.device, dtype=inputs.dtype
        )
        prediction = inputs.new_empty((inputs.shape[1], self.output_dim))
        for time, x_t in enumerate(inputs):
            state = self._step_with_transition(x_t, state, transition)
            final = lengths.to(inputs.device) == time + 1
            if torch.any(final):
                prediction[final] = self.decode(state[final])
        return prediction

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)

    def optimizer_parameter_groups(
        self, *, learning_rate: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        recurrent = [self.skew]
        remaining = [
            parameter
            for name, parameter in self.named_parameters()
            if name != "skew" and parameter.requires_grad
        ]
        return [
            {
                "params": remaining,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
                "group_name": "task_parameters",
            },
            {
                "params": recurrent,
                "lr": float(learning_rate) * self.recurrent_learning_rate_ratio,
                "weight_decay": 0.0,
                "group_name": "orthogonal_generator",
            },
        ]

    @torch.no_grad()
    def diagnostic_tensors(self, *, horizon: int = 17017) -> dict[str, torch.Tensor]:
        transition = self.transition_matrix().to(torch.float64)
        eigenvalues = torch.linalg.eigvals(transition)
        phase = torch.angle(eigenvalues)
        return {
            "eigenvalue_real": eigenvalues.real.cpu(),
            "eigenvalue_imag": eigenvalues.imag.cpu(),
            "eigenvalue_magnitude": eigenvalues.abs().cpu(),
            "eigenvalue_phase": phase.cpu(),
            "phase_at_horizon": torch.remainder(
                phase * int(horizon) + math.pi, 2.0 * math.pi
            ).sub(math.pi).cpu(),
            "signed_rotation_cycles": (
                phase * int(horizon) / (2.0 * math.pi)
            ).cpu(),
        }

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "family": "expRNN_style_real_orthogonal_rnn",
            "orthogonal_map": "matrix_exponential",
            "generator_storage": "independent_upper_triangle",
            "initialization": "henaff_uniform_rotation_blocks",
            "activation": "modReLU",
            "recurrent_learning_rate_ratio": self.recurrent_learning_rate_ratio,
            "reference": "https://github.com/Lezcano/expRNN",
        }


def _legs_nplr_diagonal(order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the official S4D-LegS diagonal A and transformed B.

    This is a compact dependency-free transcription of ``transition('legs')``
    and ``nplr('legs')`` from state-spaces/s4.  ``order`` must be even because
    one member of each conjugate eigenvalue pair is retained.
    """

    if int(order) <= 0 or int(order) % 2:
        raise ValueError("LegS order must be a positive even integer")
    n = int(order)
    q = torch.arange(n, dtype=torch.float64)
    row = q[:, None]
    column = q[None, :]
    r = 2.0 * q + 1.0
    m = -(torch.where(row >= column, r[None, :], 0.0) - torch.diag(q))
    scale = torch.sqrt(2.0 * q + 1.0)
    transition = scale[:, None] * m / scale[None, :]
    input_vector = scale
    correction = torch.sqrt(0.5 + q)
    normal = transition + correction[:, None] * correction[None, :]
    imaginary, vectors = torch.linalg.eigh(normal.to(torch.complex128) * (-1j))
    real = torch.diagonal(normal).mean()
    eigenvalues = real.to(torch.complex128) + 1j * imaginary
    order_index = torch.argsort(eigenvalues.imag)
    eigenvalues = eigenvalues[order_index][: n // 2]
    vectors = vectors[:, order_index][:, : n // 2]
    transformed_input = vectors.conj().transpose(0, 1) @ input_vector.to(
        torch.complex128
    )
    transformed_input = transformed_input.real + 1j * transformed_input.imag.clamp(
        min=-2.0, max=2.0
    )
    return eigenvalues.to(torch.complex64), transformed_input.to(torch.complex64)


class S4DLegSModel(SequenceModel):
    """Compact recurrent S4D-LegS baseline with a full-state GLU readout."""

    ssm_learning_rate_cap = 1e-3
    dt_min = 1e-3
    dt_max = 1e-1

    def __init__(self, input_dim: int, output_dim: int, width: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.state_size = 2 * self.width
        self.input_projection = nn.Linear(self.input_dim, self.width, bias=False)
        diagonal, input_vector = _legs_nplr_diagonal(2 * self.width)
        self.log_dt = nn.Parameter(
            torch.empty((), dtype=torch.float32).uniform_(
                math.log(self.dt_min), math.log(self.dt_max)
            )
        )
        self.log_A_real = nn.Parameter(torch.log(-diagonal.real))
        self.A_imag = nn.Parameter(-diagonal.imag)
        self.B_re = nn.Parameter(input_vector.real.clone())
        self.B_im = nn.Parameter(input_vector.imag.clone())
        self.decoder = GLUReadout(2 * self.width, self.width, self.output_dim)

    def continuous_eigenvalues(self) -> torch.Tensor:
        return torch.complex(-torch.exp(self.log_A_real), -self.A_imag)

    def discrete_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        eigenvalues = self.continuous_eigenvalues()
        step = torch.exp(self.log_dt)
        discrete_a = torch.exp(step * eigenvalues)
        continuous_b = torch.complex(self.B_re, self.B_im)
        discrete_b = continuous_b * (discrete_a - 1.0) / eigenvalues
        return discrete_a, discrete_b

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        hidden_re, hidden_im = state.split(self.width, dim=-1)
        hidden = torch.complex(hidden_re, hidden_im)
        drive = self.input_projection(x_t)
        discrete_a, discrete_b = self.discrete_parameters()
        next_hidden = hidden * discrete_a + drive * discrete_b
        return torch.cat([next_hidden.real, next_hidden.imag], dim=-1)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)

    def optimizer_parameter_groups(
        self, *, learning_rate: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        ssm_names = {"log_dt", "log_A_real", "A_imag", "B_re", "B_im"}
        ssm = [
            parameter
            for name, parameter in self.named_parameters()
            if name in ssm_names and parameter.requires_grad
        ]
        remaining = [
            parameter
            for name, parameter in self.named_parameters()
            if name not in ssm_names and parameter.requires_grad
        ]
        return [
            {
                "params": remaining,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
                "group_name": "task_parameters",
            },
            {
                "params": ssm,
                "lr": min(float(learning_rate), self.ssm_learning_rate_cap),
                "weight_decay": 0.0,
                "group_name": "s4d_ssm_parameters",
            },
        ]

    @torch.no_grad()
    def diagnostic_tensors(self, *, horizon: int = 17017) -> dict[str, torch.Tensor]:
        continuous = self.continuous_eigenvalues()
        discrete, _ = self.discrete_parameters()
        return {
            "continuous_eigenvalue_real": continuous.real.cpu(),
            "continuous_eigenvalue_imag": continuous.imag.cpu(),
            "discrete_eigenvalue_real": discrete.real.cpu(),
            "discrete_eigenvalue_imag": discrete.imag.cpu(),
            "discrete_eigenvalue_magnitude": discrete.abs().cpu(),
            "retention_at_horizon": discrete.abs().pow(int(horizon)).cpu(),
            "dt": torch.exp(self.log_dt.detach()).reshape(1).cpu(),
        }

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "family": "S4D_LegS",
            "continuous_initialization": "official_LegS_NPLR_diagonal_and_B",
            "discretization": "zero_order_hold",
            "continuous_real_part_parameterization": "negative_exp",
            "readout": "single_GLU_residual",
            "normalization": "none",
            "ssm_learning_rate_cap": self.ssm_learning_rate_cap,
            "reference": "https://github.com/state-spaces/s4",
        }


class ChronoLSTMModel(LSTMModel):
    """Standard PyTorch LSTM with Tallec--Ollivier gate biases."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        *,
        chrono_t_max: int = 255,
    ):
        if int(chrono_t_max) <= 2:
            raise ValueError("chrono T_max must exceed two")
        self.chrono_t_max = int(chrono_t_max)
        super().__init__(input_dim, output_dim, width)
        self._chrono_initialize()

    def _chrono_initialize(self) -> None:
        with torch.no_grad():
            self.cell.bias_ih.zero_()
            self.cell.bias_hh.zero_()
            forget = torch.log(
                torch.empty(self.width, dtype=self.cell.bias_ih.dtype).uniform_(
                    1.0, float(self.chrono_t_max - 1)
                )
            )
            self.cell.bias_ih[: self.width] = -forget
            self.cell.bias_ih[self.width : 2 * self.width] = forget

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "family": "chrono_initialized_LSTM",
            "chrono_t_max": self.chrono_t_max,
            "input_gate_bias": "negative_forget_gate_bias",
            "forget_gate_bias_distribution": "log_uniform_1_to_Tmax_minus_1",
            "readout": "standard_linear",
        }


def extended_parameter_count_formula(name: str, dimension: int, width: int) -> int:
    key = str(name).lower()
    d = int(dimension)
    n = int(width)
    if key == "orthogonal_rnn":
        return n * (n - 1) // 2 + n * (d + 2) + n + n * d + d
    if key == "s4d_legs":
        return 4 * n * n + n * (2 * d + 9) + d + 1
    if key == "chrono_lstm":
        return 4 * n * n + n * (5 * d + 16) + d
    raise ValueError(f"unknown extended baseline {name!r}")
