"""State semantics for the Ságodi continuous-attractor protocol.

The legacy experiment models expose one flat ``state`` tensor.  A
``FullBlockSequenceModel`` appends the most recently generated output stream
to its recurrent carrier even when that stream is overwritten, without
feedback, at the next step.  Treating that diagnostic stream as recurrent
state changes distances and Jacobians without changing the Markov dynamics.

``StateAdapter`` makes the distinction explicit while continuing to call the
legacy model's public ``step`` and ``decode`` methods.  It deliberately uses
duck typing rather than importing the legacy classes, so importing this module
does not mutate ``sys.path`` or couple the protocol to a particular legacy
file layout.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional

import torch


class StateParts(NamedTuple):
    """Contiguous components of a legacy reported state.

    ``stream`` is ``None`` for StepBaseline-style models.  The named tuple can
    also be unpacked as ``carrier, stream = adapter.unpack(state)``.
    """

    carrier: torch.Tensor
    stream: Optional[torch.Tensor]


def _first_floating_parameter_or_buffer(model: torch.nn.Module) -> Optional[torch.Tensor]:
    for tensor in model.parameters():
        if tensor.is_floating_point():
            return tensor
    for tensor in model.buffers():
        if tensor.is_floating_point():
            return tensor
    return None


class StateAdapter:
    """Expose the minimal Markov state of a legacy step model.

    The adapter supports the two state layouts used by the checked-in legacy
    code:

    * StepBaseline-like model: its entire flat state is primary.
    * FullBlock-like model: ``[recurrent carrier | generated stream]``.

    For a full block with ``carry_stream=False``, only the carrier is the
    primary Markov state.  The stream remains part of the *reported* state so
    that the original decoder and mechanism diagnostics can be reproduced.
    With ``carry_stream=True`` the stream feeds the next transition and the
    complete reported state is therefore primary.
    """

    def __init__(self, model: torch.nn.Module):
        required = ("state_size", "input_dim", "init_state", "step", "decode")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise TypeError(f"model is missing the legacy step interface: {missing}")

        self.model = model
        self.reported_dim = int(model.state_size)
        self.input_dim = int(model.input_dim)
        if self.reported_dim <= 0 or self.input_dim <= 0:
            raise ValueError("state_size and input_dim must be positive")

        recurrent_size = getattr(model, "recurrent_state_size", None)
        self.is_full_block = recurrent_size is not None
        if self.is_full_block:
            self.carrier_dim = int(recurrent_size)
            readout_slice = getattr(model, "readout_slice", None)
            if not isinstance(readout_slice, slice):
                raise ValueError("full-block state has no readout_slice")
            if readout_slice.step not in (None, 1):
                raise ValueError("full-block stream slice must be contiguous")
            if readout_slice.start != self.carrier_dim or readout_slice.stop != self.reported_dim:
                raise ValueError(
                    "unsupported full-block state layout: expected "
                    f"carrier [0:{self.carrier_dim}] and stream "
                    f"[{self.carrier_dim}:{self.reported_dim}], got {readout_slice}"
                )
            if not 0 < self.carrier_dim < self.reported_dim:
                raise ValueError(
                    f"invalid full-block carrier dimension {self.carrier_dim}/{self.reported_dim}"
                )
            self.stream_dim = self.reported_dim - self.carrier_dim
            self.carry_stream = bool(getattr(model, "carry_stream", False))
        else:
            self.carrier_dim = self.reported_dim
            self.stream_dim = 0
            self.carry_stream = False

        self.stream_is_primary = bool(self.is_full_block and self.carry_stream)
        self.primary_dim = self.reported_dim if self.stream_is_primary else self.carrier_dim

    @property
    def state_size(self) -> int:
        """Dimension of the primary Markov state (batch dimension excluded)."""

        return self.primary_dim

    @property
    def reported_state_size(self) -> int:
        """Dimension returned by the wrapped legacy model."""

        return self.reported_dim

    @property
    def device(self) -> torch.device:
        reference = _first_floating_parameter_or_buffer(self.model)
        return reference.device if reference is not None else torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        reference = _first_floating_parameter_or_buffer(self.model)
        return reference.dtype if reference is not None else torch.get_default_dtype()

    def _validate(self, state: torch.Tensor, dimension: int, label: str) -> None:
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor")
        if state.ndim != 2:
            raise ValueError(f"{label} must have shape [batch, {dimension}], got {tuple(state.shape)}")
        if state.shape[-1] != dimension:
            raise ValueError(
                f"{label} has final dimension {state.shape[-1]}, expected {dimension}"
            )
        if not state.is_floating_point():
            raise TypeError(f"{label} must use a floating dtype")

    def _validate_input(self, input_tensor: torch.Tensor, batch: int) -> None:
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("input_tensor must be a torch.Tensor")
        if input_tensor.ndim != 2 or input_tensor.shape != (batch, self.input_dim):
            raise ValueError(
                "input_tensor must have shape "
                f"[{batch}, {self.input_dim}], got {tuple(input_tensor.shape)}"
            )

    def zero_input(self, state_or_batch: Any) -> torch.Tensor:
        """Create the literal external zero input used by the actual map F0."""

        if isinstance(state_or_batch, torch.Tensor):
            batch = int(state_or_batch.shape[0])
            device = state_or_batch.device
            dtype = state_or_batch.dtype
        else:
            batch = int(state_or_batch)
            if batch <= 0:
                raise ValueError("batch must be positive")
            device = self.device
            dtype = self.dtype
        return torch.zeros(batch, self.input_dim, device=device, dtype=dtype)

    def init_reported(
        self,
        batch: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Initialize the original legacy state, preserving model dtype."""

        batch = int(batch)
        if batch <= 0:
            raise ValueError("batch must be positive")
        target_device = torch.device(device) if device is not None else self.device
        target_dtype = dtype if dtype is not None else self.dtype
        state = self.model.init_state(batch, target_device)
        if not isinstance(state, torch.Tensor):
            raise TypeError("legacy init_state did not return a torch.Tensor")
        state = state.to(device=target_device, dtype=target_dtype)
        self._validate(state, self.reported_dim, "reported state")
        return state

    def init_primary(
        self,
        batch: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return self.primary_from_reported(
            self.init_reported(batch, device=device, dtype=dtype)
        )

    def unpack(self, reported_state: torch.Tensor) -> StateParts:
        """Split a reported legacy state into contiguous carrier and stream."""

        self._validate(reported_state, self.reported_dim, "reported state")
        if not self.is_full_block:
            return StateParts(reported_state, None)
        return StateParts(
            reported_state[:, : self.carrier_dim],
            reported_state[:, self.carrier_dim :],
        )

    def pack(
        self,
        carrier: torch.Tensor | StateParts,
        stream: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pack state components into the legacy reported-state layout.

        Omitting ``stream`` for a full block inserts an exact zero stream.  It
        is safe for ``carry_stream=False`` and useful when evaluating the
        primary map; callers working with a carried stream should normally
        pass it explicitly.
        """

        if isinstance(carrier, StateParts):
            if stream is not None:
                raise ValueError("stream was supplied twice")
            carrier, stream = carrier
        self._validate(carrier, self.carrier_dim, "carrier")
        if not self.is_full_block:
            if stream is not None:
                raise ValueError("StepBaseline-style state has no stream component")
            return carrier
        if stream is None:
            stream = torch.zeros(
                carrier.shape[0],
                self.stream_dim,
                device=carrier.device,
                dtype=carrier.dtype,
            )
        self._validate(stream, self.stream_dim, "stream")
        if stream.shape[0] != carrier.shape[0]:
            raise ValueError("carrier and stream batch dimensions differ")
        if stream.device != carrier.device or stream.dtype != carrier.dtype:
            raise ValueError("carrier and stream must have the same device and dtype")
        return torch.cat([carrier, stream], dim=-1)

    def primary_from_reported(self, reported_state: torch.Tensor) -> torch.Tensor:
        """Project a legacy state onto its minimal Markov state."""

        self._validate(reported_state, self.reported_dim, "reported state")
        if self.stream_is_primary:
            return reported_state
        return reported_state[:, : self.primary_dim]

    def reported_from_primary(
        self,
        primary_state: torch.Tensor,
        stream: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed primary state into the legacy layout.

        For a carried-stream model, primary and reported states are identical;
        an optional ``stream`` replaces the stream coordinates explicitly.
        """

        self._validate(primary_state, self.primary_dim, "primary state")
        if not self.stream_is_primary:
            return self.pack(primary_state, stream)
        if stream is None:
            return primary_state
        parts = self.unpack(primary_state)
        return self.pack(parts.carrier, stream)

    def replace_stream(
        self, reported_state: torch.Tensor, replacement_stream: torch.Tensor
    ) -> torch.Tensor:
        if not self.is_full_block:
            raise ValueError("model has no stream component")
        parts = self.unpack(reported_state)
        return self.pack(parts.carrier, replacement_stream)

    def _input_or_zero(
        self, state: torch.Tensor, input_tensor: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if input_tensor is None:
            return self.zero_input(state)
        self._validate_input(input_tensor, int(state.shape[0]))
        if input_tensor.device != state.device or input_tensor.dtype != state.dtype:
            raise ValueError("state and input_tensor must have the same device and dtype")
        return input_tensor

    def reported_step(
        self,
        reported_state: torch.Tensor,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the wrapped model's literal one-step map.

        ``input_tensor=None`` means an exact zero tensor, not a shortcut to a
        retention-only equation.
        """

        self._validate(reported_state, self.reported_dim, "reported state")
        input_value = self._input_or_zero(reported_state, input_tensor)
        result = self.model.step(input_value, reported_state)
        self._validate(result, self.reported_dim, "next reported state")
        return result

    def primary_step(
        self,
        primary_state: torch.Tensor,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply F(s, u) to the minimal Markov state.

        For an overwritten stream, the primary state is embedded with zeros;
        the stream intervention audit verifies that this arbitrary placeholder
        cannot affect the next primary state.
        """

        self._validate(primary_state, self.primary_dim, "primary state")
        input_value = self._input_or_zero(primary_state, input_tensor)
        reported = self.reported_from_primary(primary_state)
        next_reported = self.model.step(input_value, reported)
        self._validate(next_reported, self.reported_dim, "next reported state")
        return self.primary_from_reported(next_reported)

    def actual_f0(self, primary_state: torch.Tensor) -> torch.Tensor:
        """Alias spelling out that F0 is ``model.step(zero, state)``."""

        return self.primary_step(primary_state, None)

    def decode(
        self,
        state: torch.Tensor,
        *,
        stream: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode a reported state, or a primary state when unambiguous.

        A stream-decoded full block with an overwritten stream cannot decode a
        carrier alone: the generated stream is a transition output, not a
        function of the post-transition carrier in the public legacy API.
        Pass the matching ``stream`` or the complete reported state.
        """

        if state.ndim != 2:
            raise ValueError("state must be a rank-2 tensor")
        if state.shape[-1] == self.reported_dim:
            if stream is not None:
                if not self.is_full_block:
                    raise ValueError("model has no stream component")
                state = self.replace_stream(state, stream)
            return self.model.decode(state)
        if state.shape[-1] != self.primary_dim:
            raise ValueError(
                f"state dimension must be primary={self.primary_dim} or "
                f"reported={self.reported_dim}, got {state.shape[-1]}"
            )
        decode_mode = str(getattr(self.model, "decode_mode", "model_defined"))
        if self.is_full_block and decode_mode == "stream" and stream is None:
            raise ValueError(
                "stream-decoded full block requires its matching reported stream"
            )
        reported = self.reported_from_primary(state, stream)
        return self.model.decode(reported)

    def stream_intervention(
        self,
        reported_state: torch.Tensor,
        input_tensor: Optional[torch.Tensor] = None,
        *,
        replacement_stream: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Test whether changing the old stream changes the next transition.

        The return value contains only JSON-compatible values.  For
        ``carry_stream=False`` both the next carrier and the entire next
        reported state must be equal because the old stream is overwritten
        before it can be read.
        """

        self._validate(reported_state, self.reported_dim, "reported state")
        if not self.is_full_block:
            return {
                "applicable": False,
                "required_no_feedback": False,
                "reason": "model_has_no_separate_stream",
                "passed": True,
            }
        parts = self.unpack(reported_state)
        assert parts.stream is not None
        if replacement_stream is None:
            values = torch.linspace(
                -1.0,
                1.0,
                self.stream_dim,
                device=reported_state.device,
                dtype=reported_state.dtype,
            ).unsqueeze(0)
            replacement_stream = values.expand(reported_state.shape[0], -1)
            if torch.equal(replacement_stream, parts.stream):
                replacement_stream = replacement_stream + 1.0
        self._validate(replacement_stream, self.stream_dim, "replacement stream")
        if replacement_stream.shape[0] != reported_state.shape[0]:
            raise ValueError("replacement stream batch dimension differs")
        if replacement_stream.device != reported_state.device or replacement_stream.dtype != reported_state.dtype:
            raise ValueError("replacement stream must match state device and dtype")

        changed = self.pack(parts.carrier, replacement_stream)
        input_value = self._input_or_zero(reported_state, input_tensor)
        next_base = self.reported_step(reported_state, input_value)
        next_changed = self.reported_step(changed, input_value)
        primary_base = self.primary_from_reported(next_base)
        primary_changed = self.primary_from_reported(next_changed)
        primary_delta = primary_changed - primary_base
        reported_delta = next_changed - next_base
        initial_stream_delta = replacement_stream - parts.stream
        primary_equal = bool(torch.equal(primary_base, primary_changed))
        reported_equal = bool(torch.equal(next_base, next_changed))
        required_no_feedback = not self.carry_stream
        passed = (primary_equal and reported_equal) if required_no_feedback else True
        return {
            "applicable": True,
            "required_no_feedback": required_no_feedback,
            "carry_stream": self.carry_stream,
            "initial_stream_intervention_l2": float(
                torch.linalg.vector_norm(initial_stream_delta).detach().cpu().item()
            ),
            "next_primary_max_abs_difference": float(
                primary_delta.detach().abs().max().cpu().item()
            ),
            "next_reported_max_abs_difference": float(
                reported_delta.detach().abs().max().cpu().item()
            ),
            "next_primary_bitwise_equal": primary_equal,
            "next_reported_bitwise_equal": reported_equal,
            "feeds_next_step_observed": not primary_equal,
            "overwritten_without_feedback": bool(required_no_feedback and reported_equal),
            "passed": bool(passed),
        }

    def state_spec(self) -> Dict[str, Any]:
        """Return the complete state definition as JSON-compatible metadata."""

        model_class = type(self.model).__name__
        model_module = type(self.model).__module__
        if self.is_full_block:
            components = [
                {
                    "name": "carrier",
                    "dimension": self.carrier_dim,
                    "reported_slice": [0, self.carrier_dim],
                    "in_primary_state": True,
                    "feeds_next_step": True,
                    "update": "recurrent",
                },
                {
                    "name": "stream",
                    "dimension": self.stream_dim,
                    "reported_slice": [self.carrier_dim, self.reported_dim],
                    "in_primary_state": self.stream_is_primary,
                    "feeds_next_step": self.carry_stream,
                    "update": (
                        "feedback_then_recomputed"
                        if self.carry_stream
                        else "overwritten_each_step"
                    ),
                },
            ]
            decode_mode = str(getattr(self.model, "decode_mode", "model_defined"))
            decoded_from = "stream" if decode_mode == "stream" else decode_mode
            overwritten = [] if self.carry_stream else ["stream"]
        else:
            components = [
                {
                    "name": "recurrent_state",
                    "dimension": self.reported_dim,
                    "reported_slice": [0, self.reported_dim],
                    "in_primary_state": True,
                    "feeds_next_step": True,
                    "update": "recurrent",
                }
            ]
            decoded_from = "model_defined_recurrent_state"
            overwritten = []

        return {
            "schema_version": 1,
            "adapter": "StateAdapter",
            "model_class": model_class,
            "model_module": model_module,
            "primary_state": "full_recurrent_markov_state",
            "primary_dimension": self.primary_dim,
            "reported_dimension": self.reported_dim,
            "carrier_dimension": self.carrier_dim,
            "stream_dimension": self.stream_dim,
            "components": components,
            "decoded_from": decoded_from,
            "decode_requires_reported_stream": bool(
                self.is_full_block
                and not self.stream_is_primary
                and str(getattr(self.model, "decode_mode", "")) == "stream"
            ),
            "overwritten_components": overwritten,
            "external_input_dimension": self.input_dim,
            "zero_input_definition": "literal_all_zero_tensor_passed_to_model.step",
            "legacy_reported_layout": (
                "carrier_then_stream" if self.is_full_block else "recurrent_state"
            ),
            "carry_stream": self.carry_stream,
        }


__all__ = ["StateAdapter", "StateParts"]
