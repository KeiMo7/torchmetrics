# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streaming, mergeable and serializable quantile metric backed by a fixed-size t-digest sketch."""
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.aggregation._tdigest import TDigest
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.distributed import gather_all_tensors

__all__ = ["QuantileMetric"]

_SKETCH_FLOAT_STATES = (
    "buffer_values",
    "buffer_weights",
    "centroid_mean",
    "centroid_weight",
    "total_weight",
    "min_value",
    "max_value",
)
_SKETCH_INT_STATES = ("buffer_count", "centroid_count")
_SKETCH_BOOL_STATES = ("unit_weights",)
_SKETCH_STATES = _SKETCH_FLOAT_STATES + _SKETCH_INT_STATES + _SKETCH_BOOL_STATES


class QuantileMetric(Metric):
    r"""Estimate one or multiple quantiles of a weighted stream with constant, sample-count-independent memory.

    Instead of buffering every observed score, the metric keeps a mergeable weighted `t-digest`_ sketch of fixed size.
    The same single sketch serves every requested quantile, can be merged across processes in any order, and is fully
    serialized by the standard ``state_dict`` mechanism so computation can resume after a restart.

    While fewer than ``buffer_size`` weighted observations have been seen, every raw value is retained and the metric
    returns the **exact** quantile (linear interpolation, matching :func:`torch.quantile` for unit weights). Once the
    buffer is full, values are compressed into at most ``compression`` weighted centroids. A guaranteed, data-dependent
    rank error upper bound is available through :meth:`rank_error_bound`.

    As input to ``update`` and ``forward``:

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): scalar or tensor of arbitrary shape; all dimensions are
      flattened before ingestion.
    - ``weight`` (optional :class:`~float` or :class:`~torch.Tensor`): matching scalar or tensor of non-negative,
      finite weights. A scalar weight applies to every value.

    As output of ``compute``:

    - ``quantiles`` (:class:`~torch.Tensor`): tensor with one entry per requested quantile (a scalar tensor when a
      single quantile was requested).

    Args:
        quantiles: a single float or a sequence of floats, each strictly in the open interval ``(0, 1)``.
        compression: t-digest resolution parameter ``delta`` (``>= 10``). Larger values tighten the rank error bound
            at the cost of a larger fixed state. The structural bound is ``pi / delta`` of the weighted rank range.
        buffer_size: number of raw weighted observations retained before the first compression; quantiles are exact
            while the buffer is not full. Must be a positive integer.
        nan_strategy: options:

            - ``'error'``: if any ``nan`` values are encountered will give a ``RuntimeError``
            - ``'warn'``: if any ``nan`` values are encountered will give a warning and continue
            - ``'ignore'``: all ``nan`` values are silently removed
            - ``'disable'``: disable all ``nan`` checks
            - a float: impute any ``nan`` values with this value

        kwargs: additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError: if any requested quantile is not in the open interval ``(0, 1)``
        ValueError: if ``compression`` is smaller than ``10`` or ``buffer_size`` is not a positive integer
        ValueError: if ``nan_strategy`` is not one of the supported strategies
        ValueError: if a weight contains negative or non-finite values
        RuntimeError: if the metric is evaluated before any positive-weight sample was observed

    Example:
        >>> import torch
        >>> from torchmetrics.aggregation import QuantileMetric
        >>> metric = QuantileMetric(quantiles=[0.5, 0.95, 0.99])
        >>> metric.update(torch.randn(10_000))
        >>> values = metric.compute()
        >>> bound = metric.rank_error_bound()

    .. _t-digest: https://arxiv.org/abs/1902.04023

    """

    is_differentiable = False
    higher_is_better = None
    full_state_update: bool = False

    def __init__(
        self,
        quantiles: Union[float, list[float], tuple[float, ...], Tensor],
        compression: int = 100,
        buffer_size: int = 500,
        nan_strategy: Union[Literal["error", "warn", "ignore", "disable"], float] = "error",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        allowed_nan_strategy = ("error", "warn", "ignore", "disable")
        if nan_strategy not in allowed_nan_strategy and not isinstance(nan_strategy, float):
            raise ValueError(
                f"Arg `nan_strategy` should either be a float or one of {allowed_nan_strategy}"
                f" but got {nan_strategy!r}."
            )
        q_tensor = self._validate_quantiles(quantiles)
        if not isinstance(compression, int) or isinstance(compression, bool) or compression < 10:
            raise ValueError(
                f"Arg `compression` must be an integer >= 10 so that a finite rank error upper bound can be given,"
                f" but got {compression!r}."
            )
        if not isinstance(buffer_size, int) or isinstance(buffer_size, bool) or buffer_size < 1:
            raise ValueError(f"Arg `buffer_size` must be a positive integer but got {buffer_size!r}.")

        self.nan_strategy = nan_strategy
        self.compression = compression
        self.buffer_size = buffer_size
        # requested quantiles are configuration, not accumulated state; keep them at full float64 precision so
        # that an exact-path query is bit-identical to torch.quantile regardless of the metric's working dtype
        self.register_buffer("quantile_points", q_tensor.to(dtype=torch.float64, device=self.device))

        self._register_sketch_states()
        # the sketch operates directly on the registered state tensors: single source of truth, so reset(),
        # state_dict load and distributed sync stay consistent without copying
        self._sketch = TDigest.bind(compression, buffer_size, dtype=self.dtype, device=self.device)
        self._bind_sketch()

    @staticmethod
    def _validate_quantiles(quantiles: Union[float, list[float], tuple[float, ...], Tensor]) -> Tensor:
        """Validate and normalize the requested quantiles into a 1D float tensor."""
        # validate and keep full float64 precision; the buffer is cast to the metric dtype afterwards
        if isinstance(quantiles, (int, float)) and not isinstance(quantiles, bool):
            q = torch.tensor([float(quantiles)], dtype=torch.float64)
        elif isinstance(quantiles, Tensor):
            q = quantiles.detach().reshape(-1).to(torch.float64)
        else:
            try:
                q = torch.tensor([float(v) for v in quantiles], dtype=torch.float64)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Arg `quantiles` must contain floats in the open interval (0, 1), but got {quantiles!r}."
                ) from err
        if q.numel() == 0:
            raise ValueError("Arg `quantiles` must contain at least one quantile, but got an empty sequence.")
        if torch.any(q <= 0.0) or torch.any(q >= 1.0) or torch.any(~torch.isfinite(q)):
            raise ValueError(
                f"Every element of arg `quantiles` must be strictly in the open interval (0, 1), but got"
                f" {quantiles!r}."
            )
        return q

    def _register_sketch_states(self) -> None:
        """Register the fixed-size sketch tensors as native, persistent metric states."""
        c, b, dtype, device = self.compression, self.buffer_size, self.dtype, self.device

        def zeros_float(shape: tuple) -> Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        def zeros_long(shape: tuple) -> Tensor:
            return torch.zeros(shape, dtype=torch.long, device=device)

        defaults = {
            "buffer_values": zeros_float(b),
            "buffer_weights": zeros_float(b),
            "centroid_mean": zeros_float(c),
            "centroid_weight": zeros_float(c),
            "total_weight": zeros_float(()),
            "min_value": zeros_float(()),
            "max_value": zeros_float(()),
            "buffer_count": zeros_long(()),
            "centroid_count": zeros_long(()),
            "unit_weights": torch.ones((), dtype=torch.bool, device=device),
        }
        for name, default in defaults.items():
            self.add_state(name, default=default, dist_reduce_fx=None, persistent=True)

    def _bind_sketch(self) -> None:
        """Point the sketch object at the registered state tensors."""
        for name in _SKETCH_STATES:
            setattr(self._sketch, name, getattr(self, name))
        self._sketch.dtype = self.dtype
        self._sketch.device = self.device

    def _apply(self, fn: Callable, exclude_state: Sequence[str] = "") -> Metric:
        """Re-bind the sketch after the base class moves or casts the registered state tensors."""
        result = super()._apply(fn, exclude_state)
        self._bind_sketch()
        return result

    # ---------------------------------------------------------------- update
    def update(
        self,
        value: Union[float, Tensor],
        weight: Optional[Union[float, Tensor]] = None,
    ) -> None:
        """Accumulate weighted observations; scalars and arbitrary-shaped tensors are both supported."""
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        value = value.reshape(-1).to(dtype=self.dtype, device=self.device)
        if value.numel() == 0:
            return

        if weight is None:
            # fast path: unit weights, no broadcasting or weight validation needed
            value = self._handle_nan_values(value)
            if value.numel():
                self._sketch.update(value, torch.ones_like(value))
            return

        if not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = torch.broadcast_to(weight, value.shape).reshape(-1).to(dtype=self.dtype, device=self.device)
        self._validate_weights(weight)
        value, weight = self._apply_nan_strategy(value, weight)
        if value.numel():
            self._sketch.update(value, weight)

    def _handle_nan_values(self, value: Tensor) -> Tensor:
        """Apply the configured NaN strategy on the values of the unit-weight fast path."""
        if self.nan_strategy == "disable":
            return value
        if not torch.isnan(value).any():
            return value
        if self.nan_strategy == "error":
            raise RuntimeError("Encountered `nan` values in tensor")
        if self.nan_strategy in ("ignore", "warn"):
            if self.nan_strategy == "warn":
                rank_zero_warn("Encountered `nan` values in tensor. Will be removed.", UserWarning)
            return value[~torch.isnan(value)]
        if isinstance(self.nan_strategy, float):
            value = value.clone()
            value[torch.isnan(value)] = self.nan_strategy
            return value
        raise ValueError(f"Unknown nan_strategy: {self.nan_strategy!r}")  # pragma: no cover

    @staticmethod
    def _validate_weights(weight: Tensor) -> None:
        """Give parameter-specific errors for negative or non-finite weights."""
        if torch.any(~torch.isfinite(weight)):
            raise ValueError("Arg `weight` must contain only finite values, but got non-finite weights.")
        if torch.any(weight < 0):
            raise ValueError("Arg `weight` must be non-negative, but got negative weights.")

    def _apply_nan_strategy(self, value: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
        """Handle NaNs in values and weights following the configured strategy."""
        if self.nan_strategy == "disable":
            return value, weight
        invalid = torch.isnan(value) | torch.isnan(weight)
        if not invalid.any():
            return value, weight
        if self.nan_strategy == "error":
            raise RuntimeError("Encountered `nan` values in tensor")
        if self.nan_strategy in ("ignore", "warn"):
            if self.nan_strategy == "warn":
                rank_zero_warn("Encountered `nan` values in tensor. Will be removed.", UserWarning)
            keep = ~invalid
            return value[keep], weight[keep]
        if isinstance(self.nan_strategy, float):
            value = value.clone()
            value[invalid] = self.nan_strategy
            weight = weight.clone()
            weight[invalid] = 1.0
            return value, weight
        raise ValueError(f"Unknown nan_strategy: {self.nan_strategy!r}")  # pragma: no cover

    # --------------------------------------------------------------- compute
    def compute(self) -> Tensor:
        """Return the quantile estimates; raises when no positive-weight sample has been observed."""
        if self._sketch.is_empty:
            raise RuntimeError(
                "Quantile is undefined: `QuantileMetric` has not received any positive-weight sample yet."
            )
        result = self._sketch.quantile(self.quantile_points)
        return result if self.quantile_points.numel() > 1 else result.squeeze(0)

    def rank_error_bound(self) -> Tensor:
        """Return the guaranteed rank error upper bound as a fraction in ``[0, 1]``.

        Zero is returned on the exact small-sample path. Otherwise every returned quantile value has a true weighted
        rank within this distance of its requested quantile. The bound is computed from the largest centroid span of
        the current sketch and therefore always holds for the current data and accuracy parameters.
        """
        return self._sketch.rank_error_bound()

    def reset(self) -> None:
        """Reset all metric states and rebuild an empty sketch bound to those fresh states."""
        super().reset()
        self._sketch = TDigest.bind(
            self.compression, self.buffer_size, dtype=self.dtype, device=self.device
        )
        self._bind_sketch()

    # ---------------------------------------------------------------- merge
    def merge_state(self, incoming_state: Union[dict[str, Any], Metric]) -> None:
        """Merge another ``QuantileMetric`` state, rejecting mismatched accuracy parameters explicitly."""
        if isinstance(incoming_state, Metric):
            if not isinstance(incoming_state, QuantileMetric):
                raise ValueError(
                    f"Expected incoming state to be an instance of QuantileMetric but got"
                    f" {type(incoming_state).__name__}."
                )
            self._check_compatibility(incoming_state)
            incoming_state = incoming_state.metric_state
        elif not isinstance(incoming_state, dict):
            raise ValueError(
                f"Expected incoming state to be a dict or an instance of Metric but got {type(incoming_state)}."
            )
        self._reduce_states(incoming_state)

    def _reduce_states(self, incoming_state: dict[str, Any]) -> None:
        """Merge a raw state dict (used by the Running wrapper) through the sketch merge instead of stacking."""
        self._check_state_compatibility(incoming_state)
        other = TDigest.bind(self.compression, self.buffer_size, dtype=self.dtype, device=self.device)
        for name in _SKETCH_STATES:
            loaded = incoming_state[name]
            if name in _SKETCH_INT_STATES:
                loaded = loaded.to(device=self.device, dtype=torch.long)
            elif name in _SKETCH_BOOL_STATES:
                loaded = loaded.to(device=self.device, dtype=torch.bool)
            else:
                loaded = loaded.to(device=self.device, dtype=self.dtype)
            setattr(other, name, loaded)
        self._sketch.merge(other)

    def _check_compatibility(self, other: "QuantileMetric") -> None:
        if other.compression != self.compression:
            raise ValueError(
                f"Cannot merge QuantileMetric states with different `compression`: local is {self.compression} but"
                f" incoming is {other.compression}. Recreate both metrics with identical accuracy parameters."
            )
        if other.buffer_size != self.buffer_size:
            raise ValueError(
                f"Cannot merge QuantileMetric states with different `buffer_size`: local is {self.buffer_size} but"
                f" incoming is {other.buffer_size}. Recreate both metrics with identical accuracy parameters."
            )

    def _check_state_compatibility(self, incoming_state: dict[str, Any]) -> None:
        missing = set(self._defaults) - set(incoming_state)
        if missing:
            raise ValueError(f"Expected state variables {sorted(missing)} to be present in incoming state.")
        # accuracy parameters are encoded in the fixed state shapes
        if incoming_state["centroid_mean"].shape != self.centroid_mean.shape:
            raise ValueError(
                f"Cannot merge QuantileMetric states with different `compression`: local centroid capacity is"
                f" {self.centroid_mean.shape[0]} but incoming is {incoming_state['centroid_mean'].shape[0]}."
            )
        if incoming_state["buffer_values"].shape != self.buffer_values.shape:
            raise ValueError(
                f"Cannot merge QuantileMetric states with different `buffer_size`: local buffer capacity is"
                f" {self.buffer_values.shape[0]} but incoming is {incoming_state['buffer_values'].shape[0]}."
            )

    # ------------------------------------------------------------ distributed
    def _sync_dist(self, dist_sync_fn: Any = gather_all_tensors, process_group: Optional[Any] = None) -> None:
        """Gather the fixed-size states of every rank and merge them, keeping the single-sketch invariant."""
        input_dict = {attr: getattr(self, attr) for attr in self._defaults}
        gathered = {
            attr: dist_sync_fn(tensor, group=process_group or self.process_group)
            for attr, tensor in input_dict.items()
        }
        world_size = len(next(iter(gathered.values())))

        merged = TDigest(self.compression, self.buffer_size, dtype=self.dtype, device=self.device)
        for rank in range(world_size):
            rank_sketch = TDigest.bind(
                self.compression, self.buffer_size, dtype=self.dtype, device=self.device
            )
            for name in _SKETCH_STATES:
                value = gathered[name][rank]
                if name in _SKETCH_INT_STATES:
                    value = value.to(torch.long)
                elif name in _SKETCH_BOOL_STATES:
                    value = value.to(torch.bool)
                setattr(rank_sketch, name, value)
            merged.merge(rank_sketch)

        for name in _SKETCH_STATES:
            value = getattr(merged, name)
            setattr(self, name, value)
        self._bind_sketch()

    def unsync(self, should_unsync: bool = True) -> None:
        """Re-bind the sketch after the base class restores the cached local states."""
        super().unsync(should_unsync)
        if should_unsync:
            self._bind_sketch()

    # ------------------------------------------------------------- persistence
    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Load states with an explicit device/dtype guard instead of silently producing garbage."""
        for key in list(self._defaults):
            name = prefix + key
            if name not in state_dict:
                continue
            loaded = state_dict.pop(name)
            if loaded.shape != getattr(self, key).shape:
                raise ValueError(
                    f"Cannot restore QuantileMetric state `{key}`: archived shape {tuple(loaded.shape)} does not"
                    f" match the current metric shape {tuple(getattr(self, key).shape)}. The archived metric was"
                    f" likely created with different `compression`/`buffer_size` parameters."
                )
            target = getattr(self, key)
            if key in _SKETCH_INT_STATES:
                loaded = loaded.to(target.dtype)
            elif key in _SKETCH_BOOL_STATES:
                loaded = loaded.to(torch.bool)
            elif loaded.dtype != target.dtype:
                # safe conversion: float64 -> float32 is well defined but may lose precision; float16 is refused
                if loaded.dtype == torch.float16 and target.dtype in (torch.float32, torch.float64):
                    raise TypeError(
                        f"Refusing to load float16 QuantileMetric state `{key}` into a {target.dtype} metric:"
                        " float16 checkpoints are not numerically safe for quantile bookkeeping. Recreate the"
                        " metric with dtype=torch.float16 instead."
                    )
                rank_zero_warn(
                    f"Restoring QuantileMetric state `{key}` from {loaded.dtype} to {target.dtype};"
                    " values are safely converted but precision may change.",
                    UserWarning,
                )
                loaded = loaded.to(target.dtype)
            setattr(self, key, loaded.to(target.device))
        super(Metric, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        self._bind_sketch()
