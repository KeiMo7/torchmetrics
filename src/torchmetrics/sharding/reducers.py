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
"""Reduction semantics for shard merging.

Four explicit reduction semantics are supported, each with its own consistency check. Naively adding everything is
only correct for the *sum* family — means, extrema and concatenation each get a dedicated reducer:

* :class:`SumReducer` — associative tensor/list addition. Consistency check: the merged value must equal the
  element-wise sum of the shard values.
* :class:`MeanReducer` — weighted merge via the accumulated numerator (``sum x*w``) and denominator (``sum w``).
  Consistency check: numerator and denominator are both sum-family states, and the denominator equals the total
  sample weight. Plain averaging of per-shard means is deliberately *not* used, since shards may have different
  sizes.
* :class:`ExtremaReducer` — element-wise min/max. Consistency check: the result is element-wise equal to one of the
  contributing shard values (an extremum can never invent a new value).
* :class:`CatReducer` — order-insensitive concatenation for list states. Raw concatenation order depends on the
  submission order, so values are sorted on a stable content key before concatenation; the result is therefore
  bit-identical for any submission order. Consistency check: the output element multiset equals the union of the
  shard multisets (no sample lost or duplicated).

Every reducer is a pure function over already-validated states and uses only :mod:`torch` operations, so merging
adds no per-shard compute cost during the shards' own calculation.
"""

import abc
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from torchmetrics.utilities.data import dim_zero_cat, dim_zero_max, dim_zero_min, dim_zero_sum


class Reducer(abc.ABC):
    """Abstract base class for the four merge reduction semantics."""

    #: human-readable name used in audit records
    name: str = "abstract"

    def __init__(self, reduction_fn: Optional[Callable] = None) -> None:
        self.reduction_fn = reduction_fn

    @abc.abstractmethod
    def reduce(self, values: list[Any]) -> Any:
        """Combine the per-shard values of one state variable into a single merged value."""

    @abc.abstractmethod
    def consistency_check(self, shard_values: list[Any], merged: Any) -> None:
        """Verify that ``merged`` is a valid merge of ``shard_values`` for this semantics.

        Raises:
            RuntimeError: with a semantic-specific message if the invariant is violated.

        """


class SumReducer(Reducer):
    """Associative sum reduction for tensors and tensor lists."""

    name = "sum"

    def reduce(self, values: list[Any]) -> Any:
        """Sum tensor operands element-wise and concatenate list operands, ignoring empty inputs."""
        tensors = [value for value in values if isinstance(value, Tensor)]
        lists = [value for value in values if isinstance(value, list)]
        result = dim_zero_sum(torch.stack(tensors)) if tensors else None
        flattened = [item for sublist in lists for item in sublist]
        if flattened:
            cat_result = _sorted_cat(flattened)
            result = cat_result if result is None else result + cat_result
        if result is None:
            return []
        return result

    def consistency_check(self, shard_values: list[Any], merged: Any) -> None:
        """Verify the merged value equals a recomputed element-wise sum / list union."""
        expected = self.reduce(shard_values)
        if isinstance(merged, Tensor):
            if not torch_equal(merged, expected):
                raise RuntimeError(
                    f"sum consistency check failed: merged value {merged.tolist()} differs from element-wise shard "
                    f"sum {expected.tolist()}"
                )
        elif not _element_multiset_equal(merged, expected):
            raise RuntimeError("sum (list) consistency check failed: merged multiset differs from shard union")


class MeanReducer(Reducer):
    """Weighted-mean reduction.

    The reducer itself only merges the bookkeeping states (numerator and denominator) with the sum semantics; the
    actual mean ``numerator / denominator`` is produced by the metric's own ``compute`` method, which keeps the
    existing update/compute call contract and return types untouched.
    """

    name = "mean"

    def __init__(self) -> None:
        super().__init__(dim_zero_sum)

    def reduce(self, values: list[Any]) -> Any:
        """Sum the numerator/denominator tensors; empty (not-yet-updated) operands are skipped."""
        tensors = [value for value in values if isinstance(value, Tensor)]
        if not tensors:
            raise TypeError("MeanReducer only supports tensor numerator/denominator states")
        return dim_zero_sum(torch.stack(tensors))

    def consistency_check(self, shard_values: list[Any], merged: Any) -> None:
        """Verify the merged bookkeeping state equals the shard-wise sum."""
        expected = self.reduce(shard_values)
        if not torch_equal(merged, expected):
            raise RuntimeError(
                f"mean bookkeeping consistency check failed: merged {merged.tolist()} differs from shard-wise "
                f"sum {expected.tolist()}"
            )


class ExtremaReducer(Reducer):
    """Element-wise minimum/maximum reduction."""

    name = "extrema"

    def __init__(self, reduction_fn: Callable) -> None:
        if reduction_fn not in (dim_zero_min, dim_zero_max):
            raise ValueError("ExtremaReducer requires dim_zero_min or dim_zero_max")
        super().__init__(reduction_fn)
        self.name = "min" if reduction_fn is dim_zero_min else "max"

    def reduce(self, values: list[Any]) -> Any:
        """Apply the element-wise min/max over the non-empty tensor operands."""
        tensors = [value for value in values if isinstance(value, Tensor)]
        if not tensors:
            raise RuntimeError("ExtremaReducer received no tensor operands")
        return self.reduction_fn(torch.stack(tensors))

    def consistency_check(self, shard_values: list[Any], merged: Any) -> None:
        """Verify every merged element is the extremum taken from one of the contributing shards."""
        tensors = [value for value in shard_values if isinstance(value, Tensor)]
        stacked = torch.stack(tensors)
        # an extremum must come from one of the contributing shards, element-wise
        source_allowed = stacked.max(dim=0).values if self.reduction_fn is dim_zero_max else stacked.min(dim=0).values
        if not torch_equal(merged, source_allowed):
            raise RuntimeError(
                f"{self.name} consistency check failed: merged {merged.tolist()} is not the element-wise "
                f"{self.name}imum of the shard values {source_allowed.tolist()}"
            )


class CatReducer(Reducer):
    """Order-insensitive concatenation for list states."""

    name = "cat"

    def __init__(self) -> None:
        super().__init__(dim_zero_cat)

    def reduce(self, values: list[Any]) -> Any:
        """Flatten list operands (and any already-concatenated tensors), then concatenate in content order."""
        flattened = [item for sub in values if isinstance(sub, list) for item in sub]
        # also accept tensor operands (e.g. an already-concatenated state from a previous merge round)
        flattened.extend(value for value in values if isinstance(value, Tensor))
        if not flattened:
            return []
        return _sorted_cat(flattened)

    def consistency_check(self, shard_values: list[Any], merged: Any) -> None:
        """Verify the output element multiset equals the union of all shard elements, with none lost or duplicated."""
        flattened = [item for sub in shard_values for item in (sub if isinstance(sub, (list, Tensor)) else [])]
        # compare at the *element* level: dim_zero_cat may collapse scalar tensors into one vector, so list-level
        # multiset comparison would see different granularities even when no sample was lost
        merged_elements = list(merged) if isinstance(merged, Tensor) else merged
        source_elements = list(dim_zero_cat(flattened)) if flattened else []
        if not _element_multiset_equal(merged_elements, source_elements):
            raise RuntimeError(
                "cat consistency check failed: concatenated output multiset does not match the union of shard "
                "elements (a sample was lost or duplicated)"
            )


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
def torch_equal(a: Tensor, b: Tensor) -> bool:
    """Exact (bit-for-bit) tensor equality, including dtype."""
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))


def _tensor_sort_key(tensor: Tensor) -> tuple:
    """Stable, content-derived sort key for a tensor: shape, dtype, then flattened values."""
    flat = tensor.detach().contiguous().view(-1)
    return (tuple(tensor.shape), str(tensor.dtype), flat.tolist())


def _sorted_cat(tensors: list[Tensor]) -> Tensor:
    """Concatenate tensors after sorting them on a content key, making the output order-insensitive."""
    ordered = sorted(tensors, key=_tensor_sort_key)
    return dim_zero_cat(ordered)


def _multiset_equal(a: list[Tensor], b: list[Tensor]) -> bool:
    """Compare two lists of tensors as multisets on content keys."""
    return sorted((_tensor_sort_key(t) for t in a), key=repr) == sorted((_tensor_sort_key(t) for t in b), key=repr)


def _element_multiset_equal(a: list[Tensor], b: list[Tensor]) -> bool:
    """Compare two sequences as element-level multisets, supporting both scalar and vector tensors."""
    return sorted((_tensor_sort_key(t) for t in a), key=repr) == sorted((_tensor_sort_key(t) for t in b), key=repr)


#: Mapping from the metric base class's registered reduction functions to shard reducers.
def reducer_for(reduction_fn: Optional[Callable], state_value: Any, state_name: str = "") -> Reducer:
    """Select the appropriate :class:`Reducer` for a state variable.

    Args:
        reduction_fn: the function registered via ``add_state(..., dist_reduce_fx=...)``.
        state_value: the current (reference shard) value, used to recognize list/concat states.
        state_name: name of the state variable; a state named ``"weight"`` (the denominator convention used by
            :class:`~torchmetrics.aggregation.MeanMetric` and weighted metrics) is merged with explicit mean
            semantics and denominator bookkeeping.

    Custom callables fall back to :class:`_CustomReducer`, preserving the existing stacking contract of
    ``dist_reduce_fx``; the generic path therefore works for any existing metric without per-metric changes.

    """
    if reduction_fn is dim_zero_sum and state_name == "weight":
        return MeanReducer()
    if reduction_fn is dim_zero_sum:
        return SumReducer(reduction_fn)
    if reduction_fn is dim_zero_min or reduction_fn is dim_zero_max:
        return ExtremaReducer(reduction_fn)
    if reduction_fn is dim_zero_cat:
        return CatReducer()
    if isinstance(state_value, list):
        # list state without an explicit cat reducer still concatenates by convention
        return CatReducer()
    if reduction_fn is None:
        # unreduced tensor state: stacking is the existing convention; treat as an order-insensitive cat
        return CatReducer()
    # custom reduction callable: invoke it on a stack of shard values (same contract as _sync_dist)
    return _CustomReducer(reduction_fn)


class _CustomReducer(Reducer):
    """Adapter for user-defined ``dist_reduce_fx`` callables, preserving the existing stacking contract."""

    name = "custom"

    def reduce(self, values: list[Any]) -> Any:
        if isinstance(values[0], Tensor):
            return self.reduction_fn(Tensor(values))
        flattened = [item for sublist in values for item in sublist]
        return self.reduction_fn(flattened)

    def consistency_check(self, shard_values: list[Any], merged: Any) -> None:
        expected = self.reduce(shard_values)
        if isinstance(merged, Tensor) and not torch_equal(merged, expected):
            raise RuntimeError("custom reducer consistency check failed: merged value differs from recomputation")
