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
"""Mergeable, fixed-memory weighted t-digest sketch implemented in pure PyTorch.

While fewer than ``buffer_size`` weighted observations have been seen, every raw sample is retained in a fixed-size
buffer and quantile queries return the *exact* weighted quantiles. Once the buffer is full, the observations are
compressed into at most ``compression`` weighted centroids on the arcsin scale (finest resolution in the tails) and the
raw buffer is released. From that point on the number of stored values depends only on the accuracy parameters, never
on the number of samples.

Every operation (ingestion, merging, quantile queries) is fully vectorized and deterministic, so merging is
associative and commutative. A data-dependent *rank error bound* is computed from the actual centroid weights and
therefore always holds for the current data, instead of being a nominal constant copied from documentation.
"""
from typing import Optional, Union

import torch
from torch import Tensor


class TDigest:
    """Fixed-size weighted t-digest sketch with an exact small-sample buffer.

    Args:
        compression: sketch resolution parameter ``delta``. At most ``delta`` centroids are stored after compression
            and every interior centroid spans at most ``pi / delta`` of the weighted rank range. Must be an integer
            ``>= 10``.
        buffer_size: number of raw (weighted) observations retained before the first compression. While the buffer is
            not full, quantile queries are exact. Must be an integer ``>= 1``.
        dtype: floating point dtype used by every internal tensor.
        device: device used by every internal tensor.

    """

    def __init__(
        self,
        compression: int,
        buffer_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(compression, int) or isinstance(compression, bool):
            raise ValueError(
                f"Arg `compression` must be a positive integer but got {compression!r} of type {type(compression)}."
            )
        if compression < 10:
            raise ValueError(
                f"Arg `compression` must be an integer >= 10 so that a finite rank error upper bound can be given,"
                f" but got {compression}."
            )
        if not isinstance(buffer_size, int) or isinstance(buffer_size, bool) or buffer_size < 1:
            raise ValueError(f"Arg `buffer_size` must be a positive integer but got {buffer_size!r}.")
        self.compression = compression
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.reset()

    @classmethod
    def bind(cls, compression: int, buffer_size: int, dtype: torch.dtype, device: torch.device) -> "TDigest":
        """Create an uninitialized sketch whose tensors the caller is going to attach afterwards."""
        self = cls.__new__(cls)
        self.compression = compression
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.device = torch.device(device)
        return self

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Reset the sketch to its empty state."""
        c, b = self.compression, self.buffer_size
        self.buffer_values = torch.zeros(b, dtype=self.dtype, device=self.device)
        self.buffer_weights = torch.zeros(b, dtype=self.dtype, device=self.device)
        self.buffer_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.centroid_mean = torch.zeros(c, dtype=self.dtype, device=self.device)
        self.centroid_weight = torch.zeros(c, dtype=self.dtype, device=self.device)
        self.centroid_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.total_weight = torch.zeros((), dtype=self.dtype, device=self.device)
        self.min_value = torch.zeros((), dtype=self.dtype, device=self.device)
        self.max_value = torch.zeros((), dtype=self.dtype, device=self.device)
        # whether every ingested observation so far carried unit weight (enables the torch.quantile exact path)
        self.unit_weights = torch.ones((), dtype=torch.bool, device=self.device)

    @property
    def is_empty(self) -> bool:
        """Return whether the sketch has accumulated any strictly positive weight."""
        return bool(self.total_weight.item() == 0.0)

    @property
    def is_exact(self) -> bool:
        """Return whether the sketch still holds every observation uncompressed."""
        return int(self.centroid_count.item()) == 0

    def clone(self) -> "TDigest":
        """Return an independent deep copy of the sketch."""
        new = TDigest(self.compression, self.buffer_size, self.dtype, self.device)
        for name in (
            "buffer_values",
            "buffer_weights",
            "buffer_count",
            "centroid_mean",
            "centroid_weight",
            "centroid_count",
            "total_weight",
            "min_value",
            "max_value",
            "unit_weights",
        ):
            setattr(new, name, getattr(self, name).clone())
        return new

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[Union[torch.device, str]] = None) -> "TDigest":
        """Move the sketch to another dtype and/or device and return ``self``."""
        self.dtype = self.dtype if dtype is None else dtype
        self.device = self.device if device is None else torch.device(device)
        for name in (
            "buffer_values",
            "buffer_weights",
            "centroid_mean",
            "centroid_weight",
            "total_weight",
            "min_value",
            "max_value",
        ):
            setattr(self, name, getattr(self, name).to(dtype=self.dtype, device=self.device))
        for name in ("buffer_count", "centroid_count"):
            setattr(self, name, getattr(self, name).to(device=self.device))
        for name in ("unit_weights",):
            setattr(self, name, getattr(self, name).to(device=self.device))
        return self

    # -------------------------------------------------------------- ingestion
    def update(self, values: Tensor, weights: Tensor) -> None:
        """Ingest one batch of finite flattened ``values`` with strictly positive flattened ``weights``."""
        if values.ndim != 1:
            values = values.reshape(-1)
        if weights.ndim != 1:
            weights = weights.reshape(-1)
        if values.numel() == 0:
            return
        if bool(torch.any(weights == 0)):
            keep = weights > 0
            values, weights = values[keep], weights[keep]
            if values.numel() == 0:
                return

        if self.is_empty:
            self.min_value.copy_(values.min())
            self.max_value.copy_(values.max())
        else:
            self.min_value.copy_(torch.minimum(self.min_value, values.min()))
            self.max_value.copy_(torch.maximum(self.max_value, values.max()))
        self.total_weight.add_(weights.sum())
        self.unit_weights.logical_and_(torch.all(weights == 1.0))

        start = int(self.buffer_count.item())
        end = start + values.numel()
        if end <= self.buffer_size and self.is_exact:
            # everything still fits into the raw buffer, stay exact
            self.buffer_values[start:end] = values
            self.buffer_weights[start:end] = weights
            self.buffer_count.fill_(end)
            return

        self._flush_with(values, weights)

    def _flush_with(self, values: Tensor, weights: Tensor) -> None:
        """Compress the current raw buffer (and centroids) together with a new batch."""
        k = int(self.centroid_count.item())
        b = int(self.buffer_count.item())
        all_values = [self.centroid_mean[:k], self.buffer_values[:b], values]
        all_weights = [self.centroid_weight[:k], self.buffer_weights[:b], weights]
        means, merged_weights = _recluster(
            torch.cat([v for v in all_values if v.numel()]),
            torch.cat([w for w in all_weights if w.numel()]),
            self.compression,
        )
        self._store_centroids(means, merged_weights)

    def _store_centroids(self, means: Tensor, weights: Tensor) -> None:
        n = means.numel()
        self.centroid_mean[:n] = means
        self.centroid_weight[:n] = weights
        if n < self.compression:
            self.centroid_mean[n:] = 0
            self.centroid_weight[n:] = 0
        self.centroid_count.fill_(n)
        self.buffer_values.zero_()
        self.buffer_weights.zero_()
        self.buffer_count.zero_()

    def merge(self, other: "TDigest") -> None:
        """Merge another sketch with identical accuracy parameters into this one.

        The operation is deterministic and order independent: merging the same multiset of sketches in any order
        produces bit-identical centroids, because reclustering depends only on the combined sorted
        ``(value, weight)`` multiset. Two still-exact sketches whose combined observations fit the buffer remain
        exact.
        """
        if other.compression != self.compression:
            raise ValueError(
                f"Cannot merge t-digest sketches with different `compression`: local compression is"
                f" {self.compression} but incoming compression is {other.compression}."
            )
        if other.buffer_size != self.buffer_size:
            raise ValueError(
                f"Cannot merge t-digest sketches with different `buffer_size`: local buffer_size is"
                f" {self.buffer_size} but incoming buffer_size is {other.buffer_size}."
            )
        if other.is_empty:
            return

        b_self, b_other = int(self.buffer_count.item()), int(other.buffer_count.item())
        k_self, k_other = int(self.centroid_count.item()), int(other.centroid_count.item())
        raw_count = b_self + b_other
        self.unit_weights.logical_and_(other.unit_weights)

        if self.is_empty:
            self.min_value.copy_(other.min_value)
            self.max_value.copy_(other.max_value)
        else:
            self.min_value.copy_(torch.minimum(self.min_value, other.min_value))
            self.max_value.copy_(torch.maximum(self.max_value, other.max_value))
        self.total_weight.add_(other.total_weight)

        if k_self == 0 and k_other == 0 and raw_count <= self.buffer_size:
            # both sides are still exact and the combined raw observations fit the buffer: stay exact
            self.buffer_values[b_self:raw_count] = other.buffer_values[:b_other]
            self.buffer_weights[b_self:raw_count] = other.buffer_weights[:b_other]
            self.buffer_count.fill_(raw_count)
            return

        values = torch.cat(
            [
                self.centroid_mean[:k_self],
                other.centroid_mean[:k_other],
                self.buffer_values[:b_self],
                other.buffer_values[:b_other],
            ]
        )
        weights = torch.cat(
            [
                self.centroid_weight[:k_self],
                other.centroid_weight[:k_other],
                self.buffer_weights[:b_self],
                other.buffer_weights[:b_other],
            ]
        )
        means, merged_weights = _recluster(values, weights, self.compression)
        self._store_centroids(means, merged_weights)

    # ---------------------------------------------------------------- queries
    def rank_error_bound(self) -> Tensor:
        """Return the guaranteed rank error bound as a fraction of the total weight.

        For every requested quantile ``q``, the weighted rank of the returned value differs from ``q`` by at most
        this fraction of the total weight. The bound is derived from the maximum *span* (weight represented by any
        single centroid) of the current sketch, so it is a hard, data-dependent guarantee. It is exactly zero while
        the sketch is still on the exact small-sample path.
        """
        if self.is_empty or self.is_exact:
            return torch.zeros((), dtype=self.dtype, device=self.device)
        k = int(self.centroid_count.item())
        return self.centroid_weight[:k].max() / self.total_weight

    def quantile(self, q: Tensor) -> Tensor:
        """Return weighted quantile estimates for the elements of ``q`` (all values must be in ``(0, 1)``)."""
        q = q.to(dtype=self.dtype, device=self.device).reshape(-1)
        if self.is_empty:
            raise RuntimeError("Quantile is undefined for an empty t-digest sketch (no positive-weight samples).")
        if self.is_exact:
            return self._exact_quantile(q)
        return self._sketch_quantile(q)

    def _exact_quantile(self, q: Tensor) -> Tensor:
        b = int(self.buffer_count.item())
        values = self.buffer_values[:b]
        weights = self.buffer_weights[:b]
        order = torch.argsort(values)
        values, weights = values[order], weights[order]
        if bool(self.unit_weights.item()):
            # match the reference type-7 (linear interpolation) definition of torch.quantile exactly
            return torch.quantile(values, q)
        total = weights.sum()
        cum = torch.cumsum(weights, dim=0)
        target = q * total
        idx = torch.searchsorted(cum, target, right=False).clamp(min=0, max=b - 1)
        prev = cum[idx] - weights[idx]
        frac = ((target - prev) / weights[idx]).clamp(min=0.0, max=1.0)
        prev_value = values[torch.clamp(idx - 1, min=0)]
        return prev_value + frac * (values[idx] - prev_value)

    def _sketch_quantile(self, q: Tensor) -> Tensor:
        k = int(self.centroid_count.item())
        means = self.centroid_mean[:k]
        weights = self.centroid_weight[:k]
        target = q * self.total_weight
        cum = torch.cumsum(weights, dim=0)
        # centroid i covers ranks in (cum[i-1], cum[i]]; return the mean of the first centroid reaching `target`
        idx = torch.searchsorted(cum, target, right=False).clamp(min=0, max=k - 1)
        return means[idx]


def _grid_boundaries(compression: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Return the fixed, non-uniform rank boundaries defining the ``compression - 2`` interior bins.

    Boundaries are placed on the inverse arcsin (``k1``) scale so that bins are narrow in the tails and wider in
    the middle: ``b_k = (1 - cos(pi * k / delta)) / 2`` for ``k = 1, ..., delta - 3``. They delimit ``delta - 2``
    interior bins, which together with the two forced singleton endpoint bins produce at most ``delta`` centroids.
    The boundaries depend only on ``compression`` and are identical on every process, which is what makes clustering
    order independent.
    """
    k = torch.arange(1, compression - 2, dtype=dtype, device=device)
    return 0.5 * (1.0 - torch.cos(torch.pi * k / compression))


def _recluster(values: Tensor, weights: Tensor, compression: int) -> tuple[Tensor, Tensor]:
    """Rebuild at most ``compression`` centroids from a sorted weighted multiset on a fixed non-uniform grid.

    Each weighted observation is assigned to a bin through :func:`_grid_boundaries`, evaluated at the normalized
    rank of its weighted midpoint, and all observations sharing a bin are averaged into one centroid. The global
    minimum and maximum are forced into unique singleton endpoint bins, so the total number of centroids never
    exceeds ``compression``.

    The bin of every observation depends only on its cumulative position relative to the *combined* sorted
    multiset - never on how the inputs were previously partitioned, batched or merged. Clustering is therefore a
    deterministic function of the combined multiset, which makes repeated merging associative and commutative
    (bit-identical centroids for the same multiset). The widest interior bin spans at most
    ``pi / (2 * delta)`` of the rank axis, and the rank error bound exposed by the sketch is computed from the
    actual centroid weights so it always holds for the current data.
    """
    order = torch.argsort(values)
    values, weights = values[order], weights[order]
    n = values.numel()
    if n <= 2:
        return values, weights

    boundaries = _grid_boundaries(compression, values.dtype, values.device)
    cum = torch.cumsum(weights, dim=0)
    total = cum[-1]
    # normalized rank of the weighted midpoint of every observation
    center_q = (cum - 0.5 * weights) / total
    bin_idx = torch.searchsorted(boundaries, center_q, right=False)
    # unique singleton endpoint bins for the extreme observations
    bin_idx[0] = -2
    bin_idx[-1] = -1

    _, inv = torch.unique_consecutive(bin_idx, return_inverse=True, return_counts=False)
    num_groups = int(inv.max().item()) + 1
    group_weight = torch.zeros(num_groups, dtype=weights.dtype, device=weights.device)
    group_weight.index_add_(0, inv, weights)
    weighted_sum = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    weighted_sum.index_add_(0, inv, values * weights)
    group_mean = weighted_sum / group_weight
    return group_mean, group_weight
