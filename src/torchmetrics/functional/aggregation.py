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
"""Functional interface for streaming mergeable quantile estimation."""
from typing import Optional, Union

import torch
from torch import Tensor

from torchmetrics.aggregation._tdigest import TDigest

__all__ = ["quantile", "quantile_rank_error_bound"]


def _build_sketch(
    values: Tensor,
    weights: Optional[Tensor],
    compression: int,
    buffer_size: int,
) -> TDigest:
    """Construct and ingest one batch into a t-digest sketch."""
    dtype = values.dtype if torch.is_floating_point(values) else torch.get_default_dtype()
    sketch = TDigest(compression, buffer_size, dtype=dtype, device=values.device)
    if weights is None:
        weights = torch.ones_like(values, dtype=dtype)
    sketch.update(values, weights)
    return sketch


def quantile(
    values: Tensor,
    quantiles: Union[float, list[float], tuple[float, ...], Tensor],
    weights: Optional[Tensor] = None,
    compression: int = 100,
    buffer_size: int = 500,
) -> Tensor:
    """Compute quantile estimates of ``values`` through a fixed-size t-digest sketch.

    The functional entry point mirrors :class:`~torchmetrics.aggregation.QuantileMetric`. When the number of
    observations does not exceed ``buffer_size`` the returned quantiles are exact; otherwise they are sketch
    estimates whose guaranteed rank error is available from :func:`quantile_rank_error_bound`.

    Args:
        values: scalar or tensor of arbitrary shape; all dimensions are flattened before ingestion.
        quantiles: a single float or sequence of floats, each strictly in the open interval ``(0, 1)``.
        weights: optional tensor of non-negative finite weights, broadcastable to the shape of ``values``.
        compression: t-digest resolution parameter ``delta`` (``>= 10``).
        buffer_size: raw observation capacity kept before compression (exact path).

    Returns:
        Tensor with one entry per requested quantile (a scalar tensor for a single quantile).

    Example:
        >>> import torch
        >>> from torchmetrics.functional.aggregation import quantile
        >>> quantile(torch.arange(11.0), quantiles=[0.25, 0.5, 0.75])
        tensor([2.5000, 5.0000, 7.5000])

    """
    from torchmetrics.aggregation.quantile import QuantileMetric

    q = QuantileMetric._validate_quantiles(quantiles).to(
        dtype=values.dtype if torch.is_floating_point(values) else torch.get_default_dtype(),
        device=values.device,
    )
    sketch = _build_sketch(values, weights, compression, buffer_size)
    result = sketch.quantile(q)
    return result if q.numel() > 1 else result.squeeze(0)


def quantile_rank_error_bound(
    values: Tensor,
    weights: Optional[Tensor] = None,
    compression: int = 100,
    buffer_size: int = 500,
) -> Tensor:
    """Return the guaranteed rank error bound of the t-digest that would be built from ``values``.

    The bound is zero while the batch fits the exact buffer, otherwise it equals the largest centroid span as a
    fraction of the total weight.
    """
    sketch = _build_sketch(values, weights, compression, buffer_size)
    return sketch.rank_error_bound()
