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
"""Sharded metric merging.

An opt-in toolkit for splitting a metric calculation over independent shards (processes, machines or jobs),
sealing each shard into a self-describing :class:`~torchmetrics.sharding.ShardEnvelope`, and merging them back
with explicit validation, reduction semantics and audit records.

Importing this package never changes the behavior of existing metrics: nothing in
:mod:`torchmetrics.metric` imports it, and single-process accumulation plus the distributed ``sync`` path are
untouched.

Example:
    >>> import torch
    >>> from torchmetrics.aggregation import MeanMetric
    >>> from torchmetrics.sharding import Coverage, ShardEnvelope, ShardMerger
    >>> data = torch.arange(100.0)
    >>> shards = []
    >>> for part in range(4):
    ...     metric = MeanMetric()
    ...     chunk = data[part * 25 : (part + 1) * 25]
    ...     metric.update(chunk)
    ...     shards.append(
    ...         ShardEnvelope.from_metric(
    ...             metric, Coverage(count=chunk.numel(), interval=(part * 25, (part + 1) * 25 - 1), global_count=100)
    ...         )
    ...     )
    >>> merged = MeanMetric()
    >>> merger = ShardMerger(merged, expected_total_count=100)
    >>> merger.merge(shards)  # doctest: +ELLIPSIS
    MeanMetric(...)
    >>> merged.compute()
    tensor(49.5000)

"""

from torchmetrics.sharding.envelope import (
    ENVELOPE_VERSION,
    Coverage,
    ShardEnvelope,
    canonical_json,
    config_fingerprint,
    content_checksum,
)
from torchmetrics.sharding.events import LowCoverageEvent, ShardEvent
from torchmetrics.sharding.merger import MergeManifest, ShardMerger, merge_shards
from torchmetrics.sharding.reducers import (
    CatReducer,
    ExtremaReducer,
    MeanReducer,
    Reducer,
    SumReducer,
    reducer_for,
)
from torchmetrics.sharding.validation import ValidationReport, validate_shards

__all__ = [
    "ENVELOPE_VERSION",
    "CatReducer",
    "Coverage",
    "ExtremaReducer",
    "LowCoverageEvent",
    "MeanReducer",
    "MergeManifest",
    "Reducer",
    "ShardEnvelope",
    "ShardEvent",
    "ShardMerger",
    "SumReducer",
    "ValidationReport",
    "canonical_json",
    "config_fingerprint",
    "content_checksum",
    "merge_shards",
    "reducer_for",
    "validate_shards",
]
