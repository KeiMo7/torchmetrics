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
"""Observability events emitted during shard validation and merging.

Coverage alerts are deliberately *not* silent: every merge that proceeds below the configured coverage threshold
emits a :class:`LowCoverageEvent` on the merger's event log, in addition to the standard :mod:`warnings` machinery,
so callers can hook monitoring/alerting on the structured event.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShardEvent:
    """Base class for anything noteworthy that happens during a merge."""

    kind: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class LowCoverageEvent(ShardEvent):
    """Emitted when the merged coverage ratio is at or below the configured threshold.

    Attributes:
        coverage_ratio: fraction of the expected global samples that arrived shards.
        threshold: the threshold that was not met.
        missing_shard_ids: shard ids that were declared expected but never submitted.

    """

    coverage_ratio: float = 0.0
    threshold: float = 0.0
    missing_shard_ids: list[str] = field(default_factory=list)
    kind: str = "low_coverage"
    message: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Pin the event kind for structured downstream consumers."""
        self.kind = "low_coverage"
