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
"""Shard merger: validated, order-insensitive and idempotent merging of shard envelopes.

The merger is deliberately separate from the :class:`~torchmetrics.metric.Metric` base class, so importing or using
metrics without it leaves every existing single-process accumulation and distributed synchronization path byte for
byte unchanged. When the feature is not used nothing in this module is even imported by the metric base class.

Guarantees implemented here:

* **order insensitivity** — shards are deduplicated and sorted on content before reduction, and every reducer is
  itself order-insensitive, so any submission order yields bit-identical output;
* **idempotency** — resubmitting the same shard (same id, or same content checksum under another id) is a no-op;
* **atomicity** — all validation and reduction happens against a scratch metric; the caller's target metric (and any
  output file) is only touched after every check passes, so a mid-merge failure never leaves a half-merged result;
* **partial-coverage policy** — missing shards either reject the merge or produce an explicitly annotated result
  with a :class:`~torchmetrics.sharding.events.LowCoverageEvent`;
* **auditability** — every merge returns a :class:`MergeManifest` listing the contributing shards, their counts and
  the merged total, and merged envelopes keep full lineage so they can re-enter another merge round.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union

from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.sharding.envelope import ShardEnvelope, config_fingerprint
from torchmetrics.sharding.events import LowCoverageEvent, ShardEvent
from torchmetrics.sharding.reducers import Reducer, reducer_for
from torchmetrics.sharding.validation import ValidationReport, validate_shards
from torchmetrics.utilities.exceptions import TorchMetricsUserError

#: Policy when expected shards are missing.
MISSING_POLICIES = ("reject", "partial")


def _assign_state(metric: Metric, state: dict[str, Any]) -> None:
    """Assign already-reduced state to a metric, honoring tensor vs list state kinds.

    Tensors are set directly (they already carry the merged values); list states replace the (empty, post-reset)
    list contents.
    """
    for name, value in state.items():
        if name not in metric._defaults:
            raise TorchMetricsUserError(f"cannot assign unknown state variable {name!r} to {type(metric).__name__}")
        default = metric._defaults[name]
        if isinstance(default, list):
            # list states may legitimately end up as a concatenated tensor after a cat reduction; metrics that
            # register list states (e.g. CatMetric) accept both forms in compute()
            if isinstance(value, list):
                getattr(metric, name).clear()
                getattr(metric, name).extend(value)
            elif isinstance(value, Tensor):
                setattr(metric, name, value)
            else:
                raise TorchMetricsUserError(
                    f"state variable {name!r} is registered as a list but reducer returned {type(value).__name__}"
                )
        else:
            if not isinstance(value, Tensor):
                raise TorchMetricsUserError(
                    f"state variable {name!r} is registered as a tensor but reducer returned {type(value).__name__}"
                )
            setattr(metric, name, value)


@dataclass
class MergeManifest:
    """Reviewable record of one merge operation.

    Attributes:
        metric_class: fully qualified metric class name that was merged.
        shard_records: per-shard ``(shard_id, sample_count, checksum)`` rows in the canonical merge order.
        total_count: summed sample count across accepted shards.
        expected_total_count: expected global sample count, if known.
        coverage_ratio: ``total_count / expected_total_count`` when the denominator is known.
        duplicate_shard_ids: shard submissions that were ignored as duplicates.
        missing_shard_ids: expected shard ids that never arrived.
        reduction_by_state: which reducer semantics was applied to each state variable.
        lineage: flattened ancestry of every accepted shard, carried forward into the merged envelope.
        consistency_checks: results of the per-semantics consistency checks.
        events: structured events emitted during the merge (e.g. low-coverage alerts).

    """

    metric_class: str
    shard_records: list[dict[str, Any]]
    total_count: int
    expected_total_count: Optional[int]
    coverage_ratio: Optional[float]
    duplicate_shard_ids: list[str]
    missing_shard_ids: list[str]
    reduction_by_state: dict[str, str]
    lineage: list[str]
    consistency_checks: dict[str, str] = field(default_factory=dict)
    events: list[ShardEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation of the manifest."""
        return {
            "metric_class": self.metric_class,
            "shards": self.shard_records,
            "total_count": self.total_count,
            "expected_total_count": self.expected_total_count,
            "coverage_ratio": self.coverage_ratio,
            "duplicate_shard_ids": self.duplicate_shard_ids,
            "missing_shard_ids": self.missing_shard_ids,
            "reductions": self.reduction_by_state,
            "lineage": self.lineage,
            "consistency_checks": self.consistency_checks,
            "events": [
                {"kind": event.kind, "message": event.message, "context": event.context} for event in self.events
            ],
        }


class ShardMerger:
    """Merge shard envelopes into a target metric under an explicit, auditable policy.

    Args:
        target_metric: an *empty* metric of the same class/configuration that produced the shards. Its state is only
            mutated after all validation passes.
        expected_shard_ids: optional fixed list of shard ids the job expects (e.g. the four partition ids). Enables
            missing-shard detection.
        expected_total_count: optional global sample count. Enables coverage accounting and count mismatch errors.
        missing_policy: ``"reject"`` (default) refuses to merge when expected shards are absent; ``"partial"`` merges
            what arrived and annotates the result with the coverage ratio.
        min_coverage: coverage ratio at or below which a :class:`LowCoverageEvent` is emitted (default 1.0, i.e. any
            missing sample alerts). Only relevant for ``"partial"`` policy.
        event_handler: optional callback invoked with every :class:`ShardEvent`, for wiring monitoring/alerting.

    """

    def __init__(
        self,
        target_metric: Metric,
        *,
        expected_shard_ids: Optional[list[str]] = None,
        expected_total_count: Optional[int] = None,
        missing_policy: str = "reject",
        min_coverage: float = 1.0,
        event_handler: Optional[Callable[[ShardEvent], None]] = None,
    ) -> None:
        if missing_policy not in MISSING_POLICIES:
            raise ValueError(f"missing_policy must be one of {MISSING_POLICIES}, got {missing_policy!r}")
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError(f"min_coverage must be within [0, 1], got {min_coverage}")
        self.target_metric = target_metric
        self.expected_shard_ids = list(expected_shard_ids) if expected_shard_ids is not None else None
        self.expected_total_count = expected_total_count
        self.missing_policy = missing_policy
        self.min_coverage = min_coverage
        self.event_handler = event_handler
        self.events: list[ShardEvent] = []
        self.last_manifest: Optional[MergeManifest] = None
        # provenance of everything already merged into the target metric, so repeated merge() calls stay idempotent
        self._consumed_shard_ids: set[str] = set()
        self._consumed_checksums: set[str] = set()

    # ------------------------------------------------------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------------------------------------------------------
    def merge(
        self,
        shards: List[Union[ShardEnvelope, dict[str, Any]]],
        *,
        extra_config: Optional[dict[str, Any]] = None,
    ) -> Metric:
        """Validate and merge shards into the target metric.

        Args:
            shards: envelopes (or their dict/JSON-decoded form) in any order. Duplicate submissions are ignored.
            extra_config: extra configuration values bound into the fingerprint check against the target metric.

        Returns:
            The target metric with merged state. Its ``update``/``compute`` API and return types are unchanged.

        Raises:
            TorchMetricsUserError: on any hard validation failure, configuration mismatch, or (under
                ``missing_policy="reject"``) missing shards. The target metric is left untouched.

        """
        envelopes = [shard if isinstance(shard, ShardEnvelope) else ShardEnvelope.from_dict(shard) for shard in shards]
        # idempotency across merge() calls: anything this merger has already folded in is dropped before
        # validation, so resubmitting a shard never changes the result
        fresh_envelopes: list[ShardEnvelope] = []
        redelivered: list[str] = []
        for envelope in envelopes:
            if envelope.shard_id in self._consumed_shard_ids or envelope.checksum in self._consumed_checksums:
                redelivered.append(envelope.shard_id)
                continue
            fresh_envelopes.append(envelope)
        if not fresh_envelopes:
            # pure resubmission: the target state must not change, but the audit trail still records the
            # redelivered shard ids
            if self.last_manifest is not None and redelivered:
                self.last_manifest.duplicate_shard_ids.extend(redelivered)
            return self.target_metric
        report = validate_shards(
            fresh_envelopes,
            expected_shard_ids=self.expected_shard_ids,
            expected_total_count=self.expected_total_count,
        )
        report.duplicate_shard_ids.extend(redelivered)
        # fingerprint of the live target must agree with every shard
        if envelopes:
            target_fp = config_fingerprint(self.target_metric, extra_config=extra_config)
            shard_fps = {envelope.config_fingerprint for envelope in envelopes}
            if shard_fps != {target_fp}:
                raise TorchMetricsUserError(
                    "configuration fingerprint mismatch between target metric and shards: target "
                    f"{target_fp[:12]}…, shards {sorted(fp[:12] + '…' for fp in shard_fps)}. "
                    "Merging shards computed under a different configuration is refused."
                )
            target_class = f"{self.target_metric.__class__.__module__}.{self.target_metric.__class__.__qualname__}"
            if {envelope.metric_class for envelope in envelopes} != {target_class}:
                raise TorchMetricsUserError(
                    f"metric class mismatch: target is {target_class!r} but shards originate from "
                    f"{sorted({envelope.metric_class for envelope in envelopes})}"
                )

        report.raise_if_invalid()

        if report.missing_shard_ids and self.missing_policy == "reject":
            raise TorchMetricsUserError(
                f"refusing to merge: {len(report.missing_shard_ids)} expected shard(s) missing: "
                f"{report.missing_shard_ids}. Re-run with missing_policy='partial' to merge the arrived shards "
                "with an explicit coverage annotation."
            )

        # coverage alerts only make sense for merges that actually proceed
        events = self._coverage_events(report)

        accepted = self._canonical_order(fresh_envelopes, report)

        # ---- two-phase commit: reduce onto a scratch copy of the current target; publish only after success ----
        scratch = self.target_metric.clone()
        reduction_by_state, consistency = self._reduce_into(scratch, accepted)
        # all checks passed: publish the scratch state to the real target
        _assign_state(self.target_metric, scratch.metric_state)
        # remember provenance only after a successful commit
        for envelope in accepted:
            self._consumed_shard_ids.add(envelope.shard_id)
            self._consumed_checksums.add(envelope.checksum)

        manifest = self._build_manifest(accepted, report, reduction_by_state, consistency, events)
        # redelivered shards were filtered before validation, so surface them on the manifest explicitly
        manifest.duplicate_shard_ids.extend(redelivered)
        self.last_manifest = manifest
        self.events.extend(events)
        return self.target_metric

    def merge_to_envelope(
        self,
        shards: List[Union[ShardEnvelope, dict[str, Any]]],
        coverage_count: Optional[int] = None,
        *,
        extra_config: Optional[dict[str, Any]] = None,
    ) -> ShardEnvelope:
        """Merge shards and return a new envelope that can itself participate in a later merge round.

        The merged envelope carries the flattened lineage of every input shard, so chained/hierarchical merges never
        lose source information. Its status is ``"partial"`` when coverage is incomplete.
        """
        self.merge(shards, extra_config=extra_config)
        if self.last_manifest is None:
            raise RuntimeError("merge did not produce a manifest; cannot seal an envelope")
        status = "complete"
        if self.last_manifest.coverage_ratio is not None and self.last_manifest.coverage_ratio < 1.0:
            status = "partial"
        lineage = sorted({ancestor for shard in self._accepted_envelopes(shards) for ancestor in shard.lineage})
        from torchmetrics.sharding.envelope import Coverage

        global_count = self.expected_total_count
        count = coverage_count if coverage_count is not None else self.last_manifest.total_count
        return ShardEnvelope.from_metric(
            self.target_metric,
            Coverage(count=count, global_count=global_count),
            status=status,
            extra_config=extra_config,
            lineage=lineage,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------------------------------------------------------
    def _accepted_envelopes(self, shards: List[Union[ShardEnvelope, dict[str, Any]]]) -> list[ShardEnvelope]:
        envelopes = [shard if isinstance(shard, ShardEnvelope) else ShardEnvelope.from_dict(shard) for shard in shards]
        seen: set[str] = set()
        accepted: list[ShardEnvelope] = []
        for envelope in envelopes:
            if envelope.shard_id in seen or envelope.checksum in seen:
                continue
            seen.add(envelope.shard_id)
            seen.add(envelope.checksum)
            accepted.append(envelope)
        return accepted

    def _canonical_order(self, envelopes: list[ShardEnvelope], report: ValidationReport) -> list[ShardEnvelope]:
        """Return accepted shards, deduplicated by id, in a deterministic content-derived order."""
        accepted_ids = set(report.accepted_shard_ids)
        unique: dict[str, ShardEnvelope] = {}
        for envelope in envelopes:
            if envelope.shard_id in accepted_ids and envelope.shard_id not in unique:
                unique[envelope.shard_id] = envelope
        return sorted(unique.values(), key=lambda env: (env.checksum, env.shard_id))

    def _reduce_into(self, scratch: Metric, accepted: list[ShardEnvelope]) -> tuple[dict[str, str], dict[str, str]]:
        """Reduce every state variable of the accepted shards into the scratch metric.

        The scratch metric's current state participates as one additional, already-accumulated operand, so
        incremental calls (merge two shards, then two more) produce the same state as a single four-shard merge.
        """
        reduction_by_state: dict[str, str] = {}
        consistency: dict[str, str] = {}
        merged_state: dict[str, Any] = {}
        for state_name in scratch._defaults:
            shard_values = [envelope.state[state_name] for envelope in accepted]
            current_value = scratch.metric_state[state_name]
            reducer: Reducer = reducer_for(scratch._reductions[state_name], current_value, state_name)
            operands = [current_value, *shard_values]
            merged = reducer.reduce(operands)
            # consistency check over the same operands that produced the merged value
            reducer.consistency_check(operands, merged)
            reduction_by_state[state_name] = reducer.name
            consistency[state_name] = "passed"
            merged_state[state_name] = merged
        _assign_state(scratch, merged_state)
        return reduction_by_state, consistency

    def _coverage_events(self, report: ValidationReport) -> list[ShardEvent]:
        events: list[ShardEvent] = []
        if report.coverage_ratio is not None and report.coverage_ratio < self.min_coverage:
            event = LowCoverageEvent(
                message=(
                    f"merged coverage {report.coverage_ratio:.4f} is at or below threshold {self.min_coverage:.4f}: "
                    f"{report.arrived_total_count}/{report.expected_total_count} samples arrived, missing shards "
                    f"{report.missing_shard_ids}"
                ),
                context={
                    "arrived_count": report.arrived_total_count,
                    "expected_count": report.expected_total_count,
                    "missing_shard_ids": list(report.missing_shard_ids),
                },
                coverage_ratio=report.coverage_ratio,
                threshold=self.min_coverage,
                missing_shard_ids=list(report.missing_shard_ids),
            )
            events.append(event)
            # never silent: standard warning plus structured event plus optional handler
            warnings.warn(event.message, UserWarning, stacklevel=3)
            if self.event_handler is not None:
                self.event_handler(event)
        return events

    def _build_manifest(
        self,
        accepted: list[ShardEnvelope],
        report: ValidationReport,
        reduction_by_state: dict[str, str],
        consistency: dict[str, str],
        events: list[ShardEvent],
    ) -> MergeManifest:
        lineage = sorted({ancestor for envelope in accepted for ancestor in envelope.lineage})
        return MergeManifest(
            metric_class=accepted[0].metric_class if accepted else self.target_metric.__class__.__name__,
            shard_records=[
                {
                    "shard_id": envelope.shard_id,
                    "count": envelope.coverage.count,
                    "interval": envelope.coverage.interval,
                    "checksum": envelope.checksum,
                    "status": envelope.status,
                }
                for envelope in accepted
            ],
            total_count=report.arrived_total_count,
            expected_total_count=report.expected_total_count,
            coverage_ratio=report.coverage_ratio,
            duplicate_shard_ids=list(report.duplicate_shard_ids),
            missing_shard_ids=list(report.missing_shard_ids),
            reduction_by_state=reduction_by_state,
            lineage=lineage,
            consistency_checks=consistency,
            events=events,
        )

    @property
    def consumed_shard_ids(self) -> set[str]:
        """Shard ids already folded into the target metric across all merge calls."""
        return set(self._consumed_shard_ids)


def merge_shards(
    target_metric: Metric,
    shards: List[Union[ShardEnvelope, dict[str, Any]]],
    *,
    expected_shard_ids: Optional[list[str]] = None,
    expected_total_count: Optional[int] = None,
    missing_policy: str = "reject",
    min_coverage: float = 1.0,
    event_handler: Optional[Callable[[ShardEvent], None]] = None,
    extra_config: Optional[dict[str, Any]] = None,
) -> tuple[Metric, MergeManifest]:
    """Convenience one-shot wrapper around :class:`ShardMerger`.

    Returns the merged metric together with its :class:`MergeManifest`.
    """
    merger = ShardMerger(
        target_metric,
        expected_shard_ids=expected_shard_ids,
        expected_total_count=expected_total_count,
        missing_policy=missing_policy,
        min_coverage=min_coverage,
        event_handler=event_handler,
    )
    merger.merge(shards, extra_config=extra_config)
    if merger.last_manifest is None:
        raise RuntimeError("merge did not produce a manifest")
    return target_metric, merger.last_manifest
