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
"""Pre-merge validation checklist.

The validator answers the questions the quarterly review could not:

* which shards are missing,
* which submissions are duplicates (by shard id *and* by content checksum — file names are never trusted),
* which shard intervals are out of bounds or overlap,
* whether any configuration fingerprints disagree (a hard error — mixed configurations are never merged),
* which concrete shard has the wrong sample count and by how many samples,
* which shard carries a non-finite value and in which state variable.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from torchmetrics.sharding.envelope import ShardEnvelope
from torchmetrics.utilities.exceptions import TorchMetricsUserError


@dataclass
class ValidationReport:
    """Outcome of validating a batch of shards before merging.

    Attributes:
        accepted_shard_ids: shard ids that passed every check and should participate in the merge.
        duplicate_shard_ids: shard ids submitted more than once; only the first copy is accepted.
        duplicate_checksum_ids: groups of distinct shard ids whose state payloads are byte-identical.
        missing_shard_ids: expected shard ids that were never submitted.
        expected_total_count: sample count that the full set of expected shards should cover.
        arrived_total_count: sample count carried by the accepted shards.
        coverage_ratio: ``arrived_total_count / expected_total_count`` when an expected count is known.
        errors: human-readable, shard-attributed error messages.
        non_finite_findings: per-shard, per-state-variable descriptions of non-finite values.

    """

    accepted_shard_ids: list[str] = field(default_factory=list)
    duplicate_shard_ids: list[str] = field(default_factory=list)
    duplicate_checksum_ids: list[list[str]] = field(default_factory=list)
    missing_shard_ids: list[str] = field(default_factory=list)
    expected_total_count: Optional[int] = None
    arrived_total_count: int = 0
    coverage_ratio: Optional[float] = None
    errors: list[str] = field(default_factory=list)
    non_finite_findings: list[str] = field(default_factory=list)
    count_gap: Optional[int] = None

    @property
    def ok(self) -> bool:
        """``True`` when no hard error was found (duplicates and missing shards are reported, not errors)."""
        return not self.errors and not self.non_finite_findings

    def raise_if_invalid(self) -> None:
        """Raise a :class:`~torchmetrics.utilities.exceptions.TorchMetricsUserError` aggregating all findings."""
        problems = [*self.errors, *self.non_finite_findings]
        if problems:
            raise TorchMetricsUserError(
                "Shard validation failed with {} problem(s):\n  - {}".format(len(problems), "\n  - ".join(problems))
            )


def _iter_state_tensors(state: dict) -> list[tuple[str, Tensor]]:
    """Flatten a metric state into ``(state_variable_path, tensor)`` pairs."""
    found: list[tuple[str, Tensor]] = []

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, Tensor):
            found.append((prefix, value))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                walk(f"{prefix}[{idx}]", item)

    for name, value in state.items():
        walk(name, value)
    return found


def check_finite(envelope: ShardEnvelope) -> list[str]:
    """Return descriptions of any non-finite (NaN / +-Inf) values in an envelope's state."""
    findings: list[str] = []
    for path, tensor in _iter_state_tensors(envelope.state):
        if tensor.is_floating_point() and not torch_all_finite(tensor):
            n_nan = int(torch.isnan(tensor).sum())
            n_inf = int(torch.isinf(tensor).sum())
            findings.append(
                f"shard {envelope.shard_id!r} state variable {path!r} contains non-finite values "
                f"({n_nan} NaN, {n_inf} Inf); merge output will not be written"
            )
    return findings


def torch_all_finite(tensor: Tensor) -> bool:
    """Wrapper around :func:`torch.isfinite` that also works for empty tensors."""
    return bool(torch.isfinite(tensor).all()) if tensor.numel() else True


def validate_shards(
    shards: list[ShardEnvelope],
    *,
    expected_shard_ids: Optional[list[str]] = None,
    expected_total_count: Optional[int] = None,
    expected_state_keys: Optional[list[str]] = None,
) -> ValidationReport:
    """Run the full pre-merge validation checklist.

    Args:
        shards: envelopes submitted for merging (already parsed through :meth:`ShardEnvelope.from_dict`, which
            verifies per-envelope shape and content checksums).
        expected_shard_ids: when the job knows exactly which shards should exist (e.g. four fixed partitions), pass
            their ids here so missing shards can be reported.
        expected_total_count: global sample count the job expects. Used for coverage accounting and for the
            per-shard count mismatch check.
        expected_state_keys: state variable names every shard must contain. Defaults to the keys of the first shard.

    Hard errors (collected, then raised together via :meth:`ValidationReport.raise_if_invalid`):

        * more than one metric class or more than one configuration fingerprint,
        * state shapes/keys that disagree across shards,
        * intervals outside the declared global range or overlapping accepted intervals,
        * count mismatches against declared intervals or the global total,
        * non-finite values in any state tensor.

    Duplicate submissions (same shard id or same content checksum) and missing expected shards are *reported* in the
    report rather than raised: the merger uses the report to apply idempotent deduplication or partial-coverage
    policy instead of failing outright.

    """
    report = ValidationReport(expected_total_count=expected_total_count)
    if not shards:
        report.errors.append("no shards were submitted for merging")
        return report

    reference = shards[0]
    if expected_state_keys is None:
        expected_state_keys = sorted(reference.state)

    # --- structural consistency across shards ---------------------------------------------------------
    metric_classes: set[str] = set()
    fingerprints: set[str] = set()
    seen_ids: set[str] = set()
    seen_checksums: dict[str, str] = {}
    accepted: list[ShardEnvelope] = []

    for envelope in shards:
        sid = envelope.shard_id
        metric_classes.add(envelope.metric_class)
        fingerprints.add(envelope.config_fingerprint)

        if sid in seen_ids:
            report.duplicate_shard_ids.append(sid)
            continue
        # content-side identity: identical payload under a different id is still a duplicate
        if envelope.checksum in seen_checksums:
            report.duplicate_checksum_ids.append([seen_checksums[envelope.checksum], sid])
            report.duplicate_shard_ids.append(sid)
            continue
        seen_ids.add(sid)
        seen_checksums[envelope.checksum] = sid

        missing_keys = [key for key in expected_state_keys if key not in envelope.state]
        if missing_keys:
            report.errors.append(
                f"shard {sid!r} state is missing required variable(s) {missing_keys}; "
                f"present variables: {sorted(envelope.state)}"
            )
            continue
        extra_keys = [key for key in envelope.state if key not in expected_state_keys]
        if extra_keys:
            report.errors.append(
                f"shard {sid!r} state contains unexpected variable(s) {extra_keys}; "
                f"expected variables: {expected_state_keys}"
            )
            continue

        for key in expected_state_keys:
            if key not in reference.state:
                # the reference shard itself is missing this key; every shard lacking it is reported separately
                if key in envelope.state:
                    report.errors.append(
                        f"shard {sid!r} state contains unexpected variable {key!r}; reference shard "
                        f"{reference.shard_id!r} does not define it"
                    )
                continue
            shape_problem = _check_state_shape(key, reference.state[key], envelope.state[key], sid)
            if shape_problem:
                report.errors.append(shape_problem)

        report.non_finite_findings.extend(check_finite(envelope))
        accepted.append(envelope)

    if len(metric_classes) > 1:
        report.errors.append(f"shards originate from different metric classes: {sorted(metric_classes)}")
    if len(fingerprints) > 1:
        groups: dict[str, list[str]] = {}
        for envelope in shards:
            groups.setdefault(envelope.config_fingerprint, []).append(envelope.shard_id)
        detail = "; ".join(f"fingerprint {fp[:12]}… -> shards {ids}" for fp, ids in sorted(groups.items()))
        report.errors.append(
            "configuration fingerprint mismatch: shards were computed with different configurations and cannot be "
            f"mixed ({detail})"
        )

    # --- coverage / interval checks -------------------------------------------------------------------
    intervals: list[tuple[int, int, str]] = []
    global_count = expected_total_count
    for envelope in accepted:
        cov = envelope.coverage
        if global_count is None and cov.global_count is not None:
            global_count = cov.global_count
        if cov.interval is not None:
            start, end = cov.interval
            if global_count is not None and (start < 0 or end >= global_count):
                report.errors.append(
                    f"shard {envelope.shard_id!r} interval [{start}, {end}] is out of bounds for "
                    f"global_count={global_count}"
                )
            intervals.append((start, end, envelope.shard_id))

    intervals.sort()
    for (start_a, end_a, id_a), (start_b, end_b, id_b) in zip(intervals, intervals[1:]):
        if start_b <= end_a:
            report.errors.append(
                f"overlapping shard intervals: shard {id_a!r} covers [{start_a}, {end_a}] and shard {id_b!r} "
                f"covers [{start_b}, {end_b}]"
            )

    # --- count bookkeeping ----------------------------------------------------------------------------
    report.arrived_total_count = sum(envelope.coverage.count for envelope in accepted)
    report.accepted_shard_ids = [envelope.shard_id for envelope in accepted]

    if expected_shard_ids is not None:
        report.missing_shard_ids = [sid for sid in expected_shard_ids if sid not in seen_ids]

    if expected_total_count is not None:
        report.coverage_ratio = report.arrived_total_count / expected_total_count if expected_total_count else 1.0
        diff = report.arrived_total_count - expected_total_count
        if diff != 0 and not report.missing_shard_ids and expected_shard_ids is not None:
            offending = [
                f"shard {envelope.shard_id!r} claims {envelope.coverage.count} samples" for envelope in accepted
            ]
            report.errors.append(
                f"sample count mismatch: accepted shards cover {report.arrived_total_count} samples but "
                f"{expected_total_count} were expected (difference of {diff:+d} samples). Claims: "
                + "; ".join(offending)
            )
        elif diff != 0:
            # either shards are explicitly missing, or no fixed shard list was given: treat as a coverage gap and
            # let the merger's missing/coverage policy decide (reject, or partial merge with an alert)
            report.count_gap = expected_total_count - report.arrived_total_count
    return report


def _check_state_shape(key: str, reference_value: object, incoming_value: object, shard_id: str) -> Optional[str]:
    """Return an error string when two shard state variables have incompatible shapes/types."""
    if isinstance(reference_value, Tensor) != isinstance(incoming_value, Tensor):
        return (
            f"shard {shard_id!r} state variable {key!r} has kind {type(incoming_value).__name__} but expected "
            f"{type(reference_value).__name__}"
        )
    if isinstance(reference_value, Tensor):
        if reference_value.shape != incoming_value.shape:
            return (
                f"shard {shard_id!r} state variable {key!r} has shape {tuple(incoming_value.shape)} but expected "
                f"{tuple(reference_value.shape)}"
            )
        if reference_value.dtype != incoming_value.dtype:
            return (
                f"shard {shard_id!r} state variable {key!r} has dtype {incoming_value.dtype} but expected "
                f"{reference_value.dtype}"
            )
    elif isinstance(reference_value, list):
        if not isinstance(incoming_value, list):
            return f"shard {shard_id!r} state variable {key!r} must be a list, got {type(incoming_value).__name__}"
        if reference_value and incoming_value:
            ref_t = reference_value[0]
            inc_t = incoming_value[0]
            if isinstance(ref_t, Tensor) and isinstance(inc_t, Tensor):
                if ref_t.shape[1:] != inc_t.shape[1:]:
                    return (
                        f"shard {shard_id!r} list state {key!r} element shape {tuple(inc_t.shape[1:])} does not "
                        f"match reference {tuple(ref_t.shape[1:])}"
                    )
                if ref_t.dtype != inc_t.dtype:
                    return (
                        f"shard {shard_id!r} list state {key!r} dtype {inc_t.dtype} does not match reference "
                        f"{ref_t.dtype}"
                    )
    return None
