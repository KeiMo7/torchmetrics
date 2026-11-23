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
"""Tests for the opt-in sharded merge toolkit (``torchmetrics.sharding``)."""

import copy
import itertools
import json
import subprocess
import sys
import warnings
from typing import Any

import pytest
import torch

from torchmetrics.aggregation import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics.metric import Metric
from torchmetrics.sharding import (
    Coverage,
    LowCoverageEvent,
    ShardEnvelope,
    ShardMerger,
    config_fingerprint,
    content_checksum,
    merge_shards,
    validate_shards,
)
from torchmetrics.sharding.envelope import _state_from_jsonable
from torchmetrics.sharding.reducers import CatReducer, ExtremaReducer, MeanReducer, SumReducer, reducer_for
from torchmetrics.utilities.data import dim_zero_cat, dim_zero_max, dim_zero_min, dim_zero_sum
from torchmetrics.utilities.exceptions import TorchMetricsUserError

DATA = torch.arange(100.0, dtype=torch.float32)
N_SHARDS = 4
SHARD_SIZE = 25


def _shards_for(
    metric_cls: type[Metric], data: torch.Tensor = DATA, n: int = N_SHARDS, **metric_kwargs: Any
) -> list[ShardEnvelope]:
    """Build ``n`` sealed shard envelopes for a metric class by splitting ``data`` into equal chunks."""
    shards = []
    step = len(data) // n
    for i in range(n):
        metric = metric_cls(**metric_kwargs)
        chunk = data[i * step : (i + 1) * step]
        metric.update(chunk)
        shards.append(
            ShardEnvelope.from_metric(
                metric, Coverage(count=chunk.numel(), interval=(i * step, (i + 1) * step - 1), global_count=len(data))
            )
        )
    return shards


def _single_machine(metric_cls: type[Metric], data: torch.Tensor = DATA, **metric_kwargs: Any) -> Metric:
    """Accumulate the full dataset in one metric instance, the single-machine reference."""
    metric = metric_cls(**metric_kwargs)
    metric.update(data)
    return metric


# ----------------------------------------------------------------------------------------------------------------------
# envelope construction / serialization
# ----------------------------------------------------------------------------------------------------------------------
def test_envelope_carries_identity_coverage_status_fingerprint() -> None:
    """A sealed envelope carries id, coverage interval, status, fingerprint and content checksum."""
    shards = _shards_for(SumMetric)
    envelope = shards[0]
    assert isinstance(envelope.shard_id, str)
    assert envelope.shard_id
    assert envelope.status == "complete"
    assert envelope.coverage.count == SHARD_SIZE
    assert envelope.coverage.interval == (0, 24)
    assert envelope.coverage.global_count == len(DATA)
    assert len(envelope.config_fingerprint) == 64
    assert len(envelope.checksum) == 64
    assert envelope.lineage == [envelope.shard_id]
    assert "sum_value" in envelope.state


def test_fingerprint_stable_across_instances_but_sensitive_to_config() -> None:
    """Same config yields the same fingerprint regardless of accumulated data; different config differs."""
    m1, m2 = SumMetric(), SumMetric()
    m1.update(DATA[:10])
    m2.update(DATA[20:30])
    assert config_fingerprint(m1) == config_fingerprint(m2)

    different = SumMetric(nan_strategy="ignore")
    assert config_fingerprint(different) != config_fingerprint(m1)

    mean_fp = config_fingerprint(MeanMetric())
    assert mean_fp != config_fingerprint(m1)


def test_content_checksum_detects_duplicate_and_tampering() -> None:
    """The checksum is content-derived: tampering breaks it, and a renamed identical payload keeps it."""
    shards = _shards_for(SumMetric)
    payload = json.loads(shards[0].to_json())
    rebuilt = ShardEnvelope.from_dict(payload)
    assert rebuilt.checksum == shards[0].checksum

    tampered = copy.deepcopy(payload)
    tampered["state"]["sum_value"]["data"] = 99999.0
    with pytest.raises(TorchMetricsUserError, match="checksum"):
        ShardEnvelope.from_dict(tampered)

    # same content under a different id keeps the same checksum -> identity is content-side, not name-side
    renamed = copy.deepcopy(payload)
    renamed["shard_id"] = "totally-different-name"
    renamed_envelope = ShardEnvelope.from_dict(renamed)
    assert renamed_envelope.checksum == shards[0].checksum


def test_missing_field_names_field_and_shard() -> None:
    """A malformed payload reports both the originating shard and the missing field name."""
    shards = _shards_for(SumMetric)
    payload = json.loads(shards[0].to_json())
    shard_id = payload["shard_id"]
    del payload["coverage"]
    with pytest.raises(TorchMetricsUserError, match=rf"{shard_id}.*missing required field.*coverage"):
        ShardEnvelope.from_dict(payload)


def test_json_roundtrip_preserves_everything() -> None:
    """Serializing to JSON and parsing back preserves identity, checksum, tensors and coverage."""
    shards = _shards_for(MeanMetric)
    text = shards[1].to_json()
    rebuilt = ShardEnvelope.from_json(text)
    assert rebuilt.shard_id == shards[1].shard_id
    assert rebuilt.checksum == shards[1].checksum
    assert torch.equal(rebuilt.state["mean_value"], shards[1].state["mean_value"])
    assert rebuilt.coverage.interval == (25, 49)


def test_invalid_coverage_construction() -> None:
    """Coverage rejects negative counts, interval/count mismatches and out-of-bounds intervals."""
    with pytest.raises(TorchMetricsUserError, match="non-negative integer"):
        Coverage(count=-1)
    with pytest.raises(TorchMetricsUserError, match="does not match interval length"):
        Coverage(count=3, interval=(0, 9))
    with pytest.raises(TorchMetricsUserError, match="out of bounds"):
        Coverage(count=10, interval=(95, 104), global_count=100)
    with pytest.raises(TorchMetricsUserError, match="pair of ints"):
        Coverage(count=2, interval=(1,))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------------------------------------
# acceptance 1: four shards, bit-identical result for every permutation
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("metric_cls", [SumMetric, MeanMetric, MaxMetric, MinMetric])
def test_order_insensitive_bit_identical(metric_cls: type[Metric]) -> None:
    """Every permutation of four shards merges to one single value, exactly equal to the single-machine value."""
    shards = _shards_for(metric_cls)
    reference = _single_machine(metric_cls).compute()
    values = set()
    for permutation in itertools.permutations(shards):
        target = metric_cls()
        ShardMerger(target, expected_total_count=len(DATA)).merge(list(permutation))
        values.add(target.compute().item())
    assert len(values) == 1
    merged = next(iter(values))
    assert merged == reference.item()  # exact equality, not just "same magnitude"


def test_cat_semantics_order_insensitive_multiset() -> None:
    """Concatenation produces the identical ordered tensor for every permutation and the full multiset."""
    shards = _shards_for(CatMetric)
    outputs = set()
    for permutation in itertools.permutations(shards):
        target = CatMetric()
        ShardMerger(target, expected_total_count=len(DATA)).merge(list(permutation))
        outputs.add(tuple(target.compute().tolist()))
    assert len(outputs) == 1  # all permutations produce the identical ordered tensor

    target = CatMetric()
    ShardMerger(target, expected_total_count=len(DATA)).merge(shards)
    reference = _single_machine(CatMetric).compute()
    assert torch.equal(target.compute().sort().values, reference.sort().values)
    assert target.compute().numel() == len(DATA)


# ----------------------------------------------------------------------------------------------------------------------
# acceptance 2: idempotent duplicate submission
# ----------------------------------------------------------------------------------------------------------------------
def test_duplicate_shard_id_is_idempotent() -> None:
    """Resubmitting already-merged shards leaves the result unchanged and records them as duplicates."""
    shards = _shards_for(SumMetric)
    target = SumMetric()
    merger = ShardMerger(target, expected_total_count=len(DATA))
    merger.merge(shards)
    first = target.compute().item()

    # resubmit the whole batch plus one extra copy; all five submissions are redeliveries of already-merged
    # shards and must be filtered before validation, leaving the value untouched
    merger.merge([*shards, shards[2]])
    assert target.compute().item() == first
    assert set(merger.last_manifest.duplicate_shard_ids) == {shard.shard_id for shard in shards}


def test_duplicate_content_under_new_id_is_idempotent() -> None:
    """A byte-identical payload under a new id is recognized as a duplicate via its content checksum."""
    shards = _shards_for(SumMetric)
    payload = json.loads(shards[0].to_json())
    payload["shard_id"] = "renamed"
    renamed = ShardEnvelope.from_dict(payload)
    target = SumMetric()
    merger = ShardMerger(target, expected_total_count=len(DATA))
    merger.merge([*shards, renamed])
    assert target.compute().item() == DATA.sum().item()
    assert merger.last_manifest.duplicate_shard_ids == ["renamed"]


def test_same_id_duplicate_inside_one_batch_is_deduplicated() -> None:
    """Two envelopes with the same shard id in one batch fold in exactly once."""
    shards = _shards_for(SumMetric)
    target = SumMetric()
    merger = ShardMerger(target, expected_total_count=len(DATA))
    merger.merge([*shards, shards[1]])
    # the duplicate must contribute nothing: the value is the full sum, not the sum plus an extra shard
    assert target.compute().item() == DATA.sum().item()
    assert merger.last_manifest.duplicate_shard_ids == [shards[1].shard_id]
    assert len(merger.last_manifest.shard_records) == N_SHARDS


# ----------------------------------------------------------------------------------------------------------------------
# acceptance 3: missing / duplicate / fingerprint mismatch verdicts and messages
# ----------------------------------------------------------------------------------------------------------------------
def test_missing_shard_rejected_by_default() -> None:
    """The default reject policy refuses to merge when an expected shard id is absent."""
    shards = _shards_for(SumMetric)
    ids = [shard.shard_id for shard in shards]
    with pytest.raises(TorchMetricsUserError, match=r"refusing to merge.*missing"):
        ShardMerger(SumMetric(), expected_shard_ids=ids, expected_total_count=len(DATA)).merge(shards[:3])


def test_missing_shard_partial_policy_emits_event_and_annotates() -> None:
    """Partial policy merges the arrived shards and emits an observable low-coverage event with the ratio."""
    shards = _shards_for(SumMetric)
    ids = [shard.shard_id for shard in shards]
    events: list[LowCoverageEvent] = []
    target = SumMetric()
    merger = ShardMerger(
        target,
        expected_shard_ids=ids,
        expected_total_count=len(DATA),
        missing_policy="partial",
        min_coverage=0.9,
        event_handler=events.append,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merger.merge(shards[:3])

    assert len(events) == 1
    assert isinstance(events[0], LowCoverageEvent)
    assert events[0].coverage_ratio == pytest.approx(0.75)
    assert events[0].missing_shard_ids == [ids[3]]
    assert any("coverage" in str(w.message) for w in caught)
    assert target.compute().item() == DATA[:75].sum().item()
    manifest = merger.last_manifest
    assert manifest.coverage_ratio == pytest.approx(0.75)
    assert manifest.missing_shard_ids == [ids[3]]
    assert manifest.total_count == 75


def test_config_fingerprint_mismatch_refused() -> None:
    """Shards computed under different configurations are never mixed."""
    shards = _shards_for(SumMetric)
    other = SumMetric(nan_strategy="ignore")
    other.update(DATA[:SHARD_SIZE])
    bad = ShardEnvelope.from_metric(other, Coverage(SHARD_SIZE, (0, 24), len(DATA)))
    with pytest.raises(TorchMetricsUserError, match="fingerprint"):
        ShardMerger(SumMetric(), expected_total_count=len(DATA)).merge([*shards[:3], bad])


def test_target_fingerprint_mismatch_refused() -> None:
    """A target metric whose own fingerprint differs from the shards is refused."""
    shards = _shards_for(SumMetric)
    target = SumMetric(nan_strategy="ignore")
    with pytest.raises(TorchMetricsUserError, match=r"fingerprint.*target"):
        ShardMerger(target, expected_total_count=len(DATA)).merge(shards)


# ----------------------------------------------------------------------------------------------------------------------
# validation details: intervals, counts, shapes, non-finite values
# ----------------------------------------------------------------------------------------------------------------------
def test_overlapping_intervals_rejected_with_both_shard_ids() -> None:
    """Overlapping coverage intervals are rejected and the error names both offending shards."""
    shards = _shards_for(SumMetric)
    overlapping = list(shards)
    metric = SumMetric()
    metric.update(DATA[50:75])
    # same shard id and lineage retained; only the interval deliberately collides with shard 2 ([50, 74])
    overlapping[3] = ShardEnvelope(
        shard_id=shards[3].shard_id,
        metric_class=shards[3].metric_class,
        config_fingerprint=shards[3].config_fingerprint,
        status="complete",
        coverage=Coverage(SHARD_SIZE, (50, 74), len(DATA)),
        state=shards[3].state,
        checksum=shards[3].checksum,
        lineage=shards[3].lineage,
    )
    ids = [shard.shard_id for shard in shards]
    with pytest.raises(TorchMetricsUserError, match="overlapping shard intervals") as exc:
        ShardMerger(SumMetric(), expected_shard_ids=ids, expected_total_count=len(DATA)).merge(overlapping)
    message = str(exc.value)
    assert shards[2].shard_id in message
    assert shards[3].shard_id in message


def test_count_mismatch_locates_shard_and_difference() -> None:
    """With all fixed shards present, a wrong claim reports the offending shard and signed difference."""
    # all four fixed shards arrive, but one claims one sample too many without an interval overlap
    shards = _shards_for(SumMetric)
    ids = [shard.shard_id for shard in shards]
    lying = ShardEnvelope(
        shard_id=shards[2].shard_id,
        metric_class=shards[2].metric_class,
        config_fingerprint=shards[2].config_fingerprint,
        status="complete",
        coverage=Coverage(SHARD_SIZE + 1, None, len(DATA)),
        state=shards[2].state,
        checksum=shards[2].checksum,
        lineage=[shards[2].shard_id],
    )
    altered = [*shards[:2], lying, shards[3]]
    with pytest.raises(TorchMetricsUserError, match=r"difference of \+1") as exc:
        ShardMerger(SumMetric(), expected_shard_ids=ids, expected_total_count=len(DATA)).merge(altered)
    assert ids[2] in str(exc.value)


def test_state_shape_mismatch_names_shard_and_variable() -> None:
    """An incompatible state shape is attributed to the specific shard and state variable."""
    shards = _shards_for(SumMetric)
    # force an incompatible scalar-vs-vector shape with the same fingerprint class
    weird = ShardEnvelope(
        shard_id="weird-shape",
        metric_class=shards[0].metric_class,
        config_fingerprint=shards[0].config_fingerprint,
        status="complete",
        coverage=Coverage(3),
        state={"sum_value": torch.zeros(4)},
        checksum=content_checksum({"sum_value": torch.zeros(4)}),
        lineage=["weird-shape"],
    )
    with pytest.raises(TorchMetricsUserError, match=r"weird-shape.*sum_value.*shape"):
        ShardMerger(SumMetric(), expected_total_count=28).merge([shards[0], weird])


def test_missing_state_variable_names_shard_and_variable() -> None:
    """A shard missing a required state variable is reported with its id and the variable name."""
    shards = _shards_for(MeanMetric)
    payload = json.loads(shards[0].to_json())
    del payload["state"]["weight"]
    # recompute the checksum for the truncated state so envelope parsing itself passes and validation reports
    # the missing variable rather than a checksum failure
    truncated_state = _state_from_jsonable(payload["state"], payload["shard_id"])
    payload["checksum"] = content_checksum(truncated_state)
    truncated = ShardEnvelope.from_dict(payload)
    report = validate_shards([truncated, *shards[1:]], expected_state_keys=sorted(shards[0].state))
    assert not report.ok
    assert any(truncated.shard_id in problem and "weight" in problem for problem in report.errors)


def test_non_finite_value_names_source_shard_and_variable() -> None:
    """NaN/Inf in a shard state names the source shard and variable and blocks the write."""
    metric = SumMetric(nan_strategy="disable")
    metric.update(torch.tensor([1.0, float("nan"), float("inf")]))
    envelope = ShardEnvelope.from_metric(metric, Coverage(3, (0, 2), 3))
    with pytest.raises(TorchMetricsUserError, match=rf"{envelope.shard_id}.*sum_value.*non-finite"):
        ShardMerger(SumMetric(nan_strategy="disable"), expected_total_count=3).merge([envelope])


def test_failed_merge_is_atomic_target_unchanged() -> None:
    """A rejected merge never mutates the target metric's pre-existing state."""
    shards = _shards_for(SumMetric)
    ids = [shard.shard_id for shard in shards]
    target = SumMetric()
    target.update(torch.tensor([7.0]))
    before = target.metric_state["sum_value"].clone()
    with pytest.raises(TorchMetricsUserError):
        ShardMerger(target, expected_shard_ids=ids, expected_total_count=len(DATA)).merge(shards[:2])
    assert torch.equal(target.metric_state["sum_value"], before)


# ----------------------------------------------------------------------------------------------------------------------
# acceptance 4: tolerance against single-machine run
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("metric_cls", [SumMetric, MeanMetric, MaxMetric, MinMetric])
def test_matches_single_machine_within_tolerance(metric_cls: type[Metric]) -> None:
    """Merged output matches the single-machine reference within the 1e-6 tolerance and passes all checks."""
    shards = _shards_for(metric_cls)
    target = metric_cls()
    _, manifest = merge_shards(target, shards, expected_total_count=len(DATA))
    merged = target.compute()
    reference = _single_machine(metric_cls).compute()
    abs_diff = (merged - reference).abs().max()
    # exact bit equality is achieved here; the contractual tolerance is 1e-6
    assert torch.allclose(merged, reference, atol=1e-6)
    assert abs_diff.item() == 0.0
    assert manifest.consistency_checks
    assert all(v == "passed" for v in manifest.consistency_checks.values())


def test_large_float_mean_tolerance_vs_single_machine() -> None:
    """On 100k floats the weighted shard-merge mean stays within tolerance of the single-machine mean."""
    generator = torch.Generator().manual_seed(123)
    data = torch.randn(100_000, generator=generator)
    shards = _shards_for(MeanMetric, data)
    target = MeanMetric()
    merge_shards(target, shards, expected_total_count=data.numel())
    merged = target.compute()
    reference = _single_machine(MeanMetric, data).compute()
    abs_diff = (merged - reference).abs().item()
    assert abs_diff <= 1e-6, f"abs diff {abs_diff} exceeded tolerance"


# ----------------------------------------------------------------------------------------------------------------------
# reducer semantics and their dedicated consistency checks
# ----------------------------------------------------------------------------------------------------------------------
def test_reducer_selection_covers_four_semantics() -> None:
    """Registered reduction functions map onto the four explicit reducer semantics."""
    assert isinstance(reducer_for(dim_zero_sum, torch.zeros(1)), SumReducer)
    assert isinstance(reducer_for(dim_zero_max, torch.zeros(1)), ExtremaReducer)
    assert isinstance(reducer_for(dim_zero_min, torch.zeros(1)), ExtremaReducer)
    assert isinstance(reducer_for(dim_zero_cat, []), CatReducer)
    weight_reducer = reducer_for(dim_zero_sum, torch.zeros(1), "weight")
    assert isinstance(weight_reducer, MeanReducer)


def test_mean_uses_weighted_not_naive_average() -> None:
    """Unequal shard sizes produce the global weighted mean, not a naive mean of per-shard means."""
    # shards of different sizes: naive mean-of-means would be wrong
    m_small, m_big = MeanMetric(), MeanMetric()
    m_small.update(torch.tensor([0.0, 10.0]))  # mean 5, weight 2
    m_big.update(torch.tensor([1.0, 2.0, 3.0, 4.0]))  # mean 2.5, weight 4
    env_small = ShardEnvelope.from_metric(m_small, Coverage(2, (0, 1), 6))
    env_big = ShardEnvelope.from_metric(m_big, Coverage(4, (2, 5), 6))
    target = MeanMetric()
    ShardMerger(target, expected_total_count=6).merge([env_small, env_big])
    # global mean of [0,10,1,2,3,4] = 20/6
    assert target.compute().item() == pytest.approx(20.0 / 6.0)
    assert target.compute().item() != pytest.approx((5.0 + 2.5) / 2.0)


def test_extrema_consistency_check_catches_invention() -> None:
    """The extrema check rejects a merged value not present among the contributing shards."""
    reducer = ExtremaReducer(dim_zero_max)
    shard_values = [torch.tensor(1.0), torch.tensor(3.0)]
    good = reducer.reduce(shard_values)
    reducer.consistency_check(shard_values, good)  # no raise
    with pytest.raises(RuntimeError, match="max consistency check failed"):
        reducer.consistency_check(shard_values, torch.tensor(99.0))


def test_cat_consistency_check_catches_lost_element() -> None:
    """The cat check rejects an output that dropped a shard element."""
    reducer = CatReducer()
    shard_values = [[torch.tensor([1.0, 2.0])], [torch.tensor([3.0])]]
    good = reducer.reduce(shard_values)
    reducer.consistency_check(shard_values, good)
    with pytest.raises(RuntimeError, match="cat consistency check failed"):
        reducer.consistency_check(shard_values, torch.tensor([1.0, 2.0]))  # 3.0 lost


# ----------------------------------------------------------------------------------------------------------------------
# audit manifest + hierarchical merging (lineage never lost)
# ----------------------------------------------------------------------------------------------------------------------
def test_manifest_lists_shards_counts_and_total() -> None:
    """The manifest records per-shard counts, totals, lineage and is itself JSON-serializable."""
    shards = _shards_for(SumMetric)
    target = SumMetric()
    _, manifest = merge_shards(target, shards, expected_total_count=len(DATA))
    assert len(manifest.shard_records) == N_SHARDS
    assert [row["count"] for row in manifest.shard_records] == [SHARD_SIZE] * N_SHARDS
    assert manifest.total_count == len(DATA)
    assert manifest.coverage_ratio == pytest.approx(1.0)
    assert manifest.lineage == sorted(s.shard_id for s in shards)
    assert manifest.reduction_by_state == {"sum_value": "sum"}
    assert all(row["checksum"] for row in manifest.shard_records)
    # manifest itself is JSON-serializable for reviewable storage
    json.dumps(manifest.to_dict())


def test_hierarchical_merge_preserves_lineage_and_value() -> None:
    """Merged envelopes re-enter later rounds, keeping full lineage and the same value."""
    shards = _shards_for(SumMetric)
    left = ShardMerger(SumMetric(), expected_total_count=len(DATA), missing_policy="partial").merge_to_envelope(
        shards[:2]
    )
    right = ShardMerger(SumMetric(), expected_total_count=len(DATA), missing_policy="partial").merge_to_envelope(
        shards[2:]
    )
    assert left.status == "partial"

    top_target = SumMetric()
    top_merger = ShardMerger(top_target, expected_total_count=len(DATA))
    top_envelope = top_merger.merge_to_envelope([left, right])

    assert top_envelope.status == "complete"
    assert sorted(top_envelope.lineage) == sorted(s.shard_id for s in shards)
    assert top_target.compute().item() == DATA.sum().item()

    # the merged envelope re-enters another merge round through JSON without losing provenance
    rebuilt = ShardEnvelope.from_json(top_envelope.to_json())
    again = SumMetric()
    ShardMerger(again, expected_total_count=len(DATA)).merge([rebuilt])
    assert again.compute().item() == DATA.sum().item()


# ----------------------------------------------------------------------------------------------------------------------
# generic path works for existing, unmodified metrics (no per-metric changes)
# ----------------------------------------------------------------------------------------------------------------------
def test_existing_classification_metric_merges_without_modification() -> None:
    """A multi-state domain metric merges through the generic path and matches the single-machine result exactly."""
    torch.manual_seed(0)
    n_samples, n_classes = 200, 4
    targets = torch.randint(n_classes, (n_samples,))
    preds = torch.randn(n_samples, n_classes).softmax(dim=-1)
    step = n_samples // N_SHARDS

    shards = []
    for i in range(N_SHARDS):
        metric = MulticlassAccuracy(num_classes=n_classes)
        sl = slice(i * step, (i + 1) * step)
        metric.update(preds[sl], targets[sl])
        shards.append(ShardEnvelope.from_metric(metric, Coverage(step, (i * step, (i + 1) * step - 1), n_samples)))

    merged = MulticlassAccuracy(num_classes=n_classes)
    _, manifest = merge_shards(merged, shards, expected_total_count=n_samples)
    reference = MulticlassAccuracy(num_classes=n_classes)
    reference.update(preds, targets)

    assert torch.equal(merged.compute(), reference.compute())
    # all four confusion-matrix states use the generic sum reducer, with no metric-specific code
    assert set(manifest.reduction_by_state.values()) == {"sum"}

    binary_targets = torch.randint(2, (n_samples,))
    binary_preds = torch.sigmoid(torch.randn(n_samples))
    binary_shards = []
    for i in range(N_SHARDS):
        metric = BinaryAccuracy()
        sl = slice(i * step, (i + 1) * step)
        metric.update(binary_preds[sl], binary_targets[sl])
        binary_shards.append(
            ShardEnvelope.from_metric(metric, Coverage(step, (i * step, (i + 1) * step - 1), n_samples))
        )
    binary_merged = BinaryAccuracy()
    merge_shards(binary_merged, binary_shards, expected_total_count=n_samples)
    binary_reference = BinaryAccuracy()
    binary_reference.update(binary_preds, binary_targets)
    assert binary_merged.compute().item() == binary_reference.compute().item()


# ----------------------------------------------------------------------------------------------------------------------
# feature switch: untouched existing behavior
# ----------------------------------------------------------------------------------------------------------------------
def test_incremental_merge_equivalent_to_one_shot() -> None:
    """Shards delivered across separate merge calls converge to the same state as a one-shot merge."""
    shards = _shards_for(MeanMetric)
    incremental = MeanMetric()
    merger = ShardMerger(incremental, expected_total_count=len(DATA), missing_policy="partial", min_coverage=0.0)
    merger.merge(shards[:2])
    merger.merge(shards[2:])

    one_shot = MeanMetric()
    ShardMerger(one_shot, expected_total_count=len(DATA)).merge(shards)
    reference = _single_machine(MeanMetric).compute()

    assert incremental.compute().item() == one_shot.compute().item() == reference.item()
    assert incremental.metric_state["mean_value"] == one_shot.metric_state["mean_value"]
    assert incremental.metric_state["weight"] == one_shot.metric_state["weight"]
    assert len(merger.consumed_shard_ids) == N_SHARDS


def test_existing_single_process_accumulation_unchanged() -> None:
    """Plain repeated update accumulation behaves exactly as before the sharding layer existed."""
    metric = MeanMetric()
    for chunk in DATA.split(SHARD_SIZE):
        metric.update(chunk)
    assert metric.compute().item() == DATA.mean().item()
    assert metric.update_count == N_SHARDS


def test_existing_merge_state_api_unchanged() -> None:
    """The pre-existing Metric.merge_state contract is preserved; the sharding layer never touches it."""
    # mirrors the repository's existing test_merge_state_feature_basic contract exactly; the sharding layer
    # must not alter this path (it never calls into it)
    m1, m2 = SumMetric(), SumMetric()
    m1.update(torch.tensor(1.0))
    m2.update(torch.tensor(2.0))
    m1.merge_state(m2)
    assert m1.compute().item() == 3.0
    m3 = SumMetric()
    m3.update(torch.tensor(1.0))
    m3.merge_state({"sum_value": torch.tensor(2.0)})
    assert m3.compute().item() == 3.0


def test_sharding_package_not_imported_by_metric_base() -> None:
    """Importing and using a metric in a clean interpreter never pulls in the sharding package."""
    code = (
        "import sys; import torchmetrics.aggregation; "
        "assert 'torchmetrics.sharding' not in sys.modules; "
        "from torchmetrics.aggregation import SumMetric; m=SumMetric(); m.update(1.0); print(m.compute().item())"
    )
    # the command is a hardcoded string and the executable is the current interpreter, so there is no untrusted input
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, shell=False
    )
    assert result.stdout.strip() == "1.0"
