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
import numpy as np
import pytest
import torch

from torchmetrics import Metric, MetricCollection
from torchmetrics.aggregation import MeanMetric, SumMetric
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.utilities.exceptions import TorchMetricsUserError
from torchmetrics.wrappers import GroupwiseWrapper
from unittests import NUM_PROCESSES, USE_PYTEST_POOL
from unittests._helpers import _IS_WINDOWS, seed_all
from unittests.conftest import setup_ddp

seed_all(42)


def _make_batches(num_batches=5, batch_size=16, keys=("alpha", "beta", "gamma"), seed=0):
    """Create deterministic batches of (values, groups)."""
    gen = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        values = torch.randn(batch_size, generator=gen)
        groups = [keys[i] for i in torch.randint(len(keys), (batch_size,), generator=gen).tolist()]
        batches.append((values, groups))
    return batches


def test_matches_per_group_baseline():
    """Per-group results must match a baseline of one separately created metric per group."""
    batches = _make_batches()
    keys = ["alpha", "beta", "gamma"]
    metric = GroupwiseWrapper(MeanMetric())
    baselines = {k: MeanMetric() for k in keys}
    overall_ref = MeanMetric()

    for values, groups in batches:
        metric.update(values, groups=groups)
        overall_ref.update(values)
        for key in keys:
            mask = torch.tensor([g == key for g in groups])
            if mask.any():
                baselines[key].update(values[mask])

    res = metric.compute()
    max_diff = 0.0
    for key in keys:
        diff = (res["groups"][key] - baselines[key].compute()).abs().item()
        max_diff = max(max_diff, diff)
        assert torch.allclose(res["groups"][key], baselines[key].compute(), atol=1e-6), f"group {key}"
    assert torch.allclose(res["overall"], overall_ref.compute(), atol=1e-6)
    print(f"\nmax abs diff vs per-group baseline: {max_diff}")


def test_matches_per_group_baseline_classification_metric():
    """Metrics with multiple tensor arguments (preds, target) are split per group correctly."""
    gen = torch.Generator().manual_seed(1)
    metric = GroupwiseWrapper(BinaryAccuracy())
    baselines = {"x": BinaryAccuracy(), "y": BinaryAccuracy()}
    for _ in range(4):
        preds = torch.rand(32, generator=gen)
        target = torch.randint(0, 2, (32,), generator=gen)
        groups = ["x" if i % 3 else "y" for i in torch.randint(3, (32,), generator=gen).tolist()]
        metric.update(preds, target, groups=groups)
        for key in ("x", "y"):
            mask = torch.tensor([g == key for g in groups])
            if mask.any():
                baselines[key].update(preds[mask], target[mask])
    res = metric.compute()
    for key in ("x", "y"):
        assert torch.allclose(res["groups"][key], baselines[key].compute(), atol=1e-6), f"group {key}"


def test_additive_consistency():
    """For additive metrics the sum over groups must equal the overall result."""
    metric = GroupwiseWrapper(SumMetric())
    for values, groups in _make_batches():
        metric.update(values, groups=groups)
    res = metric.compute()
    group_sum = sum(res["groups"].values())
    assert torch.allclose(group_sum, res["overall"], atol=1e-5)


def test_empty_batch_is_noop():
    """Empty batches must not create groups nor change state."""
    metric = GroupwiseWrapper(MeanMetric())
    metric.update(torch.empty(0), groups=[])
    assert metric.group_keys == []
    out = metric(torch.empty(0), groups=[])
    assert out == {"overall": None, "groups": {}}
    metric.update(torch.tensor([1.0, 2.0]), groups=["a", "a"])
    metric.update(torch.empty(0), groups=[])
    res = metric.compute()
    assert torch.allclose(res["overall"], torch.tensor(1.5))
    assert torch.allclose(res["groups"]["a"], torch.tensor(1.5))


def test_single_sample_and_single_group_batches():
    """Batches with one sample, and batches where all samples share one group, must work."""
    metric = GroupwiseWrapper(SumMetric())
    metric.update(torch.tensor([5.0]), groups=["only"])
    metric.update(torch.tensor([1.0, 2.0, 3.0]), groups=["same", "same", "same"])
    res = metric.compute()
    assert res["groups"]["only"] == 5.0
    assert res["groups"]["same"] == 6.0
    assert res["overall"] == 11.0


def test_new_keys_discovered_at_runtime():
    """Keys appearing in later batches create groups automatically, without prior declaration."""
    metric = GroupwiseWrapper(SumMetric())
    metric.update(torch.tensor([1.0, 2.0]), groups=["a", "a"])
    assert metric.group_keys == ["a"]
    metric.update(torch.tensor([3.0, 4.0]), groups=["b", "c"])
    assert metric.group_keys == ["a", "b", "c"]
    res = metric.compute()
    assert set(res["groups"]) == {"a", "b", "c"}
    assert res["groups"]["a"] == 3.0


def test_key_type_normalization():
    """Equivalent keys of different container/scalar types map to the same group; int and str differ."""
    metric = GroupwiseWrapper(SumMetric())
    metric.update(torch.tensor([1.0, 2.0, 3.0, 4.0]), groups=torch.tensor([0, 1, 1, 0]))  # int tensor
    metric.update(torch.tensor([10.0, 20.0]), groups=[True, False])  # bool -> int
    metric.update(torch.tensor([100.0]), groups=[np.int64(1)])  # numpy integer
    metric.update(torch.tensor([1000.0]), groups=["1"])  # string "1" is a different group than int 1
    res = metric.compute()
    assert set(res["groups"]) == {0, 1, "1"}
    assert res["groups"][0] == 1.0 + 4.0 + 20.0
    assert res["groups"][1] == 2.0 + 3.0 + 10.0 + 100.0
    assert res["groups"]["1"] == 1000.0


def test_keys_with_special_characters():
    """Keys containing dots, underscores and empty strings must not collide after encoding."""
    metric = GroupwiseWrapper(SumMetric())
    keys = ["a.b", "a_db", "a__db", "", "_", "a" * 50]
    for i, key in enumerate(keys):
        metric.update(torch.tensor([float(i + 1)]), groups=[key])
    res = metric.compute()
    assert set(res["groups"]) == set(keys)
    for i, key in enumerate(keys):
        assert res["groups"][key] == float(i + 1)


@pytest.mark.parametrize(
    ("groups", "error", "match"),
    [
        (None, TorchMetricsUserError, "Missing required keyword argument `groups`"),
        (["a"], ValueError, "one entry per sample"),
        (["a", None], TypeError, "index 1"),
        (["a", 1.5], TypeError, "index 1"),
        (["a", float("nan")], TypeError, "index 1"),
        (["a", ["b"]], TypeError, "index 1"),
        ("ab", TypeError, "list, tuple or 1-dimensional tensor"),
        ({"a": 0}, TypeError, "list, tuple or 1-dimensional tensor"),
        (torch.tensor([[0, 1], [2, 3]]), ValueError, "1-dimensional"),
    ],
)
def test_invalid_groups(groups, error, match):
    """Invalid `groups` arguments raise a descriptive error and never silently form a group."""
    metric = GroupwiseWrapper(SumMetric())
    with pytest.raises(error, match=match):
        metric.update(torch.tensor([1.0, 2.0]), groups=groups)
    assert metric.group_keys == []


def test_invalid_metric_argument():
    """Non-metrics, collections and full-state metrics are rejected at construction."""
    with pytest.raises(TypeError, match=r"instance of `torchmetrics\.Metric`"):
        GroupwiseWrapper("not a metric")
    with pytest.raises(TypeError, match="MetricCollection"):
        GroupwiseWrapper(MetricCollection([SumMetric()]))

    class _FullStateMetric(Metric):
        full_state_update = True

        def __init__(self) -> None:
            super().__init__()
            self.add_state("x", torch.tensor(0.0), dist_reduce_fx="sum")

        def update(self, x):
            self.x += x.sum()

        def compute(self):
            return self.x

    with pytest.raises(ValueError, match="full_state_update=True"):
        GroupwiseWrapper(_FullStateMetric())


@pytest.mark.parametrize("max_groups", [0, -1, 2.5, "3"])
def test_invalid_max_groups(max_groups):
    """Invalid `max_groups` values are rejected."""
    with pytest.raises(ValueError, match="max_groups"):
        GroupwiseWrapper(SumMetric(), max_groups=max_groups)


def test_invalid_on_exceed():
    """Invalid `on_exceed` values are rejected."""
    with pytest.raises(ValueError, match="on_exceed"):
        GroupwiseWrapper(SumMetric(), on_exceed="ignore")


def test_max_groups_error_reports_keys_and_is_atomic():
    """Exceeding the limit with on_exceed='error' fails loudly, names the keys, and changes no state."""
    metric = GroupwiseWrapper(SumMetric(), max_groups=2)
    metric.update(torch.tensor([1.0, 2.0]), groups=["a", "b"])
    with pytest.raises(TorchMetricsUserError, match="max_groups=2") as excinfo:
        metric.update(torch.tensor([3.0, 4.0, 5.0]), groups=["c", "d", "a"])
    assert "'c'" in str(excinfo.value)
    assert "'d'" in str(excinfo.value)
    # state is untouched by the rejected batch: neither groups nor overall changed
    assert metric.group_keys == ["a", "b"]
    res = metric.compute()
    assert set(res["groups"]) == {"a", "b"}
    assert res["overall"] == 3.0


def test_max_groups_overflow_degrades_and_reports():
    """on_exceed='overflow' diverts excess keys to a shared overflow group and reports them."""
    metric = GroupwiseWrapper(SumMetric(), max_groups=2, on_exceed="overflow")
    with pytest.warns(UserWarning, match="diverted"):
        metric.update(torch.tensor([1.0, 2.0, 3.0, 4.0]), groups=["a", "b", "c", "d"])
    assert metric.overflow_keys == ["c", "d"]
    # an already-diverted key keeps going to the overflow group without a new warning
    metric.update(torch.tensor([10.0]), groups=["c"])
    res = metric.compute()
    assert set(res["groups"]) == {"a", "b", "__overflow__"}
    assert res["groups"]["a"] == 1.0
    assert res["groups"]["b"] == 2.0
    assert res["groups"]["__overflow__"] == 3.0 + 4.0 + 10.0
    # nothing is silently dropped: groups + overflow still add up to the overall result
    assert res["groups"]["a"] + res["groups"]["b"] + res["groups"]["__overflow__"] == res["overall"]


def test_failed_update_does_not_pollute_other_groups():
    """An update that fails inside the wrapped metric leaves previously accumulated groups intact."""
    metric = GroupwiseWrapper(MeanMetric(nan_strategy="error"))
    metric.update(torch.tensor([1.0, 2.0]), groups=["a", "b"])
    with pytest.raises(RuntimeError, match="nan"):
        metric.update(torch.tensor([float("nan"), 5.0]), groups=["a", "b"])
    res = metric.compute()
    assert res["groups"]["a"] == 1.0
    assert res["groups"]["b"] == 2.0
    assert res["overall"] == 1.5


def test_forward_returns_batch_values_and_accumulates():
    """Forward returns per-batch group values and accumulates global state."""
    metric = GroupwiseWrapper(MeanMetric())
    out = metric(torch.tensor([1.0, 2.0, 3.0]), groups=["a", "b", "a"])
    assert out["overall"] == 2.0
    assert out["groups"] == {"a": 2.0, "b": 2.0}
    out = metric(torch.tensor([10.0]), groups=["c"])
    assert out["groups"] == {"c": 10.0}
    res = metric.compute()
    assert torch.allclose(res["overall"], torch.tensor(4.0))
    assert set(res["groups"]) == {"a", "b", "c"}


def test_reset_clears_groups():
    """Reset clears dynamically created groups and the diverted-key set."""
    metric = GroupwiseWrapper(SumMetric())
    metric.update(torch.tensor([1.0, 2.0]), groups=["a", "b"])
    metric.reset()
    assert metric.group_keys == []
    assert metric.overflow_keys == []
    metric.update(torch.tensor([5.0]), groups=["c"])
    res = metric.compute()
    assert set(res["groups"]) == {"c"}
    assert res["overall"] == 5.0


def test_state_dict_resume_matches_uninterrupted(tmp_path):
    """Save, restore and continue accumulating must give the same result as an uninterrupted run."""
    batches = _make_batches(num_batches=6, batch_size=8, keys=("alpha", "beta"), seed=7)
    # a new key appears only after the checkpoint
    batches.append((torch.tensor([1.0, 2.0]), ["gamma", "alpha"]))

    interrupted = GroupwiseWrapper(MeanMetric())
    interrupted.persistent(True)
    for values, groups in batches[:3]:
        interrupted.update(values, groups=groups)

    ckpt = tmp_path / "metric.pt"
    torch.save(interrupted.state_dict(), ckpt)

    restored = GroupwiseWrapper(MeanMetric())
    restored.persistent(True)
    restored.load_state_dict(torch.load(ckpt, weights_only=True))
    assert restored.group_keys == interrupted.group_keys

    reference = GroupwiseWrapper(MeanMetric())
    for values, groups in batches:
        reference.update(values, groups=groups)
    # the restored metric continues from the checkpoint with the remaining batches only
    for values, groups in batches[3:]:
        restored.update(values, groups=groups)

    res, ref = restored.compute(), reference.compute()
    assert set(res["groups"]) == set(ref["groups"]) == {"alpha", "beta", "gamma"}
    for key in ref["groups"]:
        assert torch.allclose(res["groups"][key], ref["groups"][key], atol=1e-6), f"group {key}"
    assert torch.allclose(res["overall"], ref["overall"], atol=1e-6)


def test_high_cardinality():
    """A few thousand distinct keys stay within sane time/memory bounds."""
    n = 5000
    keys = [f"key_{i}" for i in range(n)]
    metric = GroupwiseWrapper(SumMetric())
    metric.update(torch.ones(n), groups=keys)
    res = metric.compute()
    assert len(res["groups"]) == n
    assert res["overall"] == float(n)


def _test_ddp_disjoint_keys(rank):
    """Two processes with (partially) disjoint key sets; merged result must equal single-process run."""
    setup_ddp(rank, NUM_PROCESSES)
    metric = GroupwiseWrapper(SumMetric())
    if rank == 0:
        metric.update(torch.tensor([1.0, 2.0, 3.0, 4.0]), groups=["shared", "alpha", "shared", "alpha"])
    else:
        metric.update(torch.tensor([10.0, 20.0, 30.0]), groups=["beta", "shared", "beta"])
    res = metric.compute()

    # keys only seen on the other rank must not be lost, and shared keys must not be double counted
    assert set(res["groups"]) == {"alpha", "beta", "shared"}
    assert res["groups"]["alpha"] == 6.0
    assert res["groups"]["beta"] == 40.0
    assert res["groups"]["shared"] == 24.0
    assert res["overall"] == 70.0

    # merged result must equal a plain single-process reduction over all data
    all_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0])
    all_groups = ["shared", "alpha", "shared", "alpha", "beta", "shared", "beta"]
    assert res["overall"] == all_values.sum()
    for key in res["groups"]:
        mask = torch.tensor([g == key for g in all_groups])
        assert res["groups"][key] == all_values[mask].sum()


def _test_ddp_mean_weighted_merge(rank):
    """Mean over groups with different per-rank counts must merge with correct weights."""
    setup_ddp(rank, NUM_PROCESSES)
    metric = GroupwiseWrapper(MeanMetric())
    if rank == 0:
        metric.update(torch.tensor([1.0, 2.0, 3.0, 4.0]), groups=["shared", "alpha", "shared", "alpha"])
    else:
        metric.update(torch.tensor([10.0, 20.0, 30.0]), groups=["beta", "shared", "beta"])
    res = metric.compute()
    assert torch.allclose(res["groups"]["alpha"], torch.tensor(3.0))
    assert torch.allclose(res["groups"]["beta"], torch.tensor(20.0))
    assert torch.allclose(res["groups"]["shared"], torch.tensor(8.0))
    assert torch.allclose(res["overall"], torch.tensor(10.0))


@pytest.mark.DDP
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
@pytest.mark.parametrize("process", [_test_ddp_disjoint_keys, _test_ddp_mean_weighted_merge])
def test_groupwise_ddp(process):
    """Test that distributed merging of dynamic groups matches the single-process result."""
    pytest.pool.map(process, range(NUM_PROCESSES))
