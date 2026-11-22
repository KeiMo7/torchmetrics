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
"""Tests for the streaming, mergeable, serializable :class:`QuantileMetric` and its t-digest core."""
from __future__ import annotations

from functools import partial

import pytest
import torch

from torchmetrics.aggregation import QuantileMetric
from torchmetrics.functional.aggregation import quantile, quantile_rank_error_bound
from unittests import NUM_PROCESSES, USE_PYTEST_POOL
from unittests._helpers import _IS_WINDOWS, seed_all
from unittests.conftest import setup_ddp

seed_all(42)

_Q = torch.tensor([0.001, 0.01, 0.5, 0.95, 0.99, 0.999], dtype=torch.float64)


def _weighted_quantile_reference(values: torch.Tensor, weights: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Exact weighted quantile (linear interpolation on the weighted CDF), matching the sketch exact path."""
    order = torch.argsort(values)
    values, weights = values[order], weights[order]
    total = weights.sum()
    cum = torch.cumsum(weights, dim=0)
    target = q * total
    idx = torch.searchsorted(cum, target, right=False).clamp(min=0, max=values.numel() - 1)
    prev = cum[idx] - weights[idx]
    frac = ((target - prev) / weights[idx]).clamp(min=0.0, max=1.0)
    prev_value = values[torch.clamp(idx - 1, min=0)]
    return prev_value + frac * (values[idx] - prev_value)


def _empirical_rank(values: torch.Tensor, estimates: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Empirical (weighted) CDF rank of every estimated value on the full raw sample."""
    order = torch.argsort(values)
    values = values[order]
    if weights is None:
        return torch.searchsorted(values, estimates, right=True).double() / values.numel()
    weights = weights[order]
    ranks = [weights[values <= est].sum() / weights.sum() for est in estimates]
    return torch.stack(ranks).double()


# --------------------------------------------------------------------- exact path
@pytest.mark.parametrize("n", [1, 2, 5, 37, 99])
def test_exact_path_matches_torch_quantile(n):
    """Below the buffer threshold the metric must return exact type-7 quantiles and a zero error bound."""
    metric = QuantileMetric(_Q.tolist(), compression=50, buffer_size=100).set_dtype(torch.float64)
    data = torch.randn(n, dtype=torch.float64)
    metric.update(data)
    assert metric._sketch.is_exact
    assert metric.compute().dtype == torch.float64
    assert torch.allclose(metric.compute(), torch.quantile(data, _Q), atol=1e-12, rtol=0)
    assert metric.rank_error_bound().item() == 0.0


def test_exact_path_weighted_matches_reference():
    """The weighted exact path must match the independent weighted-CDF interpolation reference."""
    metric = QuantileMetric([0.25, 0.5, 0.75], compression=50, buffer_size=500).set_dtype(torch.float64)
    data = torch.randn(120, dtype=torch.float64)
    weights = torch.rand(120, dtype=torch.float64) + 0.05
    metric.update(data, weights)
    assert metric._sketch.is_exact
    ref = _weighted_quantile_reference(data, weights, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    assert torch.allclose(metric.compute(), ref, atol=1e-12, rtol=0)


# ------------------------------------------------------------- input shapes / batches
def test_scalar_flat_and_multidim_inputs_equivalent():
    """A scalar, a flat batch and a multi-dimensional tensor flattened must produce identical results."""
    data = torch.randn(64, dtype=torch.float64)
    m_scalar = QuantileMetric(0.5, buffer_size=500).set_dtype(torch.float64)
    for v in data:
        m_scalar.update(v.item())
    m_flat = QuantileMetric(0.5, buffer_size=500).set_dtype(torch.float64)
    m_flat.update(data)
    m_md = QuantileMetric(0.5, buffer_size=500).set_dtype(torch.float64)
    m_md.update(data.reshape(8, 8))
    assert torch.allclose(m_scalar.compute(), m_flat.compute())
    assert torch.allclose(m_flat.compute(), m_md.compute())


@pytest.mark.parametrize("batch_size", [1, 3, 17, 256, 10_000])
def test_variable_batch_sizes(batch_size):
    """Continuous accumulation with batch sizes from 1 to ten thousand must stay within the declared bound."""
    metric = QuantileMetric(_Q.tolist(), compression=100, buffer_size=500).set_dtype(torch.float64)
    chunks, total = [], 20_000
    pos = 0
    while pos < total:
        size = min(batch_size, total - pos)
        chunk = torch.randn(size, dtype=torch.float64)
        metric.update(chunk)
        chunks.append(chunk)
        pos += size
    full = torch.cat(chunks)
    estimates = metric.compute()
    bound = metric.rank_error_bound().item()
    ranks = _empirical_rank(full, estimates)
    assert torch.all((ranks - _Q).abs() <= bound + 1e-12)


def test_scalar_and_per_sample_weights():
    """A scalar broadcast weight and an explicit identical weight vector must be equivalent."""
    data = torch.randn(300, dtype=torch.float64)
    m_scalar = QuantileMetric([0.5, 0.9], buffer_size=500).set_dtype(torch.float64)
    m_scalar.update(data, 0.7)
    m_vector = QuantileMetric([0.5, 0.9], buffer_size=500).set_dtype(torch.float64)
    m_vector.update(data, torch.full((300,), 0.7, dtype=torch.float64))
    assert torch.allclose(m_scalar.compute(), m_vector.compute())


def test_multiple_quantiles_one_state():
    """All requested quantiles share a single sketch; one accumulation serves every evaluation."""
    metric = QuantileMetric([0.5, 0.95, 0.99], compression=80).set_dtype(torch.float64)
    metric.update(torch.randn(2000, dtype=torch.float64))
    out = metric.compute()
    assert out.shape == (3,)
    # a single compressed sketch backs the three values
    assert int(metric.centroid_count.item()) > 0
    assert metric.centroid_mean.numel() == 80


# ----------------------------------------------------------- rank bound correctness
@pytest.mark.parametrize("compression", [20, 50, 100, 200])
def test_rank_error_bound_holds_on_long_tail(compression):
    """The empirical rank error must never exceed the bound exposed to the caller, incl. extreme quantiles."""
    metric = QuantileMetric(_Q.tolist(), compression=compression, buffer_size=500).set_dtype(torch.float64)
    # long-tailed mixture: mostly near zero, occasional large values
    data = torch.cat(
        [
            torch.randn(40_000, dtype=torch.float64),
            torch.randn(4_000, dtype=torch.float64) * 5.0,
            torch.randn(500, dtype=torch.float64) * 20.0,
        ]
    )
    metric.update(data)
    bound = metric.rank_error_bound().item()
    ranks = _empirical_rank(data, metric.compute())
    assert torch.all((ranks - _Q).abs() <= bound + 1e-12)
    # hard structural upper bound of the fixed grid: the widest gap between adjacent grid edges is pi / delta
    assert bound <= torch.pi / compression + 1e-9


def test_rank_error_bound_holds_with_weights():
    """The weighted rank error must stay within the exposed bound."""
    metric = QuantileMetric(_Q.tolist(), compression=80, buffer_size=500).set_dtype(torch.float64)
    data = torch.randn(20_000, dtype=torch.float64)
    weights = torch.rand(20_000, dtype=torch.float64) + 0.1
    metric.update(data, weights)
    bound = metric.rank_error_bound().item()
    ranks = _empirical_rank(data, metric.compute(), weights)
    assert torch.all((ranks - _Q).abs() <= bound + 1e-12)


# -------------------------------------------------------------- degenerate inputs
def test_all_values_equal():
    """A degenerate constant stream returns that constant for every quantile."""
    metric = QuantileMetric([0.001, 0.5, 0.999], compression=50).set_dtype(torch.float64)
    metric.update(torch.full((5000,), 3.0, dtype=torch.float64))
    assert torch.all(metric.compute() == 3.0)


def test_majority_duplicates():
    """A stream dominated by duplicate values keeps the dominant value at the median."""
    metric = QuantileMetric([0.5, 0.9], compression=50, buffer_size=100).set_dtype(torch.float64)
    metric.update(torch.cat([torch.ones(4000, dtype=torch.float64), torch.full((1000,), 2.0, dtype=torch.float64)]))
    out = metric.compute()
    # the median of an 80/20 mixture must be the dominant value
    assert out[0].item() == 1.0
    assert metric.rank_error_bound().item() > 0.0


def test_extreme_quantiles_anchored():
    """Quantiles very close to the endpoints stay within the rank bound."""
    metric = QuantileMetric([0.001, 0.999], compression=100, buffer_size=500).set_dtype(torch.float64)
    data = torch.randn(50_000, dtype=torch.float64)
    metric.update(data)
    bound = metric.rank_error_bound().item()
    ranks = _empirical_rank(data, metric.compute())
    assert torch.all((ranks - torch.tensor([0.001, 0.999], dtype=torch.float64)).abs() <= bound + 1e-12)


# ------------------------------------------------------------- merge associativity
@pytest.mark.parametrize("num_shards", [2, 4, 8])
def test_merge_order_independence(num_shards):
    """Merging the same shards in every permutation must agree within the declared rank bound."""
    seed_all(7)
    shards = []
    full = []
    for _ in range(num_shards):
        chunk = torch.randn(3000, dtype=torch.float64)
        full.append(chunk)
        shard = QuantileMetric(_Q.tolist(), compression=80, buffer_size=300).set_dtype(torch.float64)
        for i in range(0, chunk.numel(), 137):
            shard.update(chunk[i : i + 137])
        shards.append(shard)
    full = torch.cat(full)

    import itertools

    results = []
    for perm in itertools.permutations(range(num_shards)):
        acc = QuantileMetric(_Q.tolist(), compression=80, buffer_size=300).set_dtype(torch.float64)
        for i in perm:
            acc.merge_state(shards[i])
        results.append(acc.compute())
    bound = acc.rank_error_bound().item()
    # every merge order must land within the declared rank error bound of the requested quantiles
    for result in results:
        ranks = _empirical_rank(full, result)
        assert torch.all((ranks - _Q).abs() <= bound + 1e-12)
    # deterministic clustering keeps the value spread a small fraction of the data scale
    spread = max((r - results[0]).abs().max().item() for r in results)
    assert spread < 0.2 * full.std().item()


def test_merge_exact_path_bit_identical():
    """Two still-exact sketches whose union fits the buffer merge into an exact, order-independent result."""
    a = QuantileMetric(0.5, compression=50, buffer_size=1000).set_dtype(torch.float64)
    b = QuantileMetric(0.5, compression=50, buffer_size=1000).set_dtype(torch.float64)
    xa, xb = torch.randn(200, dtype=torch.float64), torch.randn(200, dtype=torch.float64)
    a.update(xa)
    b.update(xb)
    a.merge_state(b)
    assert a._sketch.is_exact
    assert torch.allclose(a.compute(), torch.quantile(torch.cat([xa, xb]), torch.tensor(0.5, dtype=torch.float64)))


def test_different_partitions_same_result():
    """Splitting one dataset into 1, 4 or 16 batches must give bound-consistent estimates."""
    data = torch.randn(16_000, dtype=torch.float64)
    outputs, bounds = [], []
    for splits in (1, 4, 16):
        metric = QuantileMetric(_Q.tolist(), compression=100, buffer_size=500).set_dtype(torch.float64)
        for chunk in torch.tensor_split(data, splits):
            metric.update(chunk)
        outputs.append(metric.compute())
        bounds.append(metric.rank_error_bound().item())
    # every partition is within its declared rank bound of the true quantiles
    for out, bound in zip(outputs, bounds):
        ranks = _empirical_rank(data, out)
        assert torch.all((ranks - _Q).abs() <= bound + 1e-12)
    # and the partitions mutually agree within the rank bound (value differences at steep tails are expected)
    for out in outputs[1:]:
        ranks = _empirical_rank(data, out)
        ref_ranks = _empirical_rank(data, outputs[0])
        assert torch.all((ranks - ref_ranks).abs() <= max(bounds) + 1e-12)


# ------------------------------------------------------------- persistence
def test_save_load_resume_consistency(tmp_path):
    """A checkpoint restored in a fresh process and resumed must match an uninterrupted run."""
    data_a = torch.randn(5_000, dtype=torch.float64)
    data_b = torch.randn(5_000, dtype=torch.float64)

    uninterrupted = QuantileMetric(_Q.tolist(), compression=80, buffer_size=400).set_dtype(torch.float64)
    uninterrupted.update(data_a)
    uninterrupted.update(data_b)

    interrupted = QuantileMetric(_Q.tolist(), compression=80, buffer_size=400).set_dtype(torch.float64)
    interrupted.update(data_a)
    interrupted.persistent(True)
    path = tmp_path / "quantile_state.pth"
    torch.save(interrupted.state_dict(), path)

    restored = QuantileMetric(_Q.tolist(), compression=80, buffer_size=400).set_dtype(torch.float64)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    restored.update(data_b)

    assert torch.allclose(restored.compute(), uninterrupted.compute())
    assert restored.rank_error_bound().item() == uninterrupted.rank_error_bound().item()


def test_save_load_exact_path(tmp_path):
    """An exact-path checkpoint round-trips and stays exact after loading."""
    metric = QuantileMetric([0.5, 0.9], compression=50, buffer_size=500).set_dtype(torch.float64)
    metric.update(torch.randn(100, dtype=torch.float64))
    metric.persistent(True)
    path = tmp_path / "exact.pth"
    torch.save(metric.state_dict(), path)
    restored = QuantileMetric([0.5, 0.9], compression=50, buffer_size=500).set_dtype(torch.float64)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    assert restored._sketch.is_exact
    assert torch.allclose(restored.compute(), metric.compute())


def test_load_rejects_shape_mismatch(tmp_path):
    """Loading a state with different accuracy parameters is rejected."""
    metric = QuantileMetric(0.5, compression=50).set_dtype(torch.float64)
    metric.update(torch.randn(600, dtype=torch.float64))
    metric.persistent(True)
    path = tmp_path / "mismatch.pth"
    torch.save(metric.state_dict(), path)
    different = QuantileMetric(0.5, compression=80).set_dtype(torch.float64)
    with pytest.raises(ValueError, match=r"compression|buffer_size|shape"):
        different.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def test_cross_dtype_restore_is_safe(tmp_path):
    """Restoring a float64 state into a float32 metric converts safely with a warning."""
    metric = QuantileMetric(0.5, compression=50).set_dtype(torch.float64)
    metric.update(torch.randn(600, dtype=torch.float64))
    metric.persistent(True)
    path = tmp_path / "dtype.pth"
    torch.save(metric.state_dict(), path)
    restored = QuantileMetric(0.5, compression=50)  # float32 target
    with pytest.warns(UserWarning, match="safely converted"):
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    restored.update(torch.randn(100))
    assert restored.compute().dtype == torch.float32


# ------------------------------------------------------------- constant memory
def test_state_size_independent_of_sample_count():
    """The number of stored values must stay constant as n grows from 1k to 1m."""
    capacities = set()
    for n in (1_000, 100_000, 1_000_000):
        metric = QuantileMetric(0.5, compression=100, buffer_size=500)
        for start in range(0, n, 10_000):
            metric.update(torch.randn(min(10_000, n - start)))
        num_stored = sum(
            getattr(metric, name).numel()
            for name in (
                "buffer_values",
                "buffer_weights",
                "centroid_mean",
                "centroid_weight",
                "total_weight",
                "min_value",
                "max_value",
                "buffer_count",
                "centroid_count",
                "unit_weights",
            )
        )
        capacities.add(num_stored)
    assert len(capacities) == 1
    # exact fixed count: 2 * compression + 2 * buffer_size + 6 scalar states
    assert capacities.pop() == 2 * 100 + 2 * 500 + 6


# ------------------------------------------------------------------- error cases
@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf"), [0.5, 1.0]])
def test_invalid_quantile_raises(bad_q):
    """Quantiles outside the open interval raise a parameter-specific error."""
    with pytest.raises(ValueError, match="quantiles"):
        QuantileMetric(bad_q)


def test_empty_quantile_sequence_raises():
    """An empty quantile sequence is rejected."""
    with pytest.raises(ValueError, match="quantiles"):
        QuantileMetric([])


@pytest.mark.parametrize("bad_compression", [4, 9, 9.5, True])
def test_invalid_compression_raises(bad_compression):
    """Compression values below the minimum or non-integers are rejected."""
    with pytest.raises(ValueError, match="compression"):
        QuantileMetric(0.5, compression=bad_compression)


@pytest.mark.parametrize("bad_buffer", [0, -1, 2.5])
def test_invalid_buffer_size_raises(bad_buffer):
    """Non-positive or non-integer buffer sizes are rejected."""
    with pytest.raises(ValueError, match="buffer_size"):
        QuantileMetric(0.5, buffer_size=bad_buffer)


def test_negative_weight_raises():
    """Negative weights raise a parameter-specific error."""
    metric = QuantileMetric(0.5)
    with pytest.raises(ValueError, match="weight"):
        metric.update(torch.tensor([1.0, 2.0]), torch.tensor([0.5, -0.2]))


def test_nonfinite_weight_raises():
    """Non-finite weights raise a parameter-specific error."""
    metric = QuantileMetric(0.5)
    with pytest.raises(ValueError, match="weight"):
        metric.update(torch.tensor([1.0, 2.0]), torch.tensor([0.5, float("inf")]))


def test_empty_and_zero_weight_compute_raises():
    """Empty or fully zero-weight inputs raise instead of returning zero."""
    metric = QuantileMetric(0.5)
    with pytest.raises(RuntimeError, match="undefined"):
        metric.compute()
    metric.update(torch.randn(10), torch.zeros(10))
    with pytest.raises(RuntimeError, match="undefined"):
        metric.compute()


def test_single_sample():
    """A single positive-weight sample returns its value for every quantile."""
    metric = QuantileMetric([0.01, 0.5, 0.99], buffer_size=100).set_dtype(torch.float64)
    metric.update(torch.tensor([7.0], dtype=torch.float64))
    assert torch.all(metric.compute() == 7.0)


def test_merge_compression_mismatch_rejected():
    """Merging states with different compression is explicitly rejected."""
    a = QuantileMetric(0.5, compression=50)
    b = QuantileMetric(0.5, compression=80)
    a.update(torch.randn(600))
    b.update(torch.randn(600))
    with pytest.raises(ValueError, match="compression"):
        a.merge_state(b)


def test_merge_buffer_mismatch_rejected():
    """Merging states with different buffer size is explicitly rejected."""
    a = QuantileMetric(0.5, buffer_size=400)
    b = QuantileMetric(0.5, buffer_size=800)
    a.update(torch.randn(50))
    b.update(torch.randn(50))
    with pytest.raises(ValueError, match="buffer_size"):
        a.merge_state(b)


def test_nan_strategies():
    """All NaN strategies behave like the existing aggregation metrics."""
    data = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(RuntimeError):
        QuantileMetric(0.5, nan_strategy="error").update(data)
    ignored = QuantileMetric(0.5, nan_strategy="ignore").set_dtype(torch.float64)
    ignored.update(data)
    assert ignored.compute().item() == 2.0  # median of [1, 3]
    imputed = QuantileMetric(0.5, nan_strategy=2.0).set_dtype(torch.float64)
    imputed.update(data)
    assert imputed.compute().item() == 2.0  # median of [1, 2, 3]


# ------------------------------------------------------------------- functional
def test_functional_quantile_exact_and_bound():
    """The functional entry point returns exact values and a zero bound on a small batch."""
    data = torch.arange(11.0)
    out = quantile(data, [0.25, 0.5, 0.75])
    assert torch.allclose(out, torch.tensor([2.5, 5.0, 7.5]))
    assert quantile_rank_error_bound(data).item() == 0.0


def test_functional_quantile_compressed_bound_holds():
    """The functional compressed path respects the returned rank bound."""
    data = torch.randn(20_000)
    out = quantile(data, _Q.tolist(), compression=100, buffer_size=500)
    bound = quantile_rank_error_bound(data, compression=100, buffer_size=500).item()
    ranks = _empirical_rank(data, out)
    assert torch.all((ranks - _Q.float()).abs() <= bound + 1e-7)


# ------------------------------------------------------------- distributed sync
def _test_ddp_quantile_sync(rank, world_size):
    setup_ddp(rank, world_size)
    metric = QuantileMetric([0.5, 0.95], compression=80, buffer_size=300)
    torch.manual_seed(rank)
    local = torch.randn(2_000)
    metric.update(local)
    metric.sync()
    # every rank now holds the merged sketch of all ranks: total weight equals the global sample count
    assert int(metric.total_weight.item()) == 2_000 * world_size
    bound = metric.rank_error_bound().item()
    assert bound >= 0.0
    metric.unsync()


@pytest.mark.DDP
@pytest.mark.skipif(_IS_WINDOWS, reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
def test_ddp_quantile_sync():
    """The custom distributed reduction must gather and merge the fixed-size states across ranks."""
    pytest.pool.map(partial(_test_ddp_quantile_sync, world_size=NUM_PROCESSES), range(NUM_PROCESSES))


def test_ddp_sync_handles_rank_with_no_data():
    """A rank that received no data contributes an empty sketch without breaking the merge."""
    # emulated without a real process group: merge an empty sketch into a populated one
    populated = QuantileMetric(0.5, compression=50).set_dtype(torch.float64)
    populated.update(torch.randn(1000, dtype=torch.float64))
    empty = QuantileMetric(0.5, compression=50).set_dtype(torch.float64)
    snapshot = populated.compute().clone()
    populated.merge_state(empty.metric_state)
    assert torch.allclose(populated.compute(), snapshot)


# ------------------------------------------------------------------- reset / forward
def test_reset_clears_everything():
    """Reset empties the sketch, raises on compute, and leaves a reusable metric."""
    metric = QuantileMetric([0.5, 0.9], compression=50)
    metric.update(torch.randn(2000))
    metric.reset()
    assert metric._sketch.is_empty
    assert metric._sketch.is_exact
    assert metric.total_weight.item() == 0
    with pytest.raises(RuntimeError, match="undefined"):
        metric.compute()
    # usable again after reset
    metric.update(torch.tensor([1.0, 2.0, 3.0]))
    assert metric.compute().shape == (2,)


def test_forward_matches_update_compute():
    """Forward returns the same result as an explicit update then compute."""
    metric = QuantileMetric(0.5, buffer_size=500).set_dtype(torch.float64)
    data = torch.randn(200, dtype=torch.float64)
    forward_out = metric(data)
    expected = QuantileMetric(0.5, buffer_size=500).set_dtype(torch.float64)
    expected.update(data)
    assert torch.allclose(forward_out, expected.compute())
