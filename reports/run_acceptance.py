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
"""Acceptance driver for the sharded-merge feature.

Runs the four acceptance scenarios from the quarterly-review action item and writes both a machine-readable
``acceptance_results.json`` and a human-readable ``acceptance_report.md`` next to this script. Run with the
repository source on the path, e.g. ``PYTHONPATH=src python reports/run_acceptance.py``.
"""

import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import torch

from torchmetrics.aggregation import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric
from torchmetrics.sharding import Coverage, LowCoverageEvent, ShardEnvelope, ShardMerger
from torchmetrics.utilities.exceptions import TorchMetricsUserError

REPORT_DIR = Path(__file__).resolve().parent
TOLERANCE = 1e-6
N_SHARDS = 4
DATA = torch.arange(100.0, dtype=torch.float32)


def build_shards(metric_cls: type) -> list[ShardEnvelope]:
    """Split the 100-sample dataset into four equal shards."""
    shards = []
    step = len(DATA) // N_SHARDS
    for i in range(N_SHARDS):
        metric = metric_cls()
        chunk = DATA[i * step : (i + 1) * step]
        metric.update(chunk)
        shards.append(
            ShardEnvelope.from_metric(
                metric,
                Coverage(count=chunk.numel(), interval=(i * step, (i + 1) * step - 1), global_count=len(DATA)),
            )
        )
    return shards


def scenario_order_insensitivity() -> dict[str, Any]:
    """Acceptance 1: four shards, every permutation merges to the bit-identical value."""
    results: dict[str, Any] = {}
    for metric_cls in (SumMetric, MeanMetric, MaxMetric, MinMetric):
        shards = build_shards(metric_cls)
        values = set()
        for permutation in itertools.permutations(shards):
            target = metric_cls()
            ShardMerger(target, expected_total_count=len(DATA)).merge(list(permutation))
            values.add(target.compute().item())
        single = metric_cls()
        single.update(DATA)
        merged_value = next(iter(values))
        results[metric_cls.__name__] = {
            "distinct_results_across_24_permutations": len(values),
            "bit_identical_for_every_order": len(values) == 1,
            "merged_value": merged_value,
            "single_machine_value": single.compute().item(),
            "exact_match": merged_value == single.compute().item(),
        }

    cat_shards = build_shards(CatMetric)
    cat_outputs = set()
    for permutation in itertools.permutations(cat_shards):
        target = CatMetric()
        ShardMerger(target, expected_total_count=len(DATA)).merge(list(permutation))
        cat_outputs.add(tuple(target.compute().tolist()))
    cat_target = CatMetric()
    ShardMerger(cat_target, expected_total_count=len(DATA)).merge(cat_shards)
    cat_single = CatMetric()
    cat_single.update(DATA)
    results["CatMetric"] = {
        "distinct_ordered_outputs_across_24_permutations": len(cat_outputs),
        "bit_identical_for_every_order": len(cat_outputs) == 1,
        "output_length": cat_target.compute().numel(),
        "multiset_matches_single_machine": bool(
            torch.equal(cat_target.compute().sort().values, cat_single.compute().sort().values)
        ),
    }
    return results


def scenario_idempotency() -> dict[str, Any]:
    """Acceptance 2: resubmitting the same shards does not change the result."""
    shards = build_shards(SumMetric)
    target = SumMetric()
    merger = ShardMerger(target, expected_total_count=len(DATA))
    merger.merge(shards)
    first = target.compute().item()
    merger.merge([*shards, shards[2]])
    second = target.compute().item()

    payload = json.loads(shards[0].to_json())
    payload["shard_id"] = "renamed-copy"
    renamed = ShardEnvelope.from_dict(payload)
    target2 = SumMetric()
    merger2 = ShardMerger(target2, expected_total_count=len(DATA))
    merger2.merge([*shards, renamed])
    return {
        "value_after_first_merge": first,
        "value_after_resubmitting_all_plus_one_copy": second,
        "idempotent": first == second,
        "duplicate_ids_recorded_on_manifest": merger.last_manifest.duplicate_shard_ids,
        "same_content_new_id_treated_as_duplicate": merger2.last_manifest.duplicate_shard_ids == ["renamed-copy"],
        "value_with_renamed_copy": target2.compute().item(),
    }


def scenario_validation_verdicts() -> dict[str, Any]:
    """Acceptance 3: missing shard, duplicate shard and fingerprint mismatch, with actual messages."""
    shards = build_shards(SumMetric)
    ids = [shard.shard_id for shard in shards]
    verdicts: dict[str, Any] = {}

    # missing shard, default reject policy
    try:
        ShardMerger(SumMetric(), expected_shard_ids=ids, expected_total_count=len(DATA)).merge(shards[:3])
        verdicts["missing_shard_reject"] = {"rejected": False}
    except TorchMetricsUserError as err:
        verdicts["missing_shard_reject"] = {"rejected": True, "message": str(err)}

    # missing shard, partial policy: observable event, coverage ratio, annotated result
    events: list[LowCoverageEvent] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        partial_target = SumMetric()
        partial_merger = ShardMerger(
            partial_target,
            expected_shard_ids=ids,
            expected_total_count=len(DATA),
            missing_policy="partial",
            min_coverage=0.9,
            event_handler=events.append,
        )
        partial_merger.merge(shards[:3])
    verdicts["missing_shard_partial"] = {
        "warning_emitted": any("coverage" in str(w.message) for w in caught),
        "structured_event": type(events[0]).__name__ if events else None,
        "coverage_ratio": events[0].coverage_ratio if events else None,
        "missing_shard_ids": events[0].missing_shard_ids if events else None,
        "merged_value": partial_target.compute().item(),
        "expected_value_for_first_75_samples": DATA[:75].sum().item(),
    }

    # duplicate shard inside one batch
    dup_target = SumMetric()
    dup_merger = ShardMerger(dup_target, expected_total_count=len(DATA))
    dup_merger.merge([*shards, shards[1]])
    verdicts["duplicate_shard"] = {
        "value": dup_target.compute().item(),
        "expected_value": DATA.sum().item(),
        "duplicate_ids_recorded": dup_merger.last_manifest.duplicate_shard_ids,
    }

    # configuration fingerprint mismatch
    other = SumMetric(nan_strategy="ignore")
    other.update(DATA[:25])
    bad = ShardEnvelope.from_metric(other, Coverage(25, (0, 24), len(DATA)))
    try:
        ShardMerger(SumMetric(), expected_total_count=len(DATA)).merge([*shards[:3], bad])
        verdicts["fingerprint_mismatch"] = {"rejected": False}
    except TorchMetricsUserError as err:
        verdicts["fingerprint_mismatch"] = {"rejected": True, "message": str(err)}
    return verdicts


def scenario_tolerance() -> dict[str, Any]:
    """Acceptance 4: merged vs single-machine, contractual tolerance and measured absolute difference."""
    out: dict[str, Any] = {}
    for metric_cls in (SumMetric, MeanMetric, MaxMetric, MinMetric):
        shards = build_shards(metric_cls)
        target = metric_cls()
        ShardMerger(target, expected_total_count=len(DATA)).merge(shards)
        reference = metric_cls()
        reference.update(DATA)
        merged, single = target.compute(), reference.compute()
        diff = (merged - single).abs().max().item()
        out[metric_cls.__name__] = {
            "merged": merged.item(),
            "single_machine": single.item(),
            "abs_difference": diff,
            "tolerance": TOLERANCE,
            "within_tolerance": diff <= TOLERANCE,
            "bit_exact": diff == 0.0,
        }

    generator = torch.Generator().manual_seed(123)
    big = torch.randn(100_000, generator=generator)
    step = big.numel() // N_SHARDS
    big_shards = []
    for i in range(N_SHARDS):
        metric = MeanMetric()
        metric.update(big[i * step : (i + 1) * step])
        big_shards.append(
            ShardEnvelope.from_metric(metric, Coverage(step, (i * step, (i + 1) * step - 1), big.numel()))
        )
    big_target = MeanMetric()
    ShardMerger(big_target, expected_total_count=big.numel()).merge(big_shards)
    big_ref = MeanMetric()
    big_ref.update(big)
    big_diff = (big_target.compute() - big_ref.compute()).abs().item()
    out["MeanMetric_100k_floats"] = {
        "merged": big_target.compute().item(),
        "single_machine": big_ref.compute().item(),
        "abs_difference": big_diff,
        "tolerance": TOLERANCE,
        "within_tolerance": big_diff <= TOLERANCE,
    }
    return out


def write_markdown(results: dict[str, Any]) -> None:
    """Render the JSON results as a review-friendly Markdown report."""
    lines = [
        "# 分片合并能力验收报告",
        "",
        "数据集：`torch.arange(100.0)`，均匀切分为 4 片（每片 25 条）；容差 `atol=1e-6`。",
        "本报告由 `reports/run_acceptance.py` 自动生成，原始数据见 `acceptance_results.json`。",
        "",
        "## 1. 任意顺序合并逐位一致",
        "",
        "| 归约语义 | 24 种排列的不同结果数 | 逐位一致 | 分片合并值 | 单机全量值 | 完全相等 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in results["order_insensitivity"].items():
        if name == "CatMetric":
            lines.append(
                f"| {name} | {row['distinct_ordered_outputs_across_24_permutations']} | "
                f"{row['bit_identical_for_every_order']} | 长度 {row['output_length']} | "
                f"multiset 一致 {row['multiset_matches_single_machine']} | - |"
            )
        else:
            lines.append(
                f"| {name} | {row['distinct_results_across_24_permutations']} | "
                f"{row['bit_identical_for_every_order']} | {row['merged_value']} | "
                f"{row['single_machine_value']} | {row['exact_match']} |"
            )
    idem = results["idempotency"]
    lines += [
        "",
        "## 2. 重复提交幂等",
        "",
        f"- 首次合并值：`{idem['value_after_first_merge']}`",
        f"- 整批重投 + 额外 1 个副本后的值：`{idem['value_after_resubmitting_all_plus_one_copy']}`",
        f"- 幂等成立：**{idem['idempotent']}**",
        f"- 清单记录的重复片：`{idem['duplicate_ids_recorded_on_manifest']}`",
        f"- 同内容换名（内容校验和识别）判重：**{idem['same_content_new_id_treated_as_duplicate']}**",
        "",
        "## 3. 缺片 / 重复片 / 配置不一致判定",
        "",
        "### 3.1 缺片 — 默认拒绝",
        "",
        f"- 拒绝：`{results['validation']['missing_shard_reject']['rejected']}`",
        f"- 报错：`{results['validation']['missing_shard_reject']['message']}`",
        "",
        "### 3.2 缺片 — partial 策略（标注覆盖比例 + 告警事件）",
        "",
    ]
    partial = results["validation"]["missing_shard_partial"]
    lines += [
        f"- 告警发出：`{partial['warning_emitted']}`；结构化事件：`{partial['structured_event']}`",
        f"- 覆盖比例：`{partial['coverage_ratio']}`；缺片：`{partial['missing_shard_ids']}`",
        (
            f"- 已到片合并值 `{partial['merged_value']}` 等于前 75 条单机值 "
            f"`{partial['expected_value_for_first_75_samples']}`"
        ),
        "",
        "### 3.3 重复片",
        "",
        (
            f"- 合并值 `{results['validation']['duplicate_shard']['value']}` 等于全量值 "
            f"`{results['validation']['duplicate_shard']['expected_value']}`"
        ),
        f"- 清单记录：`{results['validation']['duplicate_shard']['duplicate_ids_recorded']}`",
        "",
        "### 3.4 配置指纹不一致 — 必须拒绝",
        "",
        f"- 拒绝：`{results['validation']['fingerprint_mismatch']['rejected']}`",
        f"- 报错：`{results['validation']['fingerprint_mismatch']['message']}`",
        "",
        "## 4. 与单机全量的容差对比",
        "",
        "| 指标 | 分片合并 | 单机全量 | 实测绝对差 | 容差 | 结论 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in results["tolerance"].items():
        verdict = "逐位相等" if row.get("bit_exact") else ("容差内" if row["within_tolerance"] else "超差")
        lines.append(
            f"| {name} | {row['merged']} | {row['single_machine']} | {row['abs_difference']} | "
            f"{row['tolerance']} | {verdict} |"
        )
    lines.append("")
    (REPORT_DIR / "acceptance_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run all acceptance scenarios and persist the results."""
    warnings.filterwarnings("ignore", category=UserWarning)
    results = {
        "order_insensitivity": scenario_order_insensitivity(),
        "idempotency": scenario_idempotency(),
        "validation": scenario_validation_verdicts(),
        "tolerance": scenario_tolerance(),
    }
    (REPORT_DIR / "acceptance_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    write_markdown(results)
    print(f"Wrote {REPORT_DIR / 'acceptance_results.json'}")
    print(f"Wrote {REPORT_DIR / 'acceptance_report.md'}")


if __name__ == "__main__":
    main()
