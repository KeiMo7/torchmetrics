# 分片合并能力设计说明（torchmetrics.sharding）

## 0. 定位与边界

独立、opt-in 的子包 `src/torchmetrics/sharding/`。**不修改** `Metric` 基类与任何既有指标
（`git diff --name-only` 中跟踪文件只有 `pyproject.toml` 的一条 lint 豁免）。
不引入新的运行时依赖：仅用标准库（`hashlib`、`json`、`dataclasses`、`uuid`、`warnings`）与 torch。
不用这套能力时，`import torchmetrics` 不会加载本包，单进程累计与分布式 `sync` 路径物理上不变。

模块划分：

| 文件 | 职责 |
| --- | --- |
| `envelope.py` | 片信封 `ShardEnvelope`、`Coverage`、配置指纹、内容校验和、JSON 序列化 |
| `validation.py` | 合并前校验清单与 `ValidationReport` |
| `reducers.py` | 求和/均值/极值/拼接四种归约语义及各自一致性检查 |
| `merger.py` | `ShardMerger`、两阶段提交、缺片策略、审计清单 `MergeManifest`、层级合并 |
| `events.py` | 可观测事件（`LowCoverageEvent` 等） |

## 1. 需求形态 → 实现映射

| # | 需求 | 实现 |
| --- | --- | --- |
| 1.1 | 片身份、覆盖区间/条数、状态、配置指纹 | `ShardEnvelope`：`shard_id`（uuid）、`Coverage(count, interval, global_count)`、`state`（原样保留 `metric_state`）、`config_fingerprint`（sha256） |
| 1.2 | 合并前校验：缺片/重复/越界/指纹 | `validate_shards()` 产出 `ValidationReport`，四类问题分别落字段/错误 |
| 1.3 | 顺序不敏感、逐位一致 | 片按 `(checksum, shard_id)` 规范化排序；所有 reducer 都是关联且顺序无关的；cat 在内容键上排序后拼接 |
| 1.4 | 合并幂等 | 双重身份：`shard_id` 与内容 `checksum`；`ShardMerger` 跨调用记忆 `_consumed_shard_ids/_consumed_checksums`，重投直接过滤 |
| 1.5 | 与单机全量容差一致 | 测试断言 `torch.allclose(atol=1e-6)`，实测四种语义绝对差均为 0（逐位相等） |
| 1.6 | 求和/均值/极值/拼接四语义 + 各自检查 | `SumReducer/MeanReducer/ExtremaReducer/CatReducer`，每个都有 `consistency_check` |
| 1.7 | 条数对不上可定位到片与差值 | 校验错误包含具体 `shard_id` 与带符号差值（如 `difference of +1`） |
| 1.8 | 缺片可拒绝或按已到片出结果并标注 | `missing_policy="reject"|"partial"`；partial 下 manifest 带 `coverage_ratio` 与 `missing_shard_ids`，信封状态标 `partial` |
| 1.9 | 可复核清单 | `MergeManifest.to_dict()`（可 JSON）：每片 id/条数/区间/校验和、总条数、覆盖比例、重复片、缺片、归约语义、lineage |
| 1.10 | 关掉能力时行为逐条不变 | 零改动基类 + 子进程导入测试守护；回归报告附实际测试输出 |
| 1.11 | 合并结果可再参与下一轮、不丢来源 | `merge_to_envelope()` 产出新信封，`lineage` 展平保留全部祖先 id；状态标 complete/partial |

## 2. 红线 → 实现映射

| # | 红线 | 保证方式 |
| --- | --- | --- |
| 2.1 | 不能把简单相加当通用合并 | 四种语义独立实现：均值走加权（分子/分母各自求和，`compute` 再相除），极值取 element-wise min/max，拼接做排序 cat；`test_mean_uses_weighted_not_naive_average` 专门证明不等片不会退化成均值的均值 |
| 2.2 | 不能只靠文件名判身份 | `content_checksum` 对规范化后的 state 内容求 sha256；改名不改变 checksum，篡改 payload 在 `from_dict` 即被拒 |
| 2.3 | 指纹不一致必须拒绝 | 校验阶段（片间）与合并阶段（片 vs 目标 metric）各查一次，任何不一致抛 `TorchMetricsUserError`，不产出结果 |
| 2.4 | 更新/取值调用方式与返回类型不变 | 合并只通过赋值 state 完成，目标 metric 仍是原类；`update/compute` 签名与返回 tensor 类型不变 |
| 2.5 | 不引入新运行时依赖 | 仅标准库 + torch |

## 3. 故障处理 → 实现映射

| # | 故障 | 行为 |
| --- | --- | --- |
| 3.1 | 状态缺字段/形状不一致 | 错误信息含缺失项名与来源片 id；形状/ dtype 不一致逐变量报告 |
| 3.2 | 非有限值 | `check_finite` 逐 state 变量统计 NaN/Inf 个数并标注来源片，硬错误，拒绝写出 |
| 3.3 | 中途失败不留半个结果 | 两阶段提交：全部校验与归约在 `target.clone()` 的 scratch 上完成，全部通过后才一次性赋值给目标；`test_failed_merge_is_atomic_target_unchanged` 守护 |
| 3.4 | 低覆盖可观测告警 | `LowCoverageEvent` 结构化事件 + 标准 `warnings.warn` + 可选 `event_handler` 回调，三路同时发出，非静默 |

## 4. 关键设计决策

### 4.1 为什么不复用 `Metric.merge_state`

既有 `merge_state` 有两个与分片场景冲突的约束：

1. 对 `full_state_update=True` 的指标（`MaxMetric/MinMetric` 等）直接拒绝；
2. mean 路径依赖 `_update_count`，而分片信封不携带、也不应该伪造更新计数。

因此 merger 在 reducer 产出最终 state 后直接赋值（`_assign_state`），按 `add_state` 注册的
tensor/list 种类分别处理。既有 `merge_state` 一行未动，其原有测试与契约保持不变。

### 4.2 配置指纹包含什么

类全限定名 + 每个 state 变量注册的归约函数名 + `__init__` 暴露的稳定配置属性
（如 `nan_strategy`、`num_classes`、`average`）。刻意排除：state 本身、buffer/parameter、
私有簿记字段、bound method（其 repr 带内存地址，会让同类不同实例指纹不稳定）。
`test_fingerprint_stable_across_instances_but_sensitive_to_config` 守护稳定性与敏感性。

### 4.3 顺序无关如何做到逐位一致

- 求和/均值：tensor 先 `torch.stack` 再沿 dim0 归约，浮点结合顺序由 stack 后的固定维度顺序决定，
  而 stack 顺序来自内容排序，因此任意提交顺序产生相同输入序列 → 相同浮点结果；
- 极值：element-wise min/max 天然顺序无关；
- 拼接：先按 `(shape, dtype, 展平值)` 内容键排序再 `dim_zero_cat`，输出张量本身逐位稳定，
  不止 multiset 稳定（`test_cat_semantics_order_insensitive_multiset` 断言 24 种排列输出同一有序张量）。

### 4.4 单片封存开销

实测（1000 万浮点的 MeanMetric 片）：`update` 26.4 ms，`from_metric` 封存 0.4 ms（约 1.5%），
JSON 往返 0.2 ms。封存是纯 CPU 序列化与哈希，不触碰片内计算热路径。

## 5. 使用方式

```python
from torchmetrics.aggregation import MeanMetric
from torchmetrics.sharding import Coverage, ShardEnvelope, ShardMerger

# 片端（各自独立进程/机器）
metric = MeanMetric()
metric.update(chunk)
envelope = ShardEnvelope.from_metric(
    metric, Coverage(count=len(chunk), interval=(start, end), global_count=n_total)
)
# 可 envelope.to_json() 落盘/传输；文件名随意，身份在内容里

# 合并端
merger = ShardMerger(
    MeanMetric(),
    expected_shard_ids=all_four_ids,   # 可选：启用缺片检测
    expected_total_count=n_total,      # 可选：启用覆盖核算
    missing_policy="reject",           # 或 "partial"
    min_coverage=0.9,                  # partial 下的告警阈值
    event_handler=monitoring.push,     # 结构化告警回调
)
merger.merge([envelope_a, envelope_b, ...])   # 任意顺序、可重复提交
result = merger.target_metric.compute()       # 普通 tensor，调用方式不变
manifest = merger.last_manifest                # 可复核清单
```

## 6. 测试与产物索引

- 单元/验收测试：`tests/unittests/sharding/test_sharding.py`（40 个）
- 可复现验收脚本：`reports/run_acceptance.py`
- 验收数值与报错：`reports/acceptance_report.md`、`reports/acceptance_results.json`
- 既有测试回归实际输出：`reports/regression_report.md` 及同目录两个 `.log`
