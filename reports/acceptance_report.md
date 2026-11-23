# 分片合并能力验收报告

数据集：`torch.arange(100.0)`，均匀切分为 4 片（每片 25 条）；容差 `atol=1e-6`。
本报告由 `reports/run_acceptance.py` 自动生成，原始数据见 `acceptance_results.json`。

## 1. 任意顺序合并逐位一致

| 归约语义 | 24 种排列的不同结果数 | 逐位一致 | 分片合并值 | 单机全量值 | 完全相等 |
| --- | --- | --- | --- | --- | --- |
| SumMetric | 1 | True | 4950.0 | 4950.0 | True |
| MeanMetric | 1 | True | 49.5 | 49.5 | True |
| MaxMetric | 1 | True | 99.0 | 99.0 | True |
| MinMetric | 1 | True | 0.0 | 0.0 | True |
| CatMetric | 1 | True | 长度 100 | multiset 一致 True | - |

## 2. 重复提交幂等

- 首次合并值：`4950.0`
- 整批重投 + 额外 1 个副本后的值：`4950.0`
- 幂等成立：**True**
- 清单记录的重复片：`['94ff58070d3a446f9442c965b99afa88', 'e8ef9c7dba604cc887028aeddcf48ec8', 'cf843714ec374b8ba393616ba7039b5f', '5f702cdc3a6c499a98b711fad9328c62', 'cf843714ec374b8ba393616ba7039b5f']`
- 同内容换名（内容校验和识别）判重：**True**

## 3. 缺片 / 重复片 / 配置不一致判定

### 3.1 缺片 — 默认拒绝

- 拒绝：`True`
- 报错：`refusing to merge: 1 expected shard(s) missing: ['745c37e2fd6d4b42b8eecde5e4f18939']. Re-run with missing_policy='partial' to merge the arrived shards with an explicit coverage annotation.`

### 3.2 缺片 — partial 策略（标注覆盖比例 + 告警事件）

- 告警发出：`True`；结构化事件：`LowCoverageEvent`
- 覆盖比例：`0.75`；缺片：`['745c37e2fd6d4b42b8eecde5e4f18939']`
- 已到片合并值 `2775.0` 等于前 75 条单机值 `2775.0`

### 3.3 重复片

- 合并值 `4950.0` 等于全量值 `4950.0`
- 清单记录：`['e2233620067c4bf6b2d44c16fedbdd5e']`

### 3.4 配置指纹不一致 — 必须拒绝

- 拒绝：`True`
- 报错：`configuration fingerprint mismatch between target metric and shards: target 4b5bd8000316…, shards ['4b5bd8000316…', '8b04d2ff3159…']. Merging shards computed under a different configuration is refused.`

## 4. 与单机全量的容差对比

| 指标 | 分片合并 | 单机全量 | 实测绝对差 | 容差 | 结论 |
| --- | --- | --- | --- | --- | --- |
| SumMetric | 4950.0 | 4950.0 | 0.0 | 1e-06 | 逐位相等 |
| MeanMetric | 49.5 | 49.5 | 0.0 | 1e-06 | 逐位相等 |
| MaxMetric | 99.0 | 99.0 | 0.0 | 1e-06 | 逐位相等 |
| MinMetric | 0.0 | 0.0 | 0.0 | 1e-06 | 逐位相等 |
| MeanMetric_100k_floats | -0.007383354473859072 | -0.007383354473859072 | 0.0 | 1e-06 | 容差内 |
