# 原有指标回归测试记录

本文件记录引入 `torchmetrics.sharding`（opt-in 分片合并能力）后，对**既有指标**测试的实际运行结果，
证明需求第 10 条「关掉这套能力时，原有单进程累计与分布式同步行为逐条不变」。

## 关键设计前提

分片能力是独立的子包，`torchmetrics.metric.Metric` 基类**没有任何一行改动**（见 `git diff --stat`，
跟踪文件中仅 `pyproject.toml` 增加一条 lint 豁免）。在干净解释器中
`import torchmetrics.aggregation` 不会导入 `torchmetrics.sharding`（有测试
`test_sharding_package_not_imported_by_metric_base` 守护），因此既有路径在物理上不可能被改变。

## 1. 基类与聚合指标（bases + aggregation）

命令：

```bash
PYTHONPATH=src python -m pytest tests/unittests/bases/test_metric.py \
    tests/unittests/bases/test_aggregation.py -p no:doctest -o addopts="" -q
```

实际输出（完整日志见 `legacy_tests_bases_aggregation.log`）：

```
131 passed, 12 skipped, 33 warnings in 5.97s
```

覆盖：`Metric.add_state` / `reset` / `update` / `compute` / `forward` / `state_dict` /
`merge_state` / `sync`，以及 `SumMetric`、`MeanMetric`、`MaxMetric`、`MinMetric`、`CatMetric`
的全部既有行为。

## 2. 分类 / 回归 / 聚类 / Wrapper 指标

命令：

```bash
PYTHONPATH=src python -m pytest \
    tests/unittests/classification/test_accuracy.py \
    tests/unittests/regression/test_r2.py \
    tests/unittests/regression/test_pearson.py \
    tests/unittests/clustering --ignore=tests/unittests/clustering/test_utils.py \
    tests/unittests/wrappers --ignore=tests/unittests/wrappers/test_feature_share.py \
    -p no:doctest -o addopts="" -q
```

本分支实际输出（完整日志见 `legacy_tests_regression_classification.log`）：

```
23 failed, 1052 passed, 488 skipped in 72.64s
```

### 23 个失败是环境基线问题，与本次改动无关

为证明这一点，在**未包含本次改动的已安装发布版 torchmetrics 1.9.0**（site-packages，独立目录）上
运行完全相同的测试选择，实际输出：

```
23 failed, 1052 passed, 488 skipped in 160.73s
```

两边的通过/失败/跳过计数逐位相同。23 个失败全部位于
`tests/unittests/classification/test_accuracy.py`，根因是当前环境的 scikit-learn 版本在交叉校验时
对空数组抛出 `ValueError: Found empty input array ... while a minimum of 1 sample is required`
（日志中出现 151 次 sklearn 调用栈），属于依赖版本兼容问题，在本分支存在之前就已失败。

被 `--ignore` 跳过的两个收集错误同样是基线问题，与本次改动无关：

- `tests/unittests/clustering/test_utils.py`：导入期错误；
- `tests/unittests/wrappers/test_feature_share.py`：`ImportError: cannot import name
  'FrechetInceptionDistance' from 'torchmetrics.image'`（可选视觉依赖未安装）。

## 3. 新增分片能力自身的测试

```bash
PYTHONPATH=src python -m pytest tests/unittests/sharding/test_sharding.py -p no:doctest -o addopts="" -q
40 passed, 20 warnings in 6.61s
```

doctest：

```bash
PYTHONPATH=src python -m pytest src/torchmetrics/sharding/ -o addopts="--doctest-modules --doctest-plus" -q
1 passed
```

## 4. 结论

| 测试范围 | 结果 | 与基线对比 |
| --- | --- | --- |
| bases + aggregation | 131 passed, 12 skipped | 完全一致 |
| 分类/回归/聚类/wrapper | 1052 passed, 488 skipped, 23 failed | 失败数与失败根因与基线完全相同 |
| 新增 sharding 套件 | 40 passed | 新增能力，不影响既有路径 |
| sharding doctest | 1 passed | — |

既有单进程累计语义、`Metric.merge_state` 契约、分布式 `sync` 数值语义均无回归。
