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
from copy import deepcopy
from numbers import Integral
from typing import Any, Optional, Union

import torch
from torch import Tensor, nn

from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities.exceptions import TorchMetricsUserError
from torchmetrics.utilities.prints import rank_zero_warn
from torchmetrics.wrappers.abstract import WrapperMetric

__all__ = ["GroupwiseWrapper"]

#: Internal module name of the overflow group. Cannot collide with encoded user keys, which always
#: start with either ``s_`` (string keys) or ``i_`` (integer keys).
_OVERFLOW_NAME = "overflow"
#: Key under which the overflow group is reported in the output of ``compute``/``forward``.
_OVERFLOW_DISPLAY_KEY = "__overflow__"


def _encode_key(key: Union[str, int]) -> str:
    """Encode a user provided group key into a string that is a valid ``nn.ModuleDict`` name.

    The encoding is deterministic and bijective, so the same key always maps to the same group,
    regardless of the process, device or batch in which it is seen, and the original key can be
    recovered from the encoded name (e.g. when loading a ``state_dict``).

    """
    if isinstance(key, str):
        # escape `_` first so that `_d` can unambiguously represent `.`
        return "s_" + key.replace("_", "__").replace(".", "_d")
    return f"i_{key}"


def _decode_key(name: str) -> Union[str, int]:
    """Recover the original group key from its encoded module name."""
    if name.startswith("i_"):
        return int(name[2:])
    if not name.startswith("s_"):
        raise ValueError(f"Cannot decode group name {name!r}: expected it to start with `s_` or `i_`.")
    s = name[2:]
    out: list[str] = []
    i = 0
    while i < len(s):
        char = s[i]
        if char == "_":
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if nxt == "_":
                out.append("_")
            elif nxt == "d":
                out.append(".")
            else:
                raise ValueError(f"Cannot decode group name {name!r}: invalid escape sequence at position {i}.")
            i += 2
        else:
            out.append(char)
            i += 1
    return "".join(out)


class GroupwiseWrapper(WrapperMetric):
    """Wrapper metric for computing a metric per group, with groups discovered dynamically at runtime.

    Each sample is assigned to a group through the ``groups`` argument of :meth:`update` and :meth:`forward`.
    The first time a group key is seen, a dedicated copy of the wrapped metric is created for it; no
    groups need to be declared upfront. The same key always maps to the same group, across batches,
    processes and devices. In addition to the per-group results, an overall result over all samples
    is always maintained, so both views of the same data are available from a single object.

    In distributed settings each process may see a different set of keys. On :meth:`compute` the key
    sets are aligned across processes (missing groups are created with empty state) before the
    underlying per-group metrics synchronize their state, so no group is lost and no sample is
    counted twice.

    Args:
        metric: base metric that should be computed per group. The metric is never updated directly;
            instead a deep copy is created for every new group key, plus one for the overall result.
            Metrics that require ``full_state_update=True`` (i.e. metrics whose ``update`` needs to see
            the full dataset at once, such as metrics relying on global ranking) cannot be decomposed
            per group and are rejected.
        max_groups: optional upper bound on the number of distinct groups. If ``None`` (default) the
            number of groups is unbounded. When a new key would exceed the bound, the behavior is
            controlled by ``on_exceed``.
        on_exceed: behavior when ``max_groups`` would be exceeded:

            - ``"error"`` (default): raise a :class:`TorchMetricsUserError` naming the rejected keys.
              The error is raised before any state is updated, so existing groups are never polluted.
            - ``"overflow"``: degrade gracefully by accumulating all further new keys into a single
              shared overflow group, reported under the key ``"__overflow__"``. The diverted keys are
              reported through a warning and the :attr:`overflow_keys` property. No key is ever
              silently dropped.

            .. note::
                With ``on_exceed="overflow"`` in a distributed setting, the diversion decision is made
                per process based on the order keys are seen locally. The overflow group therefore
                aggregates all diverted keys from all processes, and the union of diverted keys is
                reported on every process after the next :meth:`compute` call.

        kwargs: additional keyword arguments passed to the :class:`~torchmetrics.Metric` base class
            (e.g. ``process_group``, ``dist_sync_on_step``). The same arguments are also applied to
            every dynamically created per-group metric.

    Raises:
        TypeError: If ``metric`` is not an instance of :class:`~torchmetrics.Metric`.
        ValueError: If ``metric`` requires ``full_state_update=True``.
        ValueError: If ``max_groups`` is not ``None`` or a positive integer.
        ValueError: If ``on_exceed`` is not ``"error"`` or ``"overflow"``.

    Example::
        Basic example with string group keys that appear over several batches:

        >>> import torch
        >>> from torchmetrics.aggregation import MeanMetric
        >>> from torchmetrics.wrappers import GroupwiseWrapper
        >>> metric = GroupwiseWrapper(MeanMetric())
        >>> metric.update(torch.tensor([1.0, 2.0, 3.0, 4.0]), groups=["cpu", "gpu", "cpu", "gpu"])
        >>> metric.update(torch.tensor([10.0, 20.0]), groups=["tpu", "cpu"])  # new key appears at runtime
        >>> out = metric.compute()
        >>> out["overall"]
        tensor(6.6667)
        >>> out["groups"]  # doctest: +NORMALIZE_WHITESPACE
        {'cpu': tensor(8.), 'gpu': tensor(3.), 'tpu': tensor(10.)}

    Example::
        Integer keys, and the sum of an additive metric over groups equals the overall result:

        >>> from torchmetrics.aggregation import SumMetric
        >>> metric = GroupwiseWrapper(SumMetric())
        >>> _ = metric(torch.tensor([1.0, 2.0, 3.0]), groups=[0, 1, 0])
        >>> out = metric.compute()
        >>> out["overall"]
        tensor(6.)
        >>> out["groups"]
        {0: tensor(4.), 1: tensor(2.)}

    """

    is_differentiable: Optional[bool] = None
    full_state_update: Optional[bool] = False

    def __init__(
        self,
        metric: Metric,
        max_groups: Optional[int] = None,
        on_exceed: str = "error",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(metric, MetricCollection):
            raise TypeError(
                "Expected argument `metric` to be a single `torchmetrics.Metric`, not a `MetricCollection`."
                " Wrap each metric in the collection separately instead."
            )
        if not isinstance(metric, Metric):
            raise TypeError(f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}")
        if metric.full_state_update is True:
            raise ValueError(
                f"Metric {metric.__class__.__name__} cannot be grouped because it has `full_state_update=True`,"
                " meaning its `update` method needs to see the full accumulated state at once (e.g. metrics"
                " relying on global ranking or full-dataset statistics). Computing it on per-group subsets"
                " would silently produce wrong numbers, so it is rejected explicitly. Compute such metrics"
                " per group by manually splitting the data instead."
            )
        if max_groups is not None and (
            not isinstance(max_groups, Integral) or isinstance(max_groups, bool) or max_groups < 1
        ):
            raise ValueError(f"Expected argument `max_groups` to be `None` or a positive integer but got {max_groups}")
        if on_exceed not in ("error", "overflow"):
            raise ValueError(f"Expected argument `on_exceed` to be one of 'error', 'overflow' but got {on_exceed}")

        self.max_groups = int(max_groups) if max_groups is not None else None
        self.on_exceed = on_exceed

        # template is never updated; every group metric and the overall metric are deep copies of it
        self._template = metric
        self._metric_kwargs = kwargs  # Metric base-class kwargs, re-applied to every dynamically created metric
        for key, value in kwargs.items():
            setattr(self._template, key, value)

        self.overall_metric = deepcopy(self._template)
        self.group_metrics = nn.ModuleDict()
        self._diverted_keys: set = set()

    @property
    def higher_is_better(self) -> Optional[bool]:  # type: ignore[override]
        """Return if the metric is higher the better."""
        return self._template.higher_is_better

    @property
    def group_keys(self) -> list[Union[str, int]]:
        """Return the list of group keys seen so far (excluding the overflow group), in sorted order."""
        return [_decode_key(name) for name in self._sorted_group_names()]

    @property
    def overflow_keys(self) -> list[Union[str, int]]:
        """Return the list of keys that were diverted to the overflow group (empty unless limit was exceeded)."""
        return sorted(self._diverted_keys, key=repr)

    def _sorted_group_names(self) -> list[str]:
        """Return encoded names of all real groups, sorted deterministically by decoded key."""
        names = [name for name in self.group_metrics if name != _OVERFLOW_NAME]
        return sorted(names, key=lambda n: (isinstance(_decode_key(n), str), _decode_key(n)))

    def _num_groups(self) -> int:
        return len(self.group_metrics) - (1 if _OVERFLOW_NAME in self.group_metrics else 0)

    @staticmethod
    def _normalize_key(key: Any, index: int) -> Union[str, int]:
        """Normalize a single user provided key to a `str` or `int`, or raise a descriptive error."""
        if isinstance(key, Tensor) and key.ndim == 0:
            key = key.item()
        if isinstance(key, str):
            return key
        if isinstance(key, bool):
            return int(key)
        if isinstance(key, Integral):
            return int(key)
        raise TypeError(
            f"Invalid group key at index {index}: {key!r} of type {type(key).__name__}."
            " Expected every group key to be a `str` or an `int` (None, floats and containers are not supported)."
        )

    @staticmethod
    def _infer_batch_size(args: Any, kwargs: Any) -> int:
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, Tensor) and value.ndim >= 1:
                return value.shape[0]
        raise TorchMetricsUserError(
            "Could not infer the batch size of the input: expected at least one tensor argument with a leading"
            " sample dimension to `GroupwiseWrapper.update`/`forward`."
        )

    def _validate_and_normalize(self, groups: Any, args: Any, kwargs: Any) -> tuple[list[Union[str, int]], int]:
        """Validate the `groups` argument against the batch and normalize keys to `str`/`int`."""
        batch_size = self._infer_batch_size(args, kwargs)
        if groups is None:
            raise TorchMetricsUserError("Missing required keyword argument `groups` with one group key per sample.")
        if isinstance(groups, Tensor):
            if groups.ndim != 1:
                raise ValueError(
                    f"Expected argument `groups` to be a 1-dimensional tensor but got shape {tuple(groups.shape)}."
                )
            raw_keys = groups.tolist()
        elif isinstance(groups, (list, tuple)):
            raw_keys = list(groups)
        else:
            raise TypeError(
                f"Expected argument `groups` to be a list, tuple or 1-dimensional tensor of keys"
                f" but got {type(groups).__name__}."
            )
        if len(raw_keys) != batch_size:
            raise ValueError(
                f"Expected argument `groups` to have one entry per sample, i.e. length {batch_size},"
                f" but got {len(raw_keys)} entries."
            )
        return [self._normalize_key(key, i) for i, key in enumerate(raw_keys)], batch_size

    def _check_cardinality(self, keys: list[Union[str, int]]) -> None:
        """Raise (before any state is mutated) if the new keys in this batch would exceed `max_groups`."""
        if self.max_groups is None or self.on_exceed != "error":
            return
        new_keys: list[Union[str, int]] = []
        seen: set = set()
        for key in keys:
            if _encode_key(key) not in self.group_metrics and key not in self._diverted_keys and key not in seen:
                new_keys.append(key)
                seen.add(key)
        n_allowed = self.max_groups - self._num_groups()
        if len(new_keys) > n_allowed:
            rejected = new_keys[n_allowed:]
            raise TorchMetricsUserError(
                f"Number of groups would exceed the declared limit max_groups={self.max_groups}:"
                f" {len(rejected)} new key(s) rejected: {rejected}."
                " No state was updated; existing groups are unaffected."
                " Increase `max_groups` or set `on_exceed='overflow'` to divert excess keys to a shared"
                " overflow group instead."
            )

    def _overflow_metric(self) -> Metric:
        if _OVERFLOW_NAME not in self.group_metrics:
            self.group_metrics[_OVERFLOW_NAME] = deepcopy(self._template)
        return self.group_metrics[_OVERFLOW_NAME]

    def _get_or_create_group(self, key: Union[str, int]) -> Metric:
        """Return the metric responsible for `key`, creating it (or diverting to overflow) if needed."""
        name = _encode_key(key)
        if name in self.group_metrics:
            return self.group_metrics[name]
        if key in self._diverted_keys:
            return self._overflow_metric()
        if self.max_groups is not None and self._num_groups() >= self.max_groups:
            if self.on_exceed == "error":
                # normally prevented by `_check_cardinality`; kept as a safety net
                raise TorchMetricsUserError(
                    f"Number of groups would exceed the declared limit max_groups={self.max_groups}:"
                    f" key {key!r} rejected. No state was updated."
                )
            self._diverted_keys.add(key)
            rank_zero_warn(
                f"Group key {key!r} exceeds the declared limit max_groups={self.max_groups} and was diverted"
                f" to the shared '{_OVERFLOW_DISPLAY_KEY}' group. See the `overflow_keys` property for all"
                " diverted keys.",
                UserWarning,
            )
            return self._overflow_metric()
        metric = deepcopy(self._template)
        self.group_metrics[name] = metric
        return metric

    @staticmethod
    def _group_indices(keys: list[Union[str, int]]) -> list[tuple[Union[str, int], Tensor]]:
        """Return `(key, sample_indices)` pairs, one per unique key, in first-seen order."""
        unique = list(dict.fromkeys(keys))
        position = {key: i for i, key in enumerate(unique)}
        gid = torch.tensor([position[key] for key in keys], dtype=torch.long)
        order = torch.argsort(gid, stable=True)
        unique_ids, counts = torch.unique_consecutive(gid[order], return_counts=True)
        splits = torch.split(order, counts.tolist())
        return [(unique[uid], idx) for uid, idx in zip(unique_ids.tolist(), splits)]

    @staticmethod
    def _mask_inputs(args: Any, kwargs: Any, idx: Tensor, batch_size: int) -> tuple[list, dict]:
        """Select the samples belonging to one group from every tensor argument with a sample dimension."""

        def _select(value: Any) -> Any:
            if isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == batch_size:
                return value[idx.to(value.device)]
            return value

        return [_select(a) for a in args], {k: _select(v) for k, v in kwargs.items()}

    def update(self, *args: Any, groups: Any = None, **kwargs: Any) -> None:
        """Update the overall metric and the per-group metrics with a batch of samples.

        Args:
            args: positional arguments forwarded to the wrapped metric. Tensor arguments whose first
                dimension equals the batch size are split per group; all other arguments are forwarded
                unchanged to every group.
            groups: one group key per sample. A list/tuple of `str`/`int` keys, or a 1-dimensional
                integer tensor. New keys create new groups automatically.
            kwargs: keyword arguments forwarded to the wrapped metric, split like `args`.

        """
        keys, batch_size = self._validate_and_normalize(groups, args, kwargs)
        if batch_size == 0:
            return  # empty batch: nothing to accumulate
        self._check_cardinality(keys)
        self.overall_metric.update(*args, **kwargs)
        for key, idx in self._group_indices(keys):
            metric = self._get_or_create_group(key)
            masked_args, masked_kwargs = self._mask_inputs(args, kwargs, idx, batch_size)
            metric.update(*masked_args, **masked_kwargs)

    def forward(self, *args: Any, groups: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Calculate the metric on the current batch (per group and overall) and accumulate the state.

        Returns:
            A dictionary of the form ``{"overall": batch_value, "groups": {key: batch_value}}`` where
            ``groups`` only contains the groups present in this batch. For an empty batch
            ``{"overall": None, "groups": {}}`` is returned.

        """
        keys, batch_size = self._validate_and_normalize(groups, args, kwargs)
        if batch_size == 0:
            return {"overall": None, "groups": {}}
        self._check_cardinality(keys)
        overall = self.overall_metric(*args, **kwargs)
        group_values = {}
        for key, idx in self._group_indices(keys):
            metric = self._get_or_create_group(key)
            masked_args, masked_kwargs = self._mask_inputs(args, kwargs, idx, batch_size)
            group_values[key] = metric(*masked_args, **masked_kwargs)
        return {"overall": overall, "groups": group_values}

    def _create_placeholder(self, name: str) -> None:
        """Create a group metric with empty (default) state for an encoded name.

        Used to align the set of groups across processes before synchronization, and to recreate
        groups when loading a `state_dict`. The metric starts with default (zero) state, so it
        contributes nothing to the distributed reduction until it receives real updates.

        """
        metric = deepcopy(self._template)
        # mark as updated to suppress the spurious "compute before update" warning on ranks that
        # never saw this key; the state itself remains at its default
        metric._update_count = 1
        self.group_metrics[name] = metric

    def _sync_group_keys(self) -> None:
        """Align the set of groups (and diverted keys) across all processes before synchronization."""
        if not self._to_sync:
            return
        if not (callable(self.distributed_available_fn) and self.distributed_available_fn()):
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        world_size = torch.distributed.get_world_size(group=self.process_group)
        if world_size <= 1:
            return
        payload = (list(self.group_metrics.keys()), sorted(self._diverted_keys, key=repr))
        gathered: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered, payload, group=self.process_group)
        for names, diverted in gathered:
            for name in names:
                if name not in self.group_metrics:
                    self._create_placeholder(name)
            self._diverted_keys.update(diverted)

    def compute(self) -> dict[str, Any]:
        """Compute the overall metric and the per-group metrics.

        In distributed settings the group key sets of all processes are aligned first, and each
        per-group metric synchronizes its state across processes, so the merged result is identical
        to a single-process run over all data.

        Returns:
            A dictionary of the form ``{"overall": value, "groups": {key: value}}``. Groups are
            sorted by key (integers first, then strings). If keys were diverted due to `max_groups`,
            the overflow group is reported last under the key ``"__overflow__"``.

        """
        self._sync_group_keys()
        overall = self.overall_metric.compute()
        groups = {_decode_key(name): self.group_metrics[name].compute() for name in self._sorted_group_names()}
        if _OVERFLOW_NAME in self.group_metrics:
            groups[_OVERFLOW_DISPLAY_KEY] = self.group_metrics[_OVERFLOW_NAME].compute()
        return {"overall": overall, "groups": groups}

    def reset(self) -> None:
        """Reset the overall metric and remove all dynamically created groups.

        Group keys are re-discovered from the data after a reset, which also resets the `max_groups`
        budget and the set of diverted keys.

        """
        self.overall_metric.reset()
        self.group_metrics = nn.ModuleDict()
        self._diverted_keys = set()
        super().reset()

    def persistent(self, mode: bool = False) -> None:
        """Change if metric states should be saved to the state dict. Applies to all current and future groups."""
        self._template.persistent(mode)
        self.overall_metric.persistent(mode)
        for metric in self.group_metrics.values():
            metric.persistent(mode)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Recreate dynamically created groups found in the state dict before loading their state."""
        group_prefix = prefix + "group_metrics."
        for key in list(state_dict.keys()):
            if key.startswith(group_prefix):
                name = key[len(group_prefix) :].split(".", 1)[0]
                if name and name not in self.group_metrics:
                    self._create_placeholder(name)
        if any(key.startswith(prefix + "overall_metric.") for key in state_dict):
            # state is being restored, so suppress the spurious "compute before update" warning
            self.overall_metric._update_count = 1
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _filter_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Filter kwargs for the wrapped metric."""
        return self._template._filter_kwargs(**kwargs)

    def __repr__(self) -> str:
        """Return a representation of the wrapper, including the wrapped metric and current groups."""
        return (
            f"{self.__class__.__name__}(\n  metric={self._template!r},\n  max_groups={self.max_groups},"
            f"\n  on_exceed={self.on_exceed!r},\n  groups={self.group_keys}\n)"
        )
