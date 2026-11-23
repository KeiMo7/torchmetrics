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
"""Shard envelope: identity, coverage, status, configuration fingerprint and content checksum.

A shard envelope is the unit that travels between worker processes and the merger.  The identity of a shard is
never derived from its file name: :func:`content_checksum` hashes the serialized metric state itself, so two shards
with the same payload can be detected as duplicates regardless of how they are stored, and tampered payloads are
rejected because the stored checksum no longer matches.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from uuid import uuid4

from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities.exceptions import TorchMetricsUserError

#: Version of the on-the-wire envelope format. Bumped whenever the schema changes in an incompatible way.
ENVELOPE_VERSION = 1

#: Every envelope must contain exactly these top-level keys. Used for the "which field is missing / from which shard"
#: error messages required by the failure-mode spec.
REQUIRED_KEYS = ("version", "shard_id", "metric_class", "config_fingerprint", "status", "coverage", "state", "checksum")

#: Valid shard status values.
VALID_STATUSES = ("complete", "partial", "failed")


def _to_jsonable(obj: Any) -> Any:
    """Convert tensors and other non-JSON-native objects into stable, JSON-serializable values."""
    if isinstance(obj, Tensor):
        return {
            "__tensor__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.detach().cpu().tolist(),
        }
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    # fall back to repr for exotic config values (e.g. callables, dtypes)
    return repr(obj)


def canonical_json(obj: Any) -> str:
    """Serialize an object to a canonical JSON string (sorted keys, fixed separators) suitable for hashing."""
    return json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_fingerprint(metric: Metric, extra_config: Optional[dict[str, Any]] = None) -> str:
    """Compute the configuration fingerprint of a metric.

    The fingerprint binds together the metric class, the reduction semantics of every registered state variable and
    the metric's configurable attributes (anything stored on ``__init__`` that is not a state variable, buffer or
    module). Two shards may only be merged when fingerprints match exactly, which prevents mixing incompatible
    configurations (e.g. different ``num_classes`` or ``average`` settings).

    Args:
        metric: the metric instance whose configuration is fingerprinted.
        extra_config: optional additional key/values that should participate in the fingerprint (e.g. the dataset
            id or a job-level configuration version).

    """
    state_attrs = set(metric._defaults)
    config: dict[str, Any] = {
        "metric_class": f"{metric.__class__.__module__}.{metric.__class__.__qualname__}",
        "reductions": {
            name: getattr(metric._reductions[name], "__name__", str(metric._reductions[name]))
            for name in sorted(metric._defaults)
        },
        "attributes": {},
    }
    # Volatile machinery is excluded: bound/wrapped methods carry memory addresses in their repr, tensors are
    # state rather than configuration, and private attributes are internal bookkeeping. What remains are the
    # scalar configuration values set in __init__ (e.g. nan_strategy, num_classes, average), which is exactly
    # what must agree across shards.
    excluded_names = state_attrs | {"training", "update", "compute"}
    for name in sorted(vars(metric)):
        if name in excluded_names or name.startswith("_"):
            continue
        value = vars(metric)[name]
        if callable(value) or isinstance(value, Tensor):
            continue
        config["attributes"][name] = value
    if extra_config:
        config["extra"] = extra_config
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def content_checksum(state: dict[str, Union[Tensor, list]]) -> str:
    """Hash the *content* of a metric state.

    The checksum is independent of the shard id and file name. Two shards carrying byte-identical states produce the
    same checksum, which is how duplicate submissions are detected without trusting file names.
    """
    payload = canonical_json({name: state[name] for name in sorted(state)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Coverage:
    """Sample coverage claimed by a shard.

    Args:
        count: number of samples (rows/records) the shard consumed. Must be a non-negative integer.
        interval: optional inclusive ``[start, end]`` sample-index range covered by the shard. Used to detect
            out-of-bounds and overlapping ranges against the declared global range.
        global_count: optional total number of samples the whole job is expected to cover. When present, shard
            intervals are validated against ``[0, global_count - 1]`` and coverage ratios can be computed.

    """

    count: int
    interval: Optional[tuple[int, int]] = None
    global_count: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate coverage values on construction."""
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 0:
            raise TorchMetricsUserError(
                f"Coverage.count must be a non-negative integer, got {self.count!r} (type {type(self.count).__name__})"
            )
        if self.interval is not None:
            if (
                not isinstance(self.interval, (tuple, list))
                or len(self.interval) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in self.interval)
            ):
                raise TorchMetricsUserError(f"Coverage.interval must be a pair of ints, got {self.interval!r}")
            start, end = self.interval
            if start > end:
                raise TorchMetricsUserError(f"Coverage.interval start ({start}) cannot be greater than end ({end})")
            if self.count != end - start + 1:
                raise TorchMetricsUserError(
                    f"Coverage count ({self.count}) does not match interval length "
                    f"[{start}, {end}] which contains {end - start + 1} samples"
                )
        if self.global_count is not None:
            if not isinstance(self.global_count, int) or isinstance(self.global_count, bool) or self.global_count < 0:
                raise TorchMetricsUserError(
                    f"Coverage.global_count must be a non-negative int, got {self.global_count!r}"
                )
            if self.interval is not None and (self.interval[0] < 0 or self.interval[1] >= self.global_count):
                raise TorchMetricsUserError(
                    f"Shard interval {list(self.interval)} is out of bounds for global_count={self.global_count} "
                    f"(valid indices: 0..{self.global_count - 1})"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-friendly dict."""
        return {
            "count": self.count,
            "interval": list(self.interval) if self.interval is not None else None,
            "global_count": self.global_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Coverage":
        """Rebuild a :class:`Coverage` from a plain dict, validating its shape."""
        if not isinstance(data, dict):
            raise TorchMetricsUserError(f"coverage must be a dict, got {type(data).__name__}")
        for key in ("count",):
            if key not in data:
                raise TorchMetricsUserError(f"coverage is missing required field {key!r}: got {data}")
        interval = data.get("interval")
        return cls(
            count=data["count"],
            interval=tuple(interval) if interval is not None else None,
            global_count=data.get("global_count"),
        )


@dataclass(eq=False)
class ShardEnvelope:
    """Self-describing container for one shard's metric state.

    Attributes:
        shard_id: stable, unique identifier of the shard (generated with ``uuid4`` by default). Identity is *also*
            verified from content via :attr:`checksum`, never from a file name.
        metric_class: fully qualified class name of the metric that produced the state.
        config_fingerprint: sha256 hex digest produced by :func:`config_fingerprint`.
        status: one of :data:`VALID_STATUSES`.
        coverage: sample coverage of this shard.
        state: the raw metric state (dict of tensors / lists of tensors), exactly as ``metric.metric_state`` returns.
        checksum: sha256 over the canonicalized state; verified on load.
        version: envelope format version.
        lineage: ids of the shards (or merged envelopes) that were combined to produce this one. A freshly produced
            shard records its own id; a re-merged envelope carries the full ancestry so provenance is never lost.

    """

    shard_id: str
    metric_class: str
    config_fingerprint: str
    status: str
    coverage: Coverage
    state: dict[str, Any]
    checksum: str
    version: int = ENVELOPE_VERSION
    lineage: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate status and state kind on construction, defaulting lineage to the shard's own id."""
        if self.status not in VALID_STATUSES:
            raise TorchMetricsUserError(
                f"Invalid shard status {self.status!r}; expected one of {list(VALID_STATUSES)} "
                f"(shard {self.shard_id!r})"
            )
        if not isinstance(self.state, dict):
            raise TorchMetricsUserError(
                f"state must be a dict, got {type(self.state).__name__} (shard {self.shard_id!r})"
            )
        if not self.lineage:
            self.lineage = [self.shard_id]

    # --------------------------------------------------------------------------------------------------
    # construction
    # --------------------------------------------------------------------------------------------------
    @classmethod
    def from_metric(
        cls,
        metric: Metric,
        coverage: Coverage,
        *,
        shard_id: Optional[str] = None,
        status: str = "complete",
        extra_config: Optional[dict[str, Any]] = None,
        lineage: Optional[list[str]] = None,
    ) -> "ShardEnvelope":
        """Build an envelope from a live metric instance.

        Args:
            metric: metric that has accumulated one shard of data.
            coverage: sample coverage of that data.
            shard_id: explicit shard id; a random uuid4 hex is generated when omitted.
            status: shard status (defaults to ``"complete"``).
            extra_config: additional values to bind into the configuration fingerprint.
            lineage: explicit ancestry; defaults to the new shard id.

        """
        sid = shard_id or uuid4().hex
        state = {name: _detach_state_value(value) for name, value in metric.metric_state.items()}
        fingerprint = config_fingerprint(metric, extra_config=extra_config)
        checksum = content_checksum(state)
        return cls(
            shard_id=sid,
            metric_class=f"{metric.__class__.__module__}.{metric.__class__.__qualname__}",
            config_fingerprint=fingerprint,
            status=status,
            coverage=coverage,
            state=state,
            checksum=checksum,
            lineage=list(lineage) if lineage else [sid],
        )

    # --------------------------------------------------------------------------------------------------
    # serialization
    # --------------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "version": self.version,
            "shard_id": self.shard_id,
            "metric_class": self.metric_class,
            "config_fingerprint": self.config_fingerprint,
            "status": self.status,
            "coverage": self.coverage.to_dict(),
            "state": _to_jsonable(self.state),
            "checksum": self.checksum,
            "lineage": list(self.lineage),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Any) -> "ShardEnvelope":
        """Validate and rebuild an envelope from a plain dict.

        Raises:
            TorchMetricsUserError: with the missing field name and originating shard id when a required field is
                absent or malformed, and when the state checksum does not match the payload.

        """
        if not isinstance(data, dict):
            raise TorchMetricsUserError(f"Shard payload must be a dict, got {type(data).__name__}")
        shard_id = data.get("shard_id", "<unknown>")
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise TorchMetricsUserError(
                f"Shard {shard_id!r} is missing required field(s): {missing}. Present fields: {sorted(data)}"
            )
        if data["version"] != ENVELOPE_VERSION:
            raise TorchMetricsUserError(
                f"Shard {shard_id!r} has unsupported envelope version {data['version']!r}, expected {ENVELOPE_VERSION}"
            )
        try:
            coverage = Coverage.from_dict(data["coverage"])
        except TorchMetricsUserError as err:
            raise TorchMetricsUserError(f"Shard {shard_id!r}: {err}") from err

        state = _state_from_jsonable(data["state"], shard_id)

        actual_checksum = content_checksum(state)
        if actual_checksum != data["checksum"]:
            raise TorchMetricsUserError(
                f"Shard {shard_id!r} failed content checksum verification: stored {data['checksum']!r}, "
                f"recomputed {actual_checksum!r}. The state payload is corrupted or was modified after sealing."
            )
        lineage = data.get("lineage") or [shard_id]
        if not isinstance(lineage, list) or not all(isinstance(x, str) for x in lineage):
            raise TorchMetricsUserError(f"Shard {shard_id!r}: lineage must be a list of strings, got {lineage!r}")
        return cls(
            shard_id=data["shard_id"],
            metric_class=data["metric_class"],
            config_fingerprint=data["config_fingerprint"],
            status=data["status"],
            coverage=coverage,
            state=state,
            checksum=data["checksum"],
            version=data["version"],
            lineage=lineage,
        )

    @classmethod
    def from_json(cls, text: str) -> "ShardEnvelope":
        """Validate and rebuild an envelope from a JSON string."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as err:
            raise TorchMetricsUserError(f"Cannot parse shard JSON: {err}") from err
        return cls.from_dict(data)


def _detach_state_value(value: Any) -> Any:
    """Detach a state value from autograd and move it to CPU for serialization."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [_detach_state_value(v) for v in value]
    return value


def _state_from_jsonable(obj: Any, shard_id: str) -> Any:
    """Rebuild tensors encoded by :func:`_to_jsonable` back into real tensor state."""
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            if "dtype" not in obj or "shape" not in obj or "data" not in obj:
                raise TorchMetricsUserError(
                    f"Shard {shard_id!r}: serialized tensor is missing one of 'dtype'/'shape'/'data': {sorted(obj)}"
                )
            try:
                import torch

                dtype = getattr(torch, obj["dtype"].split(".")[-1])
            except AttributeError as err:
                raise TorchMetricsUserError(f"Shard {shard_id!r}: unknown tensor dtype {obj['dtype']!r}") from err
            return torch.tensor(obj["data"], dtype=dtype)
        return {key: _state_from_jsonable(value, shard_id) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_state_from_jsonable(value, shard_id) for value in obj]
    raise TorchMetricsUserError(
        f"Shard {shard_id!r}: state must be a nested dict/list structure, found scalar {obj!r} at the top level"
    )
