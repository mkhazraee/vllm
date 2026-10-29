# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM secondary-tier adapter for KVCR."""

import copy
import ctypes
import mmap
import socket
import time
import uuid
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import msgspec
from kvcr import (
    DURATION_METRIC,
    KVCR,
    ROUTER_HINT_CAPABILITIES,
    STATE_METRIC,
    TRANSFER_BLOCKS_METRIC,
    TRANSFER_BYTES_METRIC,
    KVCRBindings,
)
from kvcr.config import (
    G3Options,
    KeyAdapter,
    KVCRBackendConfigs,
    KVCRConfig,
    KVCRGuardConfig,
    LocalDramOptions,
    RemoteFWDramOptions,
)
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.policy import (
    FIFOPolicy,
    G3FIFOPolicy,
    G3LRUPolicy,
    KVCachePolicy,
    LRUPolicy,
)
from kvcr.types import (
    BlockKey,
    CacheTier,
    InventoryEvent,
    MemoryRef,
    OpHandle,
    PinRequestId,
    PinResult,
    QueryStatus,
    RegionDescriptor,
)
from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    maybe_convert_block_hash,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    Medium,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    ParentManager,
    SecondaryTierManager,
    TransferJob,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector


_REQUIRED_ROUTER_CAPABILITIES = ROUTER_HINT_CAPABILITIES

logger = init_logger(__name__)

_BUILTIN_POLICIES: dict[str, type[KVCachePolicy]] = {
    "fifo": FIFOPolicy,
    "lru": LRUPolicy,
    "g3_fifo": G3FIFOPolicy,
    "g3_lru": G3LRUPolicy,
}


def _resolve_policy(name: str | None) -> KVCachePolicy | None:
    if name is None:
        return None
    policy_type = _BUILTIN_POLICIES.get(name)
    if policy_type is None:
        if "." not in name:
            raise ValueError(
                f"Unknown KVCR policy: {name!r}. "
                f"Supported: {list(_BUILTIN_POLICIES)}; external policies "
                "must use a fully qualified module.Class path."
            )
        policy_type = resolve_obj_by_qualname(name)
        if not isinstance(policy_type, type) or not issubclass(
            policy_type, KVCachePolicy
        ):
            raise TypeError(f"{name} is not a KVCachePolicy")
    return policy_type()


def _vllm_metric_name(name: str) -> str:
    return f"vllm:{name}"


def _kvcr_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    return {
        _vllm_metric_name(DURATION_METRIC): OffloadingHistogramMetadata(
            documentation="KVCR operation and stage duration in seconds.",
            labelnames=("scope", "result"),
        ),
        _vllm_metric_name(TRANSFER_BYTES_METRIC): OffloadingCounterMetadata(
            documentation="Bytes transferred by KVCR operations.",
            labelnames=("operation",),
        ),
        _vllm_metric_name(TRANSFER_BLOCKS_METRIC): OffloadingCounterMetadata(
            documentation="KV blocks transferred by successful KVCR operations.",
            labelnames=("operation",),
        ),
        _vllm_metric_name(STATE_METRIC): OffloadingGaugeMetadata(
            documentation="Current KVCR metadata and operation counts.",
            labelnames=("resource",),
        ),
    }


class _HintClosureState:
    def __init__(self, raw_hint):
        self.raw_hint = copy.deepcopy(raw_hint)
        self.chain = None
        self.valid = False
        self.proof = None
        self.proof_id = None
        self.raw_anchors = frozenset()
        self.reason = None


def _get_request_hash_chain(req_context):
    # Scheduler is fully imported before managers handle a lookup. Keep the
    # generic scheduler independent of the KVCR backend and native SDK.
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestHashChain,
    )

    return req_context.get_state(RequestHashChain)


def _hint_key_position(proof, key):
    if type(key) is not bytes or len(key) != 36:
        return None
    span = proof["_groups"].get(int.from_bytes(key[-4:], "big"))
    end = proof["_positions"].get(key[:-4])
    if (
        span is None
        or end is None
        or end > proof["covered_through_tokens"]
        or end % span
    ):
        return None
    return end


def _strict_hint_prefix_closure(chain, raw_hint):
    if not getattr(chain, "coherent", True):
        raise ValueError("request_chain_changed")
    hashes, step = chain.block_hashes, chain.tokens_per_hash
    if (
        type(step) is not int
        or step <= 0
        or not isinstance(hashes, tuple)
        or not hashes
        or any(type(h) is not bytes or len(h) != 32 for h in hashes)
    ):
        raise ValueError("malformed_request_chain")
    groups = chain.group_chunks
    if (
        not isinstance(groups, tuple)
        or not groups
        or any(
            not isinstance(g, tuple)
            or len(g) != 2
            or type(g[0]) is not int
            or not 0 <= g[0] < 1 << 32
            or type(g[1]) is not int
            or g[1] <= 0
            or g[1] % step
            for g in groups
        )
        or len({g[0] for g in groups}) != len(groups)
    ):
        raise ValueError("malformed_group_boundaries")
    anchor_step = chain.anchor_tokens
    if type(anchor_step) is not int or anchor_step <= 0 or anchor_step % step:
        raise ValueError("unknown_anchor_geometry")
    if not isinstance(raw_hint, dict) or raw_hint.get("protocol_version") != "0.1":
        raise ValueError("unsupported_hint_protocol")
    actions = raw_hint.get("actions")
    if not isinstance(actions, list):
        raise ValueError("malformed_hint_actions")
    action = next(
        (
            a
            for a in actions
            if isinstance(a, dict) and a.get("action_type") == "kv.fetch"
        ),
        None,
    )
    if action is None or action.get("action_version") != "1.0":
        raise ValueError("unsupported_fetch_action")
    payload = action.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("mode", "copy") != "copy"
        or "no_retain" in payload
    ):
        raise ValueError("unsupported_fetch_payload")
    source, raw = payload.get("source_control_endpoint"), payload.get("block_hashes")
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(raw, list)
        or not raw
        or any(type(h) is not int or not 0 <= h < 1 << 64 for h in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError("malformed_or_repeated_anchor")
    import hashlib

    positions, full_positions = {}, {}
    raw_set = set(raw)
    chain_digest = hashlib.sha256()
    # One scan: retain only raw-anchor projections, build the exact-key index
    # by full hash, and digest the chain without a second full-hash dictionary.
    for index, h in enumerate(hashes, 1):
        chain_digest.update(h)
        end = index * step
        if h in full_positions:
            raise ValueError("repeated_request_hash")
        full_positions[h] = end
        value = int.from_bytes(h[-8:], "big")
        if value in raw_set:
            if value in positions:
                raise ValueError("foreign_or_ambiguous_anchor")
            positions[value] = end
    ends = []
    for value in raw:
        end = positions.get(value)
        if end is None:
            raise ValueError("foreign_or_ambiguous_anchor")
        if end % anchor_step or (ends and end <= ends[-1]):
            raise ValueError("unaligned_or_unordered_anchor")
        ends.append(end)
    cutoff = ends[-1]
    effective, seen = list(raw), set(raw)
    effective_digest = hashlib.sha256()
    for value in raw:
        effective_digest.update(value.to_bytes(8, "big"))
    every_position = min(span for _, span in groups) == step
    spans = tuple(set(span for _, span in groups))
    for index in range(cutoff // step):
        end = (index + 1) * step
        if every_position or any(end % span == 0 for span in spans):
            value = int.from_bytes(hashes[index][-8:], "big")
            if value not in seen:
                seen.add(value)
                effective.append(value)
                effective_digest.update(value.to_bytes(8, "big"))
    return dict(
        source_endpoint=source,
        raw_hint_hashes=list(raw),
        effective_hint_hash_count=len(effective),
        effective_hint_hashes_sha256=effective_digest.hexdigest(),
        anchor_positions=ends,
        covered_through_tokens=cutoff,
        tokens_per_hash=step,
        group_chunks=groups,
        chain_hash_count=len(hashes),
        chain_sha256=chain_digest.hexdigest(),
        full_chain_reconstructed=False,
        anchor_tokens=anchor_step,
        hash_codec="vllm-u64-big-endian-low64",
        _effective_hashes=effective,
        _positions=full_positions,
        _groups=dict(groups),
    )


class _FrameworkPinAdapter:
    """Adapt KVCR framework pin callbacks to primary-tier store jobs."""

    def __init__(self, tier: "KVCRSecondaryTierManager") -> None:
        self._tier = tier
        self._next_request_id = 0
        self._next_pin_request_id = 0
        self._pending_pins: dict[PinRequestId, tuple[BlockKey, ...]] = {}
        self._completed_pins: list[tuple[PinRequestId, PinResult]] = []
        self._pin_jobs: dict[str, int] = {}
        self._pin_job_results: list[JobResult] = []

    def request_pin(self, keys: Collection[BlockKey]) -> PinRequestId:
        # The parent callback is valid only from serve_external_requests().
        request = PinRequestId(self._next_pin_request_id)
        self._next_pin_request_id += 1
        self._pending_pins[request] = tuple(keys)
        return request

    def process_pending(self, parent: ParentManager) -> None:
        pending = self._pending_pins
        self._pending_pins = {}
        self._completed_pins.extend(
            (request, self._resolve_pin(parent, keys))
            for request, keys in pending.items()
        )

    def poll_pin_results(self) -> list[tuple[PinRequestId, PinResult]]:
        completed = self._completed_pins
        self._completed_pins = []
        return completed

    def cancel_pin_request(self, request: PinRequestId) -> None:
        self._pending_pins.pop(request, None)

    def _resolve_pin(
        self, parent: ParentManager, keys: Collection[BlockKey]
    ) -> tuple[str, dict[BlockKey, list[MemoryRef] | None]] | None:
        offload_keys = tuple(OffloadKey(bytes(key)) for key in keys)
        request_id = f"kvcr-source:{self._next_request_id}"
        self._next_request_id += 1
        req_context = ReqContext(req_id=request_id)
        started = False
        job: TransferJob | None = None
        pin_handle: str | None = None
        try:
            parent.on_new_request(req_context)
            started = True
            hit_keys = tuple(
                key
                for key in offload_keys
                if parent.lookup(key, req_context) is LookupResult.HIT
            )
            if not hit_keys:
                return None

            job = parent.create_store_job(hit_keys, req_context)
            job_keys = tuple(job.keys)
            chunk_ids = tuple(int(chunk_id) for chunk_id in job.chunk_ids)
            if len(job_keys) != len(chunk_ids) or set(job_keys) != set(hit_keys):
                return None

            descriptors: dict[BlockKey, list[MemoryRef] | None] = {
                BlockKey(bytes(key)): None for key in offload_keys
            }
            descriptors.update(
                (
                    BlockKey(bytes(key)),
                    [self._tier._make_descriptor(chunk_id)],
                )
                for key, chunk_id in zip(job_keys, chunk_ids)
            )
            pin_handle = f"kvcr-job-{job.job_id}"
            self._pin_jobs[pin_handle] = job.job_id
            return pin_handle, descriptors
        except Exception:
            logger.warning("KVCR source acquisition failed", exc_info=True)
            return None
        finally:
            if started:
                try:
                    parent.on_request_finished(req_context)
                except Exception:
                    logger.warning("KVCR source request cleanup failed", exc_info=True)
            if job is not None and pin_handle is None:
                self._pin_job_results.append(
                    JobResult(job_id=job.job_id, success=False)
                )

    def release_pin(self, pin_handle: str) -> bool:
        job_id = self._pin_jobs.pop(pin_handle, None)
        if job_id is not None:
            self._pin_job_results.append(JobResult(job_id=job_id, success=True))
        return True

    def has_active_pins(self) -> bool:
        return bool(self._pin_jobs)

    def take_pin_job_results(self) -> list[JobResult]:
        results = self._pin_job_results
        self._pin_job_results = []
        return results


class _VllmKeyAdapter:
    """Adapt vLLM's local key format to KVCR."""

    def encode(self, framework_key: object) -> BlockKey:
        if not isinstance(framework_key, bytes):
            raise TypeError("vLLM offload keys must be bytes")
        return BlockKey(framework_key)

    def decode(self, key: BlockKey) -> int | bytes:
        block_hash = BlockHash(get_offload_block_hash(OffloadKey(key)))
        return maybe_convert_block_hash(block_hash)


# Job ID, remaining blocks, aggregate success, and successful load keys.
_JobState = tuple[int, int, bool, set[OffloadKey] | None]


class KVCRSecondaryTierManager(SecondaryTierManager):
    """Secondary tier wrapper around the KVCR KV P2P API."""

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        if not extra_config.get("enable_telemetry", False):
            return {}
        return _kvcr_metric_definitions()

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        router_capabilities: Iterable[str] | None = None,
        control_host: str = "0.0.0.0",
        control_ports: list[int] | None = None,
        control_advertise_host: str | None = None,
        eager_ctrl_connect: bool = True,
        opportunistic_query: bool = False,
        enable_telemetry: bool = False,
        operation_timeout_ms: int = 1000,
        abandon_timeout_ms: int = 5000,
        metadata_retry_interval_ms: int = 100,
        secondary_g2_slots: int = 0,
        kvcr_service_socket_path: str | None = None,
        compatibility_digest: str | None = None,
        policy: str | None = None,
        g3: dict[str, Any] | None = None,
        local_dram_backend: str = "UCX",
        remote_fw_dram_backend: str = "UCX",
        backpressure_detector: "BackpressureDetector | None" = None,
        *,
        no_pinning_mode: bool = False,
    ) -> None:
        super().__init__(
            offloading_spec, primary_kv_view, tier_type, backpressure_detector
        )
        selected_policy = _resolve_policy(policy)
        if (kvcr_service_socket_path is None) != (compatibility_digest is None):
            raise ValueError(
                "kvcr_service_socket_path and compatibility_digest must be "
                "configured together"
            )
        missing = _REQUIRED_ROUTER_CAPABILITIES - set(router_capabilities or ())
        if missing:
            raise ValueError(
                f"KVCR secondary tier requires router capabilities: {sorted(missing)}"
            )
        events_enabled = offloading_spec.kv_events_config.enable_kv_cache_events
        self_describing_events = (
            offloading_spec.kv_events_config.self_describing_kv_events
        )
        if (
            events_enabled
            and (
                secondary_g2_slots > 0
                or kvcr_service_socket_path is not None
                or g3 is not None
            )
            and not self_describing_events
        ):
            raise ValueError(
                "KVCR local G2/G3 inventory requires self_describing_kv_events"
            )
        self._key_adapter: KeyAdapter = _VllmKeyAdapter()
        advertise_host = control_advertise_host or socket.gethostname()
        dp_local_rank = offloading_spec.config.parallel.data_parallel_rank_local
        if dp_local_rank is None:
            if control_ports is None or len(control_ports) != 1:
                raise ValueError(
                    "control_ports must contain exactly one port when "
                    "data_parallel_rank_local is unset"
                )
            dp_local_rank = 0
        if control_ports is None or not 0 <= dp_local_rank < len(control_ports):
            raise ValueError(
                "control_ports must contain a port for "
                f"local data-parallel rank {dp_local_rank}"
            )
        if any(not 1 <= port <= 65535 for port in control_ports):
            raise ValueError("control_ports must be between 1 and 65535")
        control = ZmqPeerControlChannel(
            control_host,
            control_ports[dp_local_rank],
            advertise_host,
        )
        # Give colocated workers distinct transport listen ports.
        with socket.socket() as _s:
            _s.bind(("", 0))
            _nixl_listen_port = _s.getsockname()[1]
        if secondary_g2_slots < 0:
            raise ValueError("secondary_g2_slots must be non-negative")
        if (
            secondary_g2_slots or kvcr_service_socket_path is not None
        ) and not events_enabled:
            logger.warning(
                "KVCR local DRAM is enabled, but KV cache events are disabled; "
                "local DRAM inventory will not be published"
            )
        local_mapping: mmap.mmap | None = None
        local_dram: LocalDramOptions | None = None
        if secondary_g2_slots:
            if kvcr_service_socket_path is not None:
                logger.warning(
                    "secondary_g2_slots is ignored when "
                    "kvcr_service_socket_path is configured"
                )
            else:
                local_mapping = mmap.mmap(
                    -1, secondary_g2_slots * self.block_size_bytes
                )
                local_dram = LocalDramOptions(
                    pools=[
                        (
                            "",
                            ctypes.addressof(ctypes.c_char.from_buffer(local_mapping)),
                            len(local_mapping),
                        )
                    ],
                    backend=local_dram_backend,
                )
        guard_config = (
            KVCRGuardConfig(
                kvcr_service_socket_path=kvcr_service_socket_path,
                guard_index=dp_local_rank,
                compatibility_digest=compatibility_digest,
            )
            if kvcr_service_socket_path is not None and compatibility_digest is not None
            else None
        )
        g3_config = None
        if g3 is not None:
            g3_values = dict(g3)
            g3_values["paths"] = tuple(Path(path) for path in g3_values["paths"])
            g3_config = G3Options(**g3_values)
        nixl_agent_name = f"KVCR-{uuid.uuid4()}"
        self._framework_pin_adapter = _FrameworkPinAdapter(self)
        self._inventory_events: list[OffloadingEvent] = []

        try:
            self._kvcr = KVCR(
                KVCRConfig(
                    pool_layouts=[("", self.block_size_bytes)],
                    enable_telemetry=enable_telemetry,
                    operation_timeout_ms=operation_timeout_ms,
                    abandon_timeout_ms=abandon_timeout_ms,
                    nixl_agent_name=nixl_agent_name,
                    nixl_listen_port=_nixl_listen_port,
                ),
                KVCRBindings(
                    request_pin=self._framework_pin_adapter.request_pin,
                    poll_pin_results=self._framework_pin_adapter.poll_pin_results,
                    release_pin=self._framework_pin_adapter.release_pin,
                    cancel_pin_request=(self._framework_pin_adapter.cancel_pin_request),
                    framework_control=control,
                    key_adapter=self._key_adapter,
                    inventory_sink=(self._record_inventory if events_enabled else None),
                    stats_factory=(
                        OffloadingConnectorStats if enable_telemetry else None
                    ),
                    policy=selected_policy,
                ),
                KVCRBackendConfigs(
                    framework_regions=[
                        RegionDescriptor(
                            addr=ctypes.addressof(
                                ctypes.c_char.from_buffer(primary_kv_view)
                            ),
                            size=self.block_size_bytes,
                            count=len(primary_kv_view),
                        )
                    ],
                    local_dram=local_dram,
                    g3=g3_config,
                    remote_fw_dram=RemoteFWDramOptions(
                        eager_ctrl_connect=eager_ctrl_connect,
                        opportunistic_query=opportunistic_query,
                        metadata_retry_interval_ms=metadata_retry_interval_ms,
                        backend=remote_fw_dram_backend,
                        no_pinning_mode=no_pinning_mode,
                    ),
                ),
                guard_config,
            )
        except BaseException:
            control.close()
            if local_mapping is not None:
                local_mapping.close()
            raise
        self._local_dram_mmap = local_mapping
        self._finished_jobs: list[JobResult] = []
        self._jobs_by_op: dict[OpHandle, _JobState] = {}
        self._load_job_parts: dict[int, list] = {}

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        closure = self._ensure_hint_closure(req_context)
        remote_allowed = closure is None or (
            closure.valid and _hint_key_position(closure.proof, bytes(key)) is not None
        )
        block_key = self._key_adapter.encode(key)
        # Missing remote authority never suppresses local Guard residency.
        status, _ = self._kvcr.query(
            (block_key,), req_context.req_id if remote_allowed else None
        )[0]
        if status is QueryStatus.FETCHING:
            return LookupResult.RETRY
        if status in (QueryStatus.HIT, QueryStatus.FETCHABLE):
            return LookupResult.HIT
        return LookupResult.MISS

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        blocks = {
            self._key_adapter.encode(key): [self._make_descriptor(int(chunk_id))]
            for key, chunk_id in zip(
                job_metadata.keys, job_metadata.chunk_ids, strict=True
            )
        }
        if not blocks:
            self._finished_jobs.append(
                JobResult(job_id=job_metadata.job_id, success=True)
            )
            return
        ctx = job_metadata.req_context
        state = self._ensure_hint_closure(ctx)
        remote, local = {}, {}
        for key, refs in blocks.items():
            allowed = state is None or (
                state.valid and _hint_key_position(state.proof, bytes(key)) is not None
            )
            (remote if allowed else local)[key] = refs
        partitions = [
            (part, request_id)
            for part, request_id in ((remote, ctx.req_id), (local, None))
            if part
        ]
        self._load_job_parts[job_metadata.job_id] = [len(partitions), True, set()]
        for part, request_id in partitions:
            # Ineligible local hits stay local-only even if evicted between
            # lookup and dispatch. Native cleanup ownership is unchanged.
            op_handle = self._kvcr.deliver(part, request_id=request_id)
            self._jobs_by_op[op_handle] = (job_metadata.job_id, len(part), True, set())
            output = getattr(self._kvcr._core, "_compact_capture", None)
            if output is not None and state is not None:
                keys = list(part)
                for offset in range(0, len(keys), 128):
                    chunk = keys[offset : offset + 128]
                    output.journal(
                        dict(
                            event="hint_prefix_selected",
                            closure_schema=1,
                            proof_id=state.proof_id,
                            agent=self._kvcr._core.nixl_agent_name,
                            request_id=ctx.req_id,
                            target_op=op_handle,
                            remote_allowed=request_id is not None,
                            source_endpoint=state.proof["source_endpoint"]
                            if state.valid
                            else None,
                            key_offset=offset,
                            total_keys=len(keys),
                            keys=chunk,
                            end_tokens=[
                                _hint_key_position(state.proof, bytes(k))
                                if state.valid
                                else None
                                for k in chunk
                            ],
                        )
                    )

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        blocks = {
            self._key_adapter.encode(key): [self._make_descriptor(int(chunk_id))]
            for key, chunk_id in zip(
                job_metadata.keys, job_metadata.chunk_ids, strict=True
            )
        }
        if not blocks:
            self._finished_jobs.append(
                JobResult(job_id=job_metadata.job_id, success=True)
            )
            return
        op_handle = self._kvcr.deposit(blocks)
        self._jobs_by_op[op_handle] = (
            job_metadata.job_id,
            len(blocks),
            True,
            None,
        )

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        results = self._finished_jobs
        self._finished_jobs = []
        results.extend(self._poll_finished_jobs())
        return results

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = cast(OffloadingConnectorStats | None, self._kvcr.get_stats())
        if stats is not None:
            # Add the framework namespace only at the vLLM boundary.
            for metrics in stats.data.values():
                for name in tuple(metrics):
                    metrics[_vllm_metric_name(name)] = metrics.pop(name)
        return stats

    @override
    def serve_external_requests(self, parent: ParentManager) -> None:
        self._framework_pin_adapter.process_pending(parent)

    def _poll_finished_jobs(self) -> list[JobResult]:
        results: list[JobResult] = []
        for op_handle, entries in self._kvcr.poll_completed():
            job_state = self._jobs_by_op.get(op_handle)
            if job_state is None:
                continue
            job_id, remaining, success, successful_keys = job_state
            remaining -= len(entries)
            success = success and all(entry.success for entry in entries.values())
            if successful_keys is not None:
                successful_keys.update(
                    (
                        OffloadKey(bytes(key))
                        for key, entry in entries.items()
                        if entry.success
                    )
                )
            if remaining > 0:
                self._jobs_by_op[op_handle] = (
                    job_id,
                    remaining,
                    success,
                    successful_keys,
                )
                continue
            self._jobs_by_op.pop(op_handle, None)
            part = (
                getattr(self, "_load_job_parts", {}).get(job_id)
                if successful_keys is not None
                else None
            )
            if part is not None:
                part[0] -= 1
                part[1] = part[1] and success
                part[2].update(successful_keys)
                if part[0]:
                    continue
                self._load_job_parts.pop(job_id)
                success, successful_keys = part[1], part[2]
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    successful_keys=successful_keys if not success else None,
                )
            )
        results.extend(self._framework_pin_adapter.take_pin_job_results())
        return results

    @override
    def has_pending_work(self) -> bool:
        # Keep scheduler polling: control messages may arrive without a load job.
        return True

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        events = self._inventory_events
        self._inventory_events = []
        return events

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        if (hint := req_context.kv_hints) is not None:
            try:
                state = _HintClosureState(msgspec.to_builtins(hint))
                req_context.set_state(state)
                self._kvcr.submit_hint(
                    request_id=req_context.req_id, hints=state.raw_hint
                )
            except (ValueError, TypeError):
                self._kvcr.discard_hint(req_context.req_id)
        return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        self._kvcr.discard_hint(req_context.req_id)

    @override
    def drain_jobs(self) -> None:
        while self._jobs_by_op or self._framework_pin_adapter.has_active_pins():
            drained = self._poll_finished_jobs()
            if drained:
                self._finished_jobs.extend(drained)
            else:
                time.sleep(0.001)

    @override
    def shutdown(self) -> None:
        self.drain_jobs()
        # TODO: Keep registered buffers alive if KVCR.close() fails.
        self._kvcr.close()
        if self._local_dram_mmap is not None:
            self._local_dram_mmap.close()
            self._local_dram_mmap = None

    def _record_inventory(self, event: InventoryEvent) -> None:
        if event.tier is CacheTier.LOCAL_G2:
            medium = Medium.CPU
        elif event.tier is CacheTier.G3:
            medium = Medium.STORAGE
        else:
            raise ValueError(f"unsupported KVCR inventory tier: {event.tier}")
        self._inventory_events.append(
            OffloadingEvent(
                keys=[OffloadKey(bytes(key)) for key in event.keys],
                medium=medium,
                removed=event.removed,
                ownership="kvcr",
                removal_expected=True,
            )
        )

    def _make_descriptor(self, chunk_id: int) -> MemoryRef:
        return MemoryRef(
            end_point_name=self._kvcr.config.nixl_agent_name,
            element_index=chunk_id,
        )

    def _ensure_hint_closure(self, req_context):
        state = req_context.get_state(_HintClosureState)
        if state is None:
            return None
        chain = _get_request_hash_chain(req_context)
        if chain is None:
            return state  # No trusted scheduler chain yet: prompt explicit miss.
        if chain is state.chain:
            return state
        previous = state.chain
        state.chain = chain
        try:
            if chain.request_id != req_context.req_id or not chain.coherent:
                raise ValueError("request_chain_scope_changed")
            if state.valid and previous is not None:
                if (
                    chain.tokens_per_hash != previous.tokens_per_hash
                    or chain.group_chunks != previous.group_chunks
                    or chain.anchor_tokens != previous.anchor_tokens
                    or len(chain.block_hashes) < len(previous.block_hashes)
                ):
                    raise ValueError("request_chain_geometry_changed")
                if any(
                    int.from_bytes(h[-8:], "big") in state.raw_anchors
                    for h in chain.block_hashes[len(previous.block_hashes) :]
                ):
                    raise ValueError("appended_ambiguous_anchor")
                return state  # Stable prefix; no recopy, resubmit or reserialization.
            proof = _strict_hint_prefix_closure(chain, state.raw_hint)
            probe = chain.block_hashes[0] + chain.group_chunks[0][0].to_bytes(4, "big")
            decoded = self._key_adapter.decode(probe)
            if type(decoded) is not int or decoded != int.from_bytes(
                probe[24:32], "big"
            ):
                raise ValueError("unsupported_runtime_hash_codec")
            expanded = copy.deepcopy(state.raw_hint)
            action = next(
                a for a in expanded["actions"] if a.get("action_type") == "kv.fetch"
            )
            action["payload"]["block_hashes"] = proof["_effective_hashes"]
            self._kvcr.submit_hint(request_id=req_context.req_id, hints=expanded)
            state.proof = proof
            state.raw_anchors = frozenset(proof["raw_hint_hashes"])
            state.valid, state.reason = True, "verified_request_prefix"
            self._record_hint_closure(req_context, state, proof)
        except (ValueError, TypeError, AttributeError) as error:
            state.valid, state.reason = False, str(error)[:160]
            state.proof = None
            self._kvcr.discard_hint(req_context.req_id)
            self._record_hint_closure(req_context, state)
        return state

    def _record_hint_closure(self, req_context, state, proof=None):
        output = getattr(self._kvcr._core, "_compact_capture", None)
        if output is None:
            return
        agent = self._kvcr._core.nixl_agent_name
        proof_id = uuid.uuid4().hex
        if state.valid:
            state.proof_id = proof_id
        record = dict(
            event="hint_prefix_closure",
            closure_schema=1,
            proof_id=proof_id,
            agent=agent,
            request_id=req_context.req_id,
            valid=state.valid,
            reason=state.reason,
        )
        if proof is not None:
            record.update((k, v) for k, v in proof.items() if not k.startswith("_"))
        output.journal(record)
        output.emit(
            dict(
                event="hint_prefix_closure",
                closure_schema=1,
                proof_id=proof_id,
                identity=agent,
                agent=agent,
                request_id=req_context.req_id,
                valid=state.valid,
                reason=state.reason,
                source_endpoint=(proof or {}).get("source_endpoint"),
                raw_hint_hash_count=len((proof or {}).get("raw_hint_hashes", ())),
                effective_hint_hash_count=(proof or {}).get(
                    "effective_hint_hash_count", 0
                ),
                covered_through_tokens=(proof or {}).get("covered_through_tokens"),
            )
        )
