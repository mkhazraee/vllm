# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests executing production scheduler, allocator, policy, and tiering methods.

Load definitions with postponed annotations to avoid GPU imports. Request data,
reachability, and transfer I/O are synthetic; allocator/refcounts, the LRU policy,
stats, and Prometheus recording use repository source.
"""

import ast
import collections
import ctypes
import dataclasses
import enum
import heapq
import itertools
import logging
import math
import sys
import time
import types
import unittest
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "vllm"
SCHED = SRC / "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
MANAGER = SRC / "v1/kv_offload/tiering/manager.py"


class Context:
    def __init__(self, req_id):
        self.req_id, self.states = req_id, {}

    def get_state(self, cls):
        return self.states.get(cls)

    def set_state(self, obj):
        self.states[type(obj)] = obj

    def get_offload_key_position(self, key):
        return None


class Spec:
    def __init__(self, ids, **kw):
        self.chunk_ids = self.block_ids = list(ids)
        self.__dict__.update(kw)


class Result(enum.Enum):
    MISS = 0
    HIT = 1
    HIT_PENDING = 2
    RETRY = 3


class RequestStatus(enum.Enum):
    RUNNING = 0
    FINISHED_ABORTED = 1
    FINISHED_STOPPED = 2


class SlidingWindowSpec:
    def __init__(self, sliding_window=1, extra_retained_tokens=0):
        self.sliding_window = sliding_window
        self.extra_retained_tokens = extra_retained_tokens


class Quiet:
    def __getattr__(self, name):
        return lambda *a, **kw: None


class Attrs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


mod = types.ModuleType("put_batching_actual")
sys.modules[mod.__name__] = mod
G = mod.__dict__
for source in (collections, itertools, dataclasses):
    G.update({k: v for k, v in vars(source).items() if not k.startswith("__")})
G.update(
    dict(
        Any=Any,
        NamedTuple=NamedTuple,
        ctypes=ctypes,
        ABC=ABC,
        abstractmethod=abstractmethod,
        heapq=heapq,
        np=np,
        time=time,
        lcm=math.lcm,
        logger=logging.getLogger("batch-test"),
        override=lambda f: f,
        OffloadingManager=object,
        ParentManager=object,
        ReqContext=Context,
        LookupResult=Result,
        OffloadPolicy=Attrs(REQUEST_LEVEL=1, CHUNK_LEVEL=0),
        Medium=Attrs(CPU="cpu"),
        CacheHitSource=Attrs(HOST="host"),
        CPULoadStoreSpec=Spec,
        GPULoadStoreSpec=Spec,
        PrepareStoreOutput=Attrs,
        OffloadingEvent=Attrs,
        RequestOffloadingContext=Attrs,
        OffloadingConnectorStats=None,
        _ConnectorMetricName=Attrs(ALLOCATION_FAILURE="allocation_failure"),
        RequestStatus=RequestStatus,
        SlidingWindowSpec=SlidingWindowSpec,
        FullAttentionSpec=type("FullAttentionSpec", (), {}),
        MambaSpec=type("MambaSpec", (), {}),
        ChunkedLocalAttentionSpec=type("ChunkedLocalAttentionSpec", (), {}),
        get_offload_group_idx=lambda key: key[0],
        cdiv=lambda a, b: -(-a // b),
        round_down=lambda a, b: a - a % b,
        TieringMetricsTracker=lambda **kw: Quiet(),
        TransferJob=Attrs,
        OffloadingConnectorMetadata=Attrs,
        ScheduleEndContext=Attrs,
        yield_req_data=lambda out: (),
        OffloadingWorkerMetadata=Attrs,
        CPUOffloadingMetrics=Attrs(
            CPU_CACHE_USAGE_PERC="usage",
            CPU_ALLOCATION_SIZE="sizes",
            CPU_CACHE_WRITE_USAGE_PERC="writes",
            CPU_CACHE_READ_USAGE_PERC="reads",
        ),
    )
)


def load(path, names=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and (names is None or n.name in names)
    ]
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)
    ] + nodes
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), G)


load(
    ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/metrics.py",
    {"KVConnectorStats"},
)
load(
    ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py",
    {"_MetricType", "_StatsKey", "OffloadingConnectorStats"},
)
Stats = G["OffloadingConnectorStats"]
load(ROOT / "vllm/v1/kv_offload/cpu/policies/base.py")
load(ROOT / "vllm/v1/kv_offload/cpu/policies/lru.py")
G["CachePolicyFactory"] = Attrs(get_cache_policy_cls=lambda *a: G["LRUCachePolicy"])
load(SRC / "v1/kv_offload/cpu/manager.py")
load(
    ROOT / "vllm/v1/kv_offload/base.py",
    {
        "OffloadingMetricMetadata",
        "OffloadingCounterMetadata",
        "OffloadingGaugeMetadata",
        "OffloadingHistogramMetadata",
    },
)
G["KVConnectorPromMetrics"] = object
load(
    ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py",
    {"OffloadPromMetrics"},
)
load(MANAGER)
Tiering = G["TieringOffloadingManager"]
Primary = G["CPUPrimaryTierOffloadingManager"]
load(SCHED)
Scheduler = G["OffloadingConnectorScheduler"]


class Tier:
    bp_detector = None
    cache_hit_source = "fixture"
    tier_type = "fixture"

    def __init__(self):
        self.jobs, self.done = [], []

    def submit_store(self, job):
        self.jobs.append(job)

    def submit_load(self, job):
        self.jobs.append(job)

    def get_finished_jobs(self):
        done, self.done = self.done, []
        return done

    def on_request_finished(self, ctx):
        pass

    def drain_jobs(self):
        for job in self.jobs:
            self.done.append(Attrs(job_id=job.job_id, success=True))
        self.jobs.clear()

    def on_schedule_end(self, ctx):
        pass

    def serve_external_requests(self, parent):
        pass

    def has_pending_work(self):
        return bool(self.jobs)

    def get_stats(self):
        return None


def manager():
    p = Primary.__new__(Primary)
    G["CPUOffloadingManager"].__init__(p, num_chunks=873)
    p.complete_read = p.complete_load
    p.prepare_write = p.prepare_store
    p.complete_write = p.complete_store
    p.get_kv_memoryview = lambda: memoryview(bytearray(873))
    tier = Tier()
    m = Tiering(p, [tier])
    return m, tier


def make_scheduler(n=5384, group_chunks=(1,), enabled=True, requests=1):
    m, tier = manager()
    s = Scheduler.__new__(Scheduler)
    groups = tuple(
        G["GroupOffloadConfig"](
            i,
            t,
            t,
            1,
            None,
            None,
            G["FullAttentionSpec"](),
            Attrs(reachable_block_mask=lambda **kw: None),
        )
        for i, t in enumerate(group_chunks)
    )
    s.config = G["SchedulerOffloadConfig"](groups, 1, 1, 1, False, False)
    s.manager, s._connector_stats = m, Stats()
    s._put_batch_chunks = 256 if enabled else 0
    s._put_retry_after = None
    s._put_deferred = {}
    s._put_retry_gate = {}
    if enabled and hasattr(m, "configure_put_batching"):
        m.configure_put_batching(256, 512, 128)
    s._req_status, s._jobs, s._block_id_to_pending_jobs = {}, {}, {}
    s._current_batch_allocated_block_ids = set()
    s._current_batch_jobs_to_flush = set()
    s._current_batch_load_jobs = {}
    s._job_counter = s._stale_job_threshold = 0
    s._chunks_being_loaded = None
    s._events_tracker = Quiet()
    s._sliding_window_groups = ()
    s._partial_tail_block_size = 0
    s._cow_source_groups = frozenset()
    for i in range(requests):
        req = Attrs(
            request_id=str(i),
            num_tokens=n,
            num_prompt_tokens=n,
            num_computed_tokens=n,
            status=RequestStatus.RUNNING,
            shared_prefix_boundary=None,
            kv_transfer_params=None,
        )
        req.is_finished = lambda req=req: req.status != RequestStatus.RUNNING
        ctx = Context(str(i))
        state = G["RequestOffloadState"](s.config, req, ctx, Attrs(policy=0))
        for g, st in zip(groups, state.group_states):
            st.offload_keys = [
                (g.group_idx, i, k) for k in range(n // g.tokens_per_chunk)
            ]
            st.block_ids = list(
                range(1 + i * 10000, 1 + i * 10000 + len(st.offload_keys))
            )
        s._req_status[str(i)] = state
        m._req_state[str(i)] = G["RequestState"](ctx)
    return s, m, tier


def output(s):
    return Attrs(
        num_scheduled_tokens={r: 0 for r in s._req_status},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        kv_connector_block_state=None,
        kv_cache_block_copies=[],
    )


def drain(s, m, tier, jobs):
    for jid in jobs:
        job = s._jobs.pop(jid)
        st = s._req_status[job.req_id]
        m.complete_store(job.keys, st.req_context)
        st.transfer_jobs.remove(jid)
    for job in tier.jobs:
        tier.done.append(Attrs(job_id=job.job_id, success=True))
    tier.jobs.clear()
    m._process_finished_jobs()
    m._processed_jobs_this_step = False


class BatchingTests(unittest.TestCase):
    def test_5384_chunks_progress_in_22_batches(self):
        s, m, t = make_scheduler()
        seen = set()
        sizes = []
        for _ in range(22):
            jobs = s._build_store_jobs(output(s))
            self.assertEqual(len(jobs), 1)
            keys = next(iter(s._jobs.values())).keys
            self.assertFalse(seen & keys)
            seen.update(keys)
            sizes.append(len(keys))
            drain(s, m, t, jobs)
        self.assertEqual(sizes, [256] * 21 + [8])
        self.assertEqual(len(seen), 5384)
        self.assertEqual(s._req_status["0"].group_states[0].next_stored_chunk_idx, 5384)

    def test_busy_credits_return_none_and_live_through_cascade_and_reader(self):
        s, m, t = make_scheduler(2000)
        ctx = s._req_status["0"].req_context
        a = m.prepare_store([(0, 0, k) for k in range(256)], ctx)
        b = m.prepare_store([(0, 0, k) for k in range(256, 512)], ctx)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertIsNone(m.prepare_store([(0, 0, 512)], ctx))
        m.complete_store(a.keys_to_store, ctx)
        self.assertIsNone(m.prepare_store([(0, 0, 512)], ctx))
        m.prepare_load(a.keys_to_store, ctx)
        job = t.jobs.pop()
        t.done.append(Attrs(job_id=job.job_id, success=False))
        m._process_finished_jobs()
        m._processed_jobs_this_step = False
        self.assertIsNone(m.prepare_store([(0, 0, 512)], ctx))
        m.complete_load(a.keys_to_store, ctx)
        self.assertIsNotNone(m.prepare_store([(0, 0, 512)], ctx))

    def test_mixed_groups_share_token_frontier_without_holes(self):
        s, m, t = make_scheduler(600, (1, 2))
        jobs = s._build_store_jobs(output(s))
        states = s._req_status["0"].group_states
        self.assertEqual([st.next_stored_chunk_idx for st in states], [170, 85])
        self.assertEqual(len(next(iter(s._jobs.values())).keys), 255)
        drain(s, m, t, jobs)
        s._build_store_jobs(output(s))
        self.assertEqual([st.next_stored_chunk_idx for st in states], [340, 170])

    def test_c72_fair_retries(self):
        s, m, t = make_scheduler(1000, requests=72)
        served = set()
        for _ in range(36):
            jobs = s._build_store_jobs(output(s))
            served.update(s._jobs[j].req_id for j in jobs)
            drain(s, m, t, jobs)
        self.assertEqual(len(served), 72)

    def test_disabled_retains_atomic_oversized_rejection(self):
        s, m, t = make_scheduler(enabled=False)
        self.assertEqual(s._build_store_jobs(output(s)), {})
        self.assertEqual(s._req_status["0"].group_states[0].next_stored_chunk_idx, 0)

    def test_failed_d2h_does_not_publish_ready_and_returns_credit(self):
        s, m, t = make_scheduler(1000)
        ctx = s._req_status["0"].req_context
        keys = [(0, 0, k) for k in range(256)]
        m.prepare_store(keys, ctx)
        m.complete_store(keys, ctx, False)
        self.assertTrue(all(m.primary_tier.lookup(k, ctx) is Result.MISS for k in keys))
        self.assertEqual(t.jobs, [])
        self.assertTrue(
            hasattr(m, "get_put_admission_budget"), "missing admission credits"
        )
        self.assertEqual(m.get_put_admission_budget(), 512)

    def test_get_reserve_and_atomic_no_subset(self):
        s, m, t = make_scheduler(1000)
        ctx = s._req_status["0"].req_context
        p = m.primary_tier
        keys = [(9, 0, k) for k in range(700)]
        result = p.prepare_store(keys, ctx)
        self.assertIsNotNone(result)
        self.assertTrue(
            hasattr(m, "get_put_admission_budget"), "missing admission reserve"
        )
        self.assertEqual(m.get_put_admission_budget(), 45)
        self.assertIsNone(m.prepare_store([(0, 0, k) for k in range(46)], ctx))
        self.assertTrue(all(p.lookup((0, 0, k), ctx) is Result.MISS for k in range(46)))

    def test_duplicate_keys_have_one_gpu_source_per_destination(self):
        s, m, t = make_scheduler(10)
        st = s._req_status["0"].group_states[0]
        st.offload_keys[1] = st.offload_keys[0]
        jobs = s._build_store_jobs(output(s))
        job = next(iter(jobs.values()))
        self.assertEqual(len(job.src_spec.block_ids), len(job.dst_spec.chunk_ids))
        self.assertEqual(st.next_stored_chunk_idx, 10)

    def test_finished_tail_abandoned_and_accepted_fences_drain_after_all_workers(self):
        s, m, t = make_scheduler(1000)
        s.config = s.config._replace(num_workers=2)
        st = s._req_status["0"]
        st.req.status = RequestStatus.FINISHED_STOPPED
        out = output(s)
        out.num_scheduled_tokens = {}
        out.finished_req_ids = {"0"}
        meta = s.build_connector_meta(out)
        self.assertEqual(len(meta.store_jobs), 1)
        jid = next(iter(meta.store_jobs))
        job = s._jobs[jid]
        self.assertEqual(len(job.keys), 256)
        self.assertEqual(len(s._block_id_to_pending_jobs), 256)
        self.assertEqual(
            s._connector_stats.reduce()["vllm:guard_put_abandoned_chunks"], 744
        )
        self.assertEqual(s._put_deferred, {})
        # Partial worker completion must retain every GPU fence and primary write.
        update = Attrs(
            kv_connector_worker_meta=Attrs(
                completed_jobs={jid: 1}, transfer_stats=Attrs(is_empty=lambda: True)
            )
        )
        s.update_connector_output(update)
        self.assertEqual(len(s._block_id_to_pending_jobs), 256)
        s.update_connector_output(update)
        self.assertEqual(s._block_id_to_pending_jobs, {})
        self.assertEqual(s._req_status, {})
        self.assertEqual(m.get_put_admission_budget(), 256)
        cascade = t.jobs.pop()
        t.done.append(Attrs(job_id=cascade.job_id, success=True))
        m._process_finished_jobs()
        self.assertEqual(m.get_put_admission_budget(), 512)

    def test_finished_sources_reused_in_same_step_never_stored(self):
        s, m, t = make_scheduler(20)
        st = s._req_status["0"]
        st.req.status = RequestStatus.FINISHED_ABORTED
        s._current_batch_allocated_block_ids = set(range(1, 21))
        jobs = s._build_store_jobs(output(s))
        self.assertEqual(jobs, {})
        self.assertEqual(st.group_states[0].next_stored_chunk_idx, 20)
        self.assertEqual(
            s._connector_stats.reduce()[
                "vllm:guard_put_source_unavailable_before_admission"
            ],
            20,
        )

    def test_busy_scheduler_does_not_rescan_or_advance(self):
        s, m, t = make_scheduler(2000)
        jobs = s._build_store_jobs(output(s))
        jobs.update(s._build_store_jobs(output(s)))
        st = s._req_status["0"].group_states[0]
        self.assertEqual(st.next_stored_chunk_idx, 512)

        # A source which raises on iteration proves the credit gate precedes scanning.
        class NoRead(list):
            def __getitem__(self, key):
                raise AssertionError("busy source scanned")

        original = st.offload_keys
        st.offload_keys = NoRead(original)
        self.assertEqual(s._build_store_jobs(output(s)), {})
        st.offload_keys = original
        drain(s, m, t, jobs)
        self.assertTrue(s._build_store_jobs(output(s)))
        self.assertEqual(st.next_stored_chunk_idx, 768)

    def test_preemption_abandons_only_unadmitted_tail(self):
        s, m, t = make_scheduler(1000)
        jobs = s._build_store_jobs(output(s))
        out = output(s)
        out.preempted_req_ids = {"0"}
        self.assertEqual(s._build_store_jobs(out), {})
        self.assertEqual(len(s._jobs), 1)
        self.assertEqual(
            s._connector_stats.reduce()["vllm:guard_put_abandoned_chunks"], 744
        )
        self.assertEqual(s._put_deferred, {})
        drain(s, m, t, jobs)
        self.assertEqual(m.get_put_admission_budget(), 512)

    def test_speculative_volatile_tail_and_skips_reconsidered(self):
        s, m, t = make_scheduler(10)
        group = s.config.kv_group_configs[0]._replace(is_eagle_group=True)
        s.config = s.config._replace(kv_group_configs=(group,))
        st = s._req_status["0"]
        st.config = s.config
        st.req.num_prompt_tokens = 4
        st.group_states[0].block_ids[2] = 0
        jobs = s._build_store_jobs(output(s))
        self.assertEqual(st.group_states[0].next_stored_chunk_idx, 9)
        self.assertEqual(len(s._jobs[next(iter(jobs))].keys), 8)
        drain(s, m, t, jobs)
        st.req.status = RequestStatus.FINISHED_STOPPED
        # Finished commit point excludes final sampled token: 9 tokens, not 10.
        st.req.num_tokens = 11
        jobs = s._build_store_jobs(output(s))
        self.assertEqual(st.group_states[0].next_stored_chunk_idx, 10)
        self.assertEqual(len(s._jobs[next(iter(jobs))].keys), 1)

    def test_swa_original_horizon_survives_clipping(self):
        s, m, t = make_scheduler(1000, (1, 2))
        horizons = []

        def reachable(**kw):
            horizons.append(kw["final_segment_end_block"])
            return [i % 2 == 1 for i in range(kw["start_block"], kw["end_block"])]

        group = s.config.kv_group_configs[1]._replace(
            sliding_window_size_in_chunks=1,
            kv_cache_spec=SlidingWindowSpec(),
            manager_cls=Attrs(reachable_block_mask=reachable),
        )
        s.config = s.config._replace(
            kv_group_configs=(s.config.kv_group_configs[0], group), alignment_tokens=2
        )
        st = s._req_status["0"]
        st.config = s.config
        jobs = s._build_store_jobs(output(s))
        self.assertEqual(horizons, [500])
        states = st.group_states
        self.assertEqual(
            states[0].next_stored_chunk_idx, states[1].next_stored_chunk_idx * 2
        )
        self.assertLessEqual(len(s._jobs[next(iter(jobs))].keys), 256)

    def test_minimum_aligned_unit_exception_stays_inside_real_credit(self):
        s, m, t = make_scheduler(600)
        s.config = s.config._replace(alignment_tokens=300)
        jobs = s._build_store_jobs(output(s))
        self.assertEqual(len(s._jobs[next(iter(jobs))].keys), 300)
        self.assertEqual(s._build_store_jobs(output(s)), {})
        self.assertEqual(s._req_status["0"].group_states[0].next_stored_chunk_idx, 300)

    def test_get_stats_exposes_batching_counters(self):
        s, m, t = make_scheduler(1000)
        s._build_store_jobs(output(s))
        # Tier metrics tracker is a no-op; primary and connector stats are real methods.
        result = s.get_stats().reduce()
        self.assertEqual(result["vllm:guard_put_offered_chunks"], 256)
        self.assertEqual(result["vllm:guard_put_admitted_chunks"], 256)
        self.assertEqual(result["vllm:guard_put_deferred_chunks"], 744)
        self.assertEqual(result["vllm:guard_put_staging_chunks"], 256)
        self.assertEqual(result["vllm:guard_put_staging_high_water"], 256)

    def test_allocator_rejection_retries_only_after_progress_or_bounded_liveness(self):
        s, m, t = make_scheduler(1000)
        ctx = s._req_status["0"].req_context
        p = m.primary_tier
        resident = [(9, 0, i) for i in range(873)]
        p.prepare_store(resident, ctx)
        p.complete_store(resident, ctx)
        calls = []
        original = p._policy.evict

        def blocked(n, protected):
            calls.append(n)
            return None

        p._policy.evict = blocked
        self.assertEqual(s._build_store_jobs(output(s)), {})
        for _ in range(5):
            self.assertEqual(s._build_store_jobs(output(s)), {})
        self.assertEqual(len(calls), 1)
        p._policy.evict = original
        for _ in range(16):
            if s._build_store_jobs(output(s)):
                break
        else:
            self.fail("no bounded liveness retry")

    def test_get_promotions_keep_reserved_headroom_and_partial_failure_is_not_ready(
        self,
    ):
        s, m, t = make_scheduler(1000)
        ctx = s._req_status["0"].req_context
        s._build_store_jobs(output(s))
        s._build_store_jobs(output(s))
        get_keys = [(8, 0, k) for k in range(128)]
        for key in get_keys:
            self.assertTrue(m._initiate_promotion(0, key, ctx))
        m._flush_pending_promotions()
        self.assertEqual(m.get_put_admission_budget(), 0)
        job = t.jobs.pop()
        t.done.append(
            Attrs(job_id=job.job_id, success=False, successful_keys=get_keys[:64])
        )
        m._process_finished_jobs()
        self.assertTrue(
            all(m.primary_tier.lookup(k, ctx) is Result.HIT for k in get_keys[:64])
        )
        self.assertTrue(
            all(m.primary_tier.lookup(k, ctx) is Result.MISS for k in get_keys[64:])
        )
        self.assertEqual(len(m._put_staging_keys), 512)

    def test_already_present_ranges_resolve_only_the_bounded_frontier(self):
        s, m, t = make_scheduler(1000)
        ctx = s._req_status["0"].req_context
        keys = s._req_status["0"].group_states[0].offload_keys[:256]
        m.primary_tier.prepare_store(keys, ctx)
        m.primary_tier.complete_store(keys, ctx)
        self.assertEqual(s._build_store_jobs(output(s)), {})
        self.assertEqual(s._req_status["0"].group_states[0].next_stored_chunk_idx, 256)
        jobs = s._build_store_jobs(output(s))
        self.assertEqual(len(s._jobs[next(iter(jobs))].keys), 256)
        self.assertEqual(s._req_status["0"].group_states[0].next_stored_chunk_idx, 512)

    def test_swa_finish_reconsiders_tail_before_existing_cursor(self):
        s, m, t = make_scheduler(14)

        def reachable(**kw):
            horizon = kw["final_segment_end_block"]
            return [
                (i + 1) % 4 == 0 or i + 1 == horizon
                for i in range(kw["start_block"], kw["end_block"])
            ]

        group = s.config.kv_group_configs[0]._replace(
            sliding_window_size_in_chunks=1,
            kv_cache_spec=SlidingWindowSpec(),
            manager_cls=Attrs(reachable_block_mask=reachable),
        )
        s.config = s.config._replace(kv_group_configs=(group,), alignment_tokens=4)
        st = s._req_status["0"]
        st.config = s.config
        st.req.num_prompt_tokens = 10
        st.req.num_tokens = 10
        jobs = s._build_store_jobs(output(s))
        drain(s, m, t, jobs)
        self.assertEqual(st.group_states[0].next_stored_chunk_idx, 10)
        st.req.num_tokens = 14
        jobs = s._build_store_jobs(output(s))
        drain(s, m, t, jobs)
        self.assertEqual(st.group_states[0].next_stored_chunk_idx, 14)
        st.req.status = RequestStatus.FINISHED_STOPPED
        st.req.num_tokens = 15
        jobs = s._build_store_jobs(output(s))
        keys = s._jobs[next(iter(jobs))].keys
        self.assertEqual(keys, {(0, 0, 13)})
        self.assertEqual(st.group_states[0].next_stored_chunk_idx, 14)

    def test_parked_request_level_cascade_keeps_credit_across_promotion_completion(
        self,
    ):
        s, m, t = make_scheduler(1000)
        ctx = s._req_status["0"].req_context
        m._req_state["0"].request_level_tiers = {0}
        keys = [(8, 0, k) for k in range(128)]
        for key in keys:
            self.assertTrue(m._initiate_promotion(0, key, ctx))
        m._flush_pending_promotions()
        result = m.prepare_store(keys, ctx)
        self.assertEqual(result.keys_to_store, [])
        self.assertEqual(m.get_put_admission_budget(), 384)
        job = t.jobs.pop()
        t.done.append(Attrs(job_id=job.job_id, success=True, successful_keys=None))
        m._process_finished_jobs()
        self.assertEqual(m.get_put_admission_budget(), 384)
        m._flush_pending_cascades()
        self.assertEqual(m.get_put_admission_budget(), 384)
        job = t.jobs.pop()
        t.done.append(Attrs(job_id=job.job_id, success=True))
        m._process_finished_jobs()
        self.assertEqual(m.get_put_admission_budget(), 512)

    def test_rejected_frontier_does_not_count_unavailable_twice(self):
        s, m, t = make_scheduler(1000)
        ctx = s._req_status["0"].req_context
        st = s._req_status["0"].group_states[0]
        st.block_ids[0] = 0
        keys = [(9, 0, k) for k in range(873)]
        m.primary_tier.prepare_store(keys, ctx)
        m.primary_tier.complete_store(keys, ctx)
        original = m.primary_tier._policy.evict
        m.primary_tier._policy.evict = lambda *a: None
        self.assertEqual(s._build_store_jobs(output(s)), {})
        self.assertEqual(
            s._connector_stats.reduce().get(
                "vllm:guard_put_source_unavailable_before_admission", 0
            ),
            0,
        )
        m.primary_tier._policy.evict = original
        for _ in range(16):
            if s._build_store_jobs(output(s)):
                break
        self.assertEqual(
            s._connector_stats.reduce()[
                "vllm:guard_put_source_unavailable_before_admission"
            ],
            1,
        )

    def test_finished_unavailable_chunk_beyond_admitted_frontier_is_counted(self):
        s, m, t = make_scheduler(1000)
        st = s._req_status["0"]
        st.req.status = RequestStatus.FINISHED_STOPPED
        st.group_states[0].block_ids[-1] = 0
        out = output(s)
        out.finished_req_ids = {"0"}
        out.num_scheduled_tokens = {}
        s.build_connector_meta(out)
        self.assertEqual(
            s._connector_stats.reduce()["vllm:guard_put_abandoned_chunks"], 744
        )

    def test_unavailable_accounting_uses_each_group_resolved_frontier(self):
        s, m, t = make_scheduler(1000, (1, 1))
        states = s._req_status["0"].group_states
        states[0].next_stored_chunk_idx = 1000
        states[1].block_ids[900] = 0
        jobs = s._build_store_jobs(output(s))
        self.assertEqual(states[1].next_stored_chunk_idx, 256)
        self.assertEqual(
            s._connector_stats.reduce().get(
                "vllm:guard_put_source_unavailable_before_admission", 0
            ),
            0,
        )
        drain(s, m, t, jobs)
        for _ in range(3):
            jobs = s._build_store_jobs(output(s))
            drain(s, m, t, jobs)
        self.assertEqual(
            s._connector_stats.reduce()[
                "vllm:guard_put_source_unavailable_before_admission"
            ],
            1,
        )

    def test_reset_clears_retry_and_deferred_bookkeeping(self):
        s, m, t = make_scheduler(1000)
        s._put_deferred["0"] = 1000
        s._put_retry_gate["0"] = ((0,), 3)
        s._put_retry_after = "0"
        s.reset_cache()
        self.assertEqual(s._put_deferred, {})
        self.assertEqual(s._put_retry_gate, {})
        self.assertIsNone(s._put_retry_after)
        self.assertEqual(m.get_put_admission_budget(), 512)

    def test_constructor_activates_only_with_batch_config(self):
        s, m, t = make_scheduler()
        cls = G["SchedulerOffloadConfig"]
        original = cls.from_spec
        G["OffloadingEventsTracker"] = lambda config: Quiet()
        original_resolve = G["resolve_mamba_align_size"]
        G["resolve_mamba_align_size"] = lambda *a: None
        cls.from_spec = classmethod(lambda *args: s.config)
        try:
            for extra, enabled in [
                ({}, False),
                (
                    {
                        "guard_put_batch_chunks": 256,
                        "guard_put_max_inflight_chunks": 512,
                        "guard_put_get_reserve_chunks": 128,
                    },
                    True,
                ),
            ]:
                fresh, tier = manager()
                spec = Attrs(
                    extra_config=extra,
                    get_manager=lambda manager=fresh: manager,
                    kv_events_config=None,
                )
                config = Attrs(cache_config=Attrs(enable_prefix_caching=True))
                actual = Scheduler(spec, config, None)
                self.assertEqual(hasattr(fresh, "_put_max_inflight"), enabled)
                self.assertEqual(actual._put_batch_chunks, 256 if enabled else 0)
        finally:
            cls.from_spec = original
            G["resolve_mamba_align_size"] = original_resolve

    def test_real_stats_prometheus_registry_boundary(self):
        from functools import partial

        from prometheus_client import (
            CollectorRegistry,
            Counter,
            Gauge,
            Histogram,
            generate_latest,
        )

        self.assertTrue(
            "get_guard_put_metric_definitions" in G, "missing metric metadata hook"
        )
        hook = G["get_guard_put_metric_definitions"]
        self.assertEqual(hook({}), {})
        definitions = hook({"guard_put_batch_chunks": 256})
        s, m, t = make_scheduler(1000)
        s._build_store_jobs(output(s))
        stats = s.get_stats()
        payload = Stats()
        for name, values in stats._values.items():
            if name.startswith("vllm:guard_put_"):
                payload._types[name] = stats._types[name]
                payload._values[name] = values
        self.assertTrue(set(payload._values) <= set(definitions))
        cls = G["OffloadPromMetrics"]
        recorder = cls.__new__(cls)
        recorder._offloading_metric_metadata = definitions
        registry = CollectorRegistry()
        recorder._counter_cls = partial(Counter, registry=registry)
        recorder._gauge_cls = partial(Gauge, registry=registry)
        recorder._histogram_cls = partial(Histogram, registry=registry)
        recorder._labelnames = ["model_name"]
        recorder.per_engine_labelvalues = {0: ["fixture"]}
        recorder._observe_deprecated_metrics = False
        recorder.offloading_metrics = {}
        recorder._offloading_metric_defs = {
            key: recorder._create_metric(key, meta) for key, meta in definitions.items()
        }
        recorder.observe(payload.data)
        exported = generate_latest(registry).decode()
        self.assertIn(
            'vllm:guard_put_offered_chunks_total{model_name="fixture"} 256.0', exported
        )
        self.assertIn(
            'vllm:guard_put_admitted_chunks_total{model_name="fixture"} 256.0', exported
        )
        self.assertIn(
            'vllm:guard_put_deferred_chunks_total{model_name="fixture"} 744.0', exported
        )
        self.assertIn(
            'vllm:guard_put_staging_chunks{model_name="fixture"} 256.0', exported
        )
        self.assertNotIn("vllm:vllm:", exported)
        recorder._offloading_metric_defs = {}
        with self.assertRaises(AssertionError):
            recorder.observe(payload.data)

    def test_every_emitted_guard_metric_has_real_metadata(self):
        definitions = G["get_guard_put_metric_definitions"](
            {"guard_put_batch_chunks": 256}
        )
        emitted = set()
        for path in (SCHED, MANAGER):
            tree = ast.parse(path.read_text())
            emitted.update(
                n.value
                for n in ast.walk(tree)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and n.value.startswith("vllm:guard_put_")
                and n.value != "vllm:guard_put_"
            )
        self.assertEqual(emitted, set(definitions))
        for name, meta in definitions.items():
            self.assertIsInstance(meta, G["OffloadingMetricMetadata"])
            self.assertTrue(name.startswith("vllm:guard_put_"))
            self.assertFalse(name.startswith("vllm:vllm:"))

    def test_manager_telemetry_covers_non_normal_store_paths(self):
        s, m, t = make_scheduler(1000)
        ctx = s._req_status["0"].req_context
        keys = [(7, 0, k) for k in range(3)]
        m.prepare_store(keys, ctx)
        metrics = m.get_stats().reduce()
        self.assertEqual(metrics.get("vllm:guard_put_offered_chunks"), 3)
        self.assertEqual(metrics.get("vllm:guard_put_admitted_chunks"), 3)
        self.assertEqual(metrics.get("vllm:guard_put_resolved_chunks"), 3)
        self.assertEqual(metrics.get("vllm:guard_put_batch_chunks_sum"), 3)

    def test_duplicate_zero_source_cannot_replace_eligible_source(self):
        s, m, t = make_scheduler(2)
        st = s._req_status["0"].group_states[0]
        st.offload_keys[1] = st.offload_keys[0]
        st.block_ids[0] = 0
        jobs = s._build_store_jobs(output(s))
        job = next(iter(jobs.values()))
        self.assertEqual(job.src_spec.block_ids, [2])
        self.assertEqual(job.src_spec.group_sizes, [1])
        self.assertEqual(job.src_spec.block_indices, [1])
        self.assertEqual(len(job.dst_spec.chunk_ids), 1)
        status = s._jobs[next(iter(jobs))]
        self.assertEqual(status.deferred_fence_block_ids, [2])
        self.assertEqual(st.next_stored_chunk_idx, 2)

    def test_duplicate_reallocated_source_cannot_replace_eligible_finished_source(self):
        s, m, t = make_scheduler(2)
        state = s._req_status["0"]
        st = state.group_states[0]
        st.offload_keys[1] = st.offload_keys[0]
        state.req.status = RequestStatus.FINISHED_STOPPED
        s._current_batch_allocated_block_ids = {1}
        jobs = s._build_store_jobs(output(s))
        job = next(iter(jobs.values()))
        self.assertEqual(job.src_spec.block_ids, [2])
        self.assertEqual(job.src_spec.group_sizes, [1])
        self.assertEqual(job.src_spec.block_indices, [1])
        self.assertEqual(len(job.dst_spec.chunk_ids), 1)
        self.assertEqual(set(s._block_id_to_pending_jobs), {2})
        self.assertEqual(s._current_batch_jobs_to_flush, set())

    def test_duplicate_unreachable_source_cannot_replace_eligible_source(self):
        s, m, t = make_scheduler(2)
        state = s._req_status["0"]
        st = state.group_states[0]
        st.offload_keys[1] = st.offload_keys[0]
        group = s.config.kv_group_configs[0]._replace(
            manager_cls=Attrs(reachable_block_mask=lambda **kw: [False, True])
        )
        s.config = s.config._replace(kv_group_configs=(group,))
        state.config = s.config
        jobs = s._build_store_jobs(output(s))
        job = next(iter(jobs.values()))
        self.assertEqual(job.src_spec.block_ids, [2])
        self.assertEqual(job.src_spec.block_indices, [1])
        self.assertEqual(len(job.dst_spec.chunk_ids), 1)

    def test_clipped_source_mapping_is_bounded_and_keeps_group_chunk_coordinates(self):
        s, m, t = make_scheduler(600, (1, 2))
        st = s._req_status["0"].group_states[0]
        st.offload_keys[1] = st.offload_keys[0]
        st.block_ids[0] = 0
        original = s._clip_put_frontier
        captured = []

        def capture(*args):
            result = original(*args)
            captured.append(result)
            return result

        s._clip_put_frontier = capture
        jobs = s._build_store_jobs(output(s))
        ranges, sources = captured[0]
        self.assertIsInstance(sources, dict)
        self.assertLessEqual(len(sources), 256)
        self.assertEqual(sources[st.offload_keys[0]], (0, 1))
        for key, (group_idx, chunk_idx) in sources.items():
            start, end = ranges[group_idx]
            self.assertTrue(start <= chunk_idx < end)
            state = s._req_status["0"].group_states[group_idx]
            self.assertEqual(state.offload_keys[chunk_idx], key)
            self.assertNotEqual(state.block_ids[chunk_idx], 0)
        job = next(iter(jobs.values()))
        self.assertEqual(len(job.src_spec.block_ids), len(job.dst_spec.chunk_ids))
        self.assertEqual(job.src_spec.block_ids[0], 2)
        self.assertEqual(job.src_spec.block_indices[0], 1)
        drain(s, m, t, jobs)
        self.assertTrue(s._build_store_jobs(output(s)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
