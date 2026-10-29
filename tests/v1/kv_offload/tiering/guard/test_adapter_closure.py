# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import copy
import dataclasses
import enum
import hashlib
import unittest
import uuid
from types import SimpleNamespace as A

import test_put_batching as f
from test_hint_closure import ADAPTER, chain_and_hint


class Status(enum.Enum):
    HIT = 1
    FETCHABLE = 2
    FETCHING = 3
    MISS = 4


class Sink:
    def __init__(self):
        self.records = []

    def journal(self, r):
        self.records.append(r)
        return True

    def emit(self, r):
        return True


class Context(f.Context):
    def __init__(self, req_id, raw):
        super().__init__(req_id)
        self.kv_hints = raw
        self.positions = {}
        self.load_tier_filter = A(allows=lambda *args: True)

    def set_offload_key_position(self, k, p):
        self.positions[k] = p

    def get_offload_key_position(self, k):
        return self.positions.get(k)


class Native:
    def __init__(self):
        self.hints = {}
        self.local = set()
        self.queries = []
        self.dispatch = []
        self.finished = []
        self.seq = 0
        self.missing = set()
        self._core = A(_compact_capture=Sink(), nixl_agent_name="worker")

    def submit_hint(self, request_id, hints):
        self.hints[request_id] = copy.deepcopy(hints)

    def discard_hint(self, request_id):
        self.hints.pop(request_id, None)

    def query(self, keys, request_id):
        self.queries.append((tuple(keys), request_id))
        allowed = (
            self.hints.get(request_id, {})
            .get("actions", [{}])[0]
            .get("payload", {})
            .get("block_hashes", [])
        )
        return [
            (
                Status.HIT
                if k in self.local
                else Status.FETCHABLE
                if k not in self.missing and int.from_bytes(k[24:32], "big") in allowed
                else Status.MISS,
                None,
            )
            for k in keys
        ]

    def deliver(self, blocks, request_id):
        self.seq += 1
        self.dispatch.append((self.seq, dict(blocks), request_id))
        return self.seq

    def poll_completed(self):
        done, self.finished = self.finished, []
        for _, entries in done:
            self.missing.update(k for k, e in entries.items() if not e.success)
        return done


def adapter():
    tree = ast.parse(ADAPTER.read_text())
    names = {
        "_HintClosureState",
        "_strict_hint_prefix_closure",
        "_hint_key_position",
        "KVCRSecondaryTierManager",
    }
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)
    ] + [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    g = dict(
        copy=copy,
        uuid=uuid,
        override=lambda fn: fn,
        SecondaryTierManager=object,
        LookupResult=f.Result,
        QueryStatus=Status,
        OffloadKey=bytes,
        JobResult=f.Attrs,
        RequestOffloadingContext=f.Attrs,
        msgspec=A(to_builtins=copy.deepcopy),
        _get_request_hash_chain=lambda ctx: ctx.get_state(f.G["RequestHashChain"]),
    )
    exec(compile(ast.fix_missing_locations(tree), str(ADAPTER), "exec"), g)
    obj = g["KVCRSecondaryTierManager"].__new__(g["KVCRSecondaryTierManager"])
    obj._kvcr = Native()
    obj._key_adapter = A(encode=bytes, decode=lambda k: int.from_bytes(k[24:32], "big"))
    obj._jobs_by_op = {}
    obj._load_job_parts = {}
    obj._finished_jobs = []
    obj._make_descriptor = lambda i: i
    obj._framework_pin_adapter = A(take_pin_job_results=lambda: [])
    obj.medium = "cpu"
    obj.locality = "remote"
    obj.cache_hit_source = "kvcr"
    obj.tier_type = "kvcr"
    obj.bp_detector = None
    return obj, g


def context():
    chain, raw = chain_and_hint()
    ctx = Context("req", raw)
    ctx.set_state(
        f.G["RequestHashChain"](
            "req", chain.block_hashes, 4, chain.group_chunks, 256, 1
        )
    )
    return ctx, chain


class AdapterTests(unittest.TestCase):
    def test_closure_and_load_work_without_capture_sink(self):
        a, g = adapter()
        a._kvcr._core = A()
        ctx, chain = context()
        a.on_new_request(ctx)
        key = chain.block_hashes[15] + (0).to_bytes(4, "big")
        self.assertIs(a.lookup(key, ctx), f.Result.HIT)
        a.submit_load(f.Attrs(keys=[key], chunk_ids=[0], req_context=ctx, job_id=7))
        op, keys, request_id = a._kvcr.dispatch[0]
        self.assertEqual(request_id, "req")
        a._kvcr.finished = [(op, {key: f.Attrs(success=True)})]
        self.assertTrue(a.get_finished_jobs()[0].success)

    def test_missing_trusted_chain_never_authorizes_remote_lookup(self):
        a, g = adapter()
        ctx, chain = context()
        ctx.states.clear()
        a.on_new_request(ctx)
        key = chain.block_hashes[63] + (0).to_bytes(4, "big")
        self.assertIs(a.lookup(key, ctx), f.Result.MISS)
        self.assertIsNone(a._kvcr.queries[-1][1])

    def test_appended_anchor_collision_revokes_remote_membership(self):
        a, g = adapter()
        ctx, chain = context()
        a.on_new_request(ctx)
        key = chain.block_hashes[63] + (0).to_bytes(4, "big")
        self.assertIs(a.lookup(key, ctx), f.Result.HIT)
        previous = ctx.get_state(f.G["RequestHashChain"])
        ctx.set_state(
            dataclasses.replace(
                previous,
                block_hashes=previous.block_hashes
                + (b"x" * 24 + chain.block_hashes[63][-8:],),
            )
        )
        self.assertIs(a.lookup(key, ctx), f.Result.MISS)
        self.assertNotIn("req", a._kvcr.hints)

    def test_local_hits_beyond_remote_boundary_and_invalid_hint_remain_hits(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                a, g = adapter()
                ctx, chain = context()
                if invalid:
                    ctx.kv_hints["actions"][0]["payload"]["block_hashes"].append(True)
                a.on_new_request(ctx)
                key = chain.block_hashes[128] + (4).to_bytes(4, "big")
                a._kvcr.local.add(key)
                self.assertIs(a.lookup(key, ctx), f.Result.HIT)
                self.assertIsNone(a._kvcr.queries[-1][1])
                a._kvcr.local.clear()
                self.assertIs(a.lookup(key, ctx), f.Result.MISS)
                self.assertIsNone(a._kvcr.queries[-1][1])

    def test_invalid_metadata_does_not_crash_and_clears_hint(self):
        a, g = adapter()
        ctx, chain = context()
        ctx.kv_hints["actions"][0]["payload"]["block_hashes"].append(True)
        a.on_new_request(ctx)
        key = chain.block_hashes[15] + (0).to_bytes(4, "big")
        self.assertIs(a.lookup(key, ctx), f.Result.MISS)
        self.assertNotIn("req", a._kvcr.hints)

    def test_stable_chain_does_not_reemit_or_resubmit(self):
        a, g = adapter()
        ctx, chain = context()
        a.on_new_request(ctx)
        key = chain.block_hashes[15] + (0).to_bytes(4, "big")
        self.assertIs(a.lookup(key, ctx), f.Result.HIT)
        proof_count = len(a._kvcr._core._compact_capture.records)
        native_hint = a._kvcr.hints["req"]
        for _ in range(50):
            self.assertIs(a.lookup(key, ctx), f.Result.HIT)
        self.assertEqual(len(a._kvcr._core._compact_capture.records), proof_count)
        self.assertIs(a._kvcr.hints["req"], native_hint)

    def test_mixed_loads_partition_and_wait_for_both_without_remote_fallback(self):
        a, g = adapter()
        ctx, chain = context()
        a.on_new_request(ctx)
        remote = chain.block_hashes[15] + (0).to_bytes(4, "big")
        local = chain.block_hashes[128] + (4).to_bytes(4, "big")
        a._kvcr.local.add(local)
        self.assertIs(a.lookup(local, ctx), f.Result.HIT)
        a._kvcr.local.clear()  # eviction between lookup and dispatch
        a.submit_load(
            f.Attrs(keys=[remote, local], chunk_ids=[0, 1], req_context=ctx, job_id=99)
        )
        self.assertEqual(len(a._kvcr.dispatch), 2)
        by_scope = {scope: (op, keys) for op, keys, scope in a._kvcr.dispatch}
        self.assertEqual(set(by_scope[None][1]), {local})
        self.assertEqual(set(by_scope["req"][1]), {remote})
        rop = by_scope["req"][0]
        lop = by_scope[None][0]
        a._kvcr.finished = [(rop, {remote: f.Attrs(success=True)})]
        self.assertEqual(a.get_finished_jobs(), [])
        a._kvcr.finished = [(lop, {local: f.Attrs(success=False)})]
        jobs = a.get_finished_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertFalse(jobs[0].success)
        self.assertEqual(jobs[0].successful_keys, {remote})
        for record in a._kvcr._core._compact_capture.records:
            if record["event"] == "hint_prefix_selected":
                self.assertIn(record["target_op"], (rop, lop))

    def test_full_key_boundary_not_effective_hash_alone(self):
        a, g = adapter()
        ctx, chain = context()
        a.on_new_request(ctx)
        for group, index in ((99, 63), (0, 0)):
            key = chain.block_hashes[index] + group.to_bytes(4, "big")
            self.assertIs(a.lookup(key, ctx), f.Result.MISS)
            self.assertIsNone(a._kvcr.queries[-1][1])


class SchedulerTests(unittest.TestCase):
    def test_five_groups_pending_then_positive_and_real_cpu_gpu_job(self):
        original = {
            name: f.G.get(name)
            for name in ("make_offload_key", "get_offload_group_idx")
        }
        self.addCleanup(f.G.update, original)
        s, m, _ = f.make_scheduler(n=3073, group_chunks=(64, 64, 256, 8, 4))
        groups = []
        for i, t in enumerate((64, 64, 256, 8, 4)):
            window = {0: 2, 3: 16, 4: 2}.get(i)
            spec = (
                f.SlidingWindowSpec({0: 128, 3: 128, 4: 8}[i])
                if window
                else f.G["FullAttentionSpec"]()
            )
            groups.append(
                f.G["GroupOffloadConfig"](
                    i,
                    t,
                    t,
                    t // 4,
                    None,
                    window,
                    spec,
                    f.Attrs(reachable_block_mask=lambda **kw: None),
                    False,
                    i == 1,
                )
            )
        s.config = s.config._replace(
            kv_group_configs=tuple(groups),
            tokens_per_hash=4,
            alignment_tokens=256,
            num_workers=1,
        )
        s._lookup_groups = (1, 2, 0, 3, 4)
        s._sliding_window_groups = (0, 3, 4)
        s._mamba_align_size = None
        req = f.Attrs(
            request_id="req",
            num_tokens=3073,
            num_prompt_tokens=3073,
            num_computed_tokens=0,
            block_hashes=[hashlib.sha256(str(i).encode()).digest() for i in range(768)],
            kv_transfer_params=None,
        )
        raw = {
            "protocol_version": "0.1",
            "actions": [
                {
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "source_control_endpoint": "tcp://source:1",
                        "block_hashes": [
                            int.from_bytes(req.block_hashes[i][-8:], "big")
                            for i in range(63, 768, 64)
                        ],
                    },
                }
            ],
        }
        ctx = Context("req", raw)
        a, g = adapter()
        a.on_new_request(ctx)
        m.secondary_tiers = [a]
        m._tier_index = {a: 0}
        m._req_state = {"req": f.G["RequestState"](ctx)}
        state = f.G["RequestOffloadState"](s.config, req, ctx, f.Attrs(policy=0))
        s._req_status = {"req": state}
        s._jobs = {}
        m._processed_jobs_this_step = False
        f.G["make_offload_key"] = lambda h, i: h + i.to_bytes(4, "big")
        f.G["get_offload_group_idx"] = lambda k: int.from_bytes(k[-4:], "big")
        state.update_offload_keys()
        first = s._lookup_complete_chunks(state)
        self.assertIsNone(first)
        wave_results = [first]
        positions = {h: (i + 1) * 4 for i, h in enumerate(req.block_hashes)}

        def ready(key):
            group = int.from_bytes(key[-4:], "big")
            end = positions[key[:-4]]
            return group in (1, 2) or (
                end % 256 == 0 or end % 256 > {0: 128, 3: 128, 4: 248}[group]
            )

        completed = 0
        for _ in range(10):
            m._flush_pending_promotions()
            self.assertTrue(a._kvcr.dispatch)
            for op, keys, _ in a._kvcr.dispatch[completed:]:
                a._kvcr.finished.append(
                    (op, {k: f.Attrs(success=ready(k)) for k in keys})
                )
            completed = len(a._kvcr.dispatch)
            m._process_finished_jobs()
            m._processed_jobs_this_step = False
            matched = s._lookup_complete_chunks(state)
            wave_results.append(matched)
            if matched is not None:
                break
        self.assertGreater(matched, 0)
        self.assertEqual(matched, 2816)
        self.assertGreaterEqual(len(wave_results), 3)
        # SWA groups retain only the suffix blocks required at the hit boundary.
        block_groups = []
        for config in groups:
            count = (matched + config.tokens_per_block - 1) // config.tokens_per_block
            keep = config.sliding_window_size_in_chunks or count
            block_groups.append(
                [
                    f.Attrs(
                        block_id=10000 * config.group_idx + i + 1,
                        is_null=i < count - keep,
                        block_hash=None,
                    )
                    for i in range(count)
                ]
            )
        s.update_state_after_alloc(req, f.Attrs(blocks=block_groups), matched)
        self.assertEqual(len(s._current_batch_load_jobs), 1)
        job = next(iter(s._current_batch_load_jobs.values()))
        self.assertTrue(job.src_spec.chunk_ids)
        self.assertTrue(job.dst_spec.block_ids)
        self.assertTrue(all(n > 0 for n in job.dst_spec.group_sizes))
        selected = {
            int.from_bytes(k[-4:], "big")
            for r in a._kvcr._core._compact_capture.records
            if r["event"] == "hint_prefix_selected" and r["remote_allowed"]
            for k in r["keys"]
        }
        self.assertEqual(selected, set(range(5)))


if __name__ == "__main__":
    unittest.main()
