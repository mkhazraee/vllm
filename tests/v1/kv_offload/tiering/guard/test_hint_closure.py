# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import copy
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
ADAPTER = ROOT / "vllm/v1/kv_offload/tiering/kvcr/manager.py"


def helper():
    tree = ast.parse(ADAPTER.read_text())
    found = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_strict_hint_prefix_closure", "_hint_key_position")
    ]
    if not found:
        raise AssertionError("strict closure helper absent")
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)
    ] + found
    ns = {}
    exec(compile(ast.fix_missing_locations(tree), str(ADAPTER), "exec"), ns)
    fn = ns["_strict_hint_prefix_closure"]
    fn.key_position = ns["_hint_key_position"]
    return fn


def chain_and_hint():
    hashes = tuple(hashlib.sha256(str(i).encode()).digest() for i in range(192))
    chain = SimpleNamespace(
        request_id="req",
        block_hashes=hashes,
        tokens_per_hash=4,
        group_chunks=((0, 64), (1, 64), (2, 256), (3, 8), (4, 4)),
        anchor_tokens=256,
    )
    anchors = [int.from_bytes(hashes[i], "big") & ((1 << 64) - 1) for i in (63, 127)]
    raw = {
        "protocol_version": "0.1",
        "actions": [
            {
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": "tcp://source:1",
                    "block_hashes": anchors,
                },
            }
        ],
    }
    return chain, raw


class ClosureTests(unittest.TestCase):
    def test_fine_prefix_is_admitted_only_inside_anchor_and_configured_groups(self):
        chain, raw = chain_and_hint()
        before = copy.deepcopy(raw)
        p = helper()(chain, raw)
        self.assertEqual(raw, before)
        self.assertEqual(
            p["raw_hint_hashes"], raw["actions"][0]["payload"]["block_hashes"]
        )
        self.assertEqual(p["anchor_positions"], [256, 512])
        self.assertEqual(p["covered_through_tokens"], 512)
        self.assertEqual(p["source_endpoint"], "tcp://source:1")
        for group, span in chain.group_chunks:
            key = chain.block_hashes[span // 4 - 1] + group.to_bytes(4, "big")
            self.assertIsNotNone(helper().key_position(p, key))
        self.assertIsNone(
            helper().key_position(p, chain.block_hashes[128] + (4).to_bytes(4, "big"))
        )
        self.assertIsNone(
            helper().key_position(p, chain.block_hashes[0] + (0).to_bytes(4, "big"))
        )
        self.assertIsNone(
            helper().key_position(p, chain.block_hashes[63] + (99).to_bytes(4, "big"))
        )
        self.assertIsNone(helper().key_position(p, b"x" * 32 + (0).to_bytes(4, "big")))

    def test_bad_anchors_fail_closed(self):
        for kind in ["duplicate", "foreign", "reversed", "bool", "unaligned"]:
            with self.subTest(kind=kind):
                chain, raw = chain_and_hint()
                a = raw["actions"][0]["payload"]["block_hashes"]
                if kind == "duplicate":
                    a.append(a[0])
                elif kind == "foreign":
                    a.append(123)
                elif kind == "reversed":
                    a.reverse()
                elif kind == "bool":
                    a[0] = True
                else:
                    a[0] = int.from_bytes(chain.block_hashes[0], "big") & (
                        (1 << 64) - 1
                    )
                with self.assertRaises(ValueError):
                    helper()(chain, raw)

    def test_ambiguous_projection_fails_closed(self):
        chain, raw = chain_and_hint()
        hashes = list(chain.block_hashes)
        hashes[0] = b"x" * 24 + hashes[63][-8:]
        chain.block_hashes = tuple(hashes)
        with self.assertRaises(ValueError):
            helper()(chain, raw)

    def test_bad_chain_group_or_endpoint_fails_closed(self):
        for kind in [
            "short_hash",
            "bad_step",
            "bad_group",
            "duplicate_group",
            "no_source",
            "no_anchor_step",
        ]:
            with self.subTest(kind=kind):
                chain, raw = chain_and_hint()
                if kind == "short_hash":
                    chain.block_hashes = (b"x",) + chain.block_hashes[1:]
                elif kind == "bad_step":
                    chain.tokens_per_hash = 0
                elif kind == "bad_group":
                    chain.group_chunks = ((True, 64),)
                elif kind == "duplicate_group":
                    chain.group_chunks = ((0, 64), (0, 64))
                elif kind == "no_source":
                    raw["actions"][0]["payload"]["source_control_endpoint"] = ""
                else:
                    chain.anchor_tokens = None
                with self.assertRaises(ValueError):
                    helper()(chain, raw)


if __name__ == "__main__":
    unittest.main()
