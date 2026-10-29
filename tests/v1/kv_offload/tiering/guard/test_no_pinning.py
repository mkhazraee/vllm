# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the adapter constructor without loading GPU or native transports."""

import ast
import ctypes
import mmap
import socket
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace as A

ADAPTER = (
    Path(__file__).resolve().parents[5] / "vllm/v1/kv_offload/tiering/kvcr/manager.py"
)


class Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Base:
    def __init__(self, *args):
        self.block_size_bytes = 8


class Native:
    def __init__(self, config, bindings, backends, guard):
        self.config, self.bindings, self.backends = config, bindings, backends


def construct(**kwargs):
    tree = ast.parse(ADAPTER.read_text())
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)
    ] + [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name
        in ("_FrameworkPinAdapter", "_VllmKeyAdapter", "KVCRSecondaryTierManager")
    ]
    g = dict(
        SecondaryTierManager=Base,
        override=lambda f: f,
        ctypes=ctypes,
        mmap=mmap,
        socket=socket,
        uuid=uuid,
        _resolve_policy=lambda p: None,
        _REQUIRED_ROUTER_CAPABILITIES=set(),
        ZmqPeerControlChannel=lambda *args: A(close=lambda: None),
        KVCR=Native,
        KVCRConfig=Options,
        KVCRBindings=Options,
        KVCRBackendConfigs=Options,
        RemoteFWDramOptions=Options,
        RegionDescriptor=Options,
        OffloadingConnectorStats=Options,
    )
    exec(compile(ast.fix_missing_locations(tree), str(ADAPTER), "exec"), g)
    spec = A(
        config=A(parallel=A(data_parallel_rank_local=0)),
        kv_events_config=A(
            enable_kv_cache_events=False, self_describing_kv_events=False
        ),
    )
    return g["KVCRSecondaryTierManager"](
        spec,
        memoryview(bytearray(32)).cast("B", (4, 8)),
        "kvcr",
        control_ports=[12345],
        **kwargs,
    )


class NoPinningTests(unittest.TestCase):
    def test_native_option_defaults_to_false(self):
        tier = construct()
        self.assertIs(tier.no_pinning_mode, False)
        self.assertIs(tier._kvcr.backends.remote_fw_dram.no_pinning_mode, False)

    def test_option_reaches_native_config_without_capture_or_telemetry(self):
        tier = construct(no_pinning_mode=True)
        self.assertIs(tier.no_pinning_mode, True)
        self.assertIs(tier._kvcr.backends.remote_fw_dram.no_pinning_mode, True)
        self.assertIsNone(tier._kvcr.bindings.stats_factory)
        self.assertEqual(tier._kvcr.backends.framework_regions[0].count, 4)
        self.assertEqual(tier._kvcr.backends.framework_regions[0].size, 8)


if __name__ == "__main__":
    unittest.main()
