"""The rpc channel between the Engine and the rank-zero scheduler.

The scheduler binds an ipc endpoint and the Engine connects. With no Engine, the
scheduler of an ipc endpoint makes no connect attempt. The Engine still binds a
tcp endpoint.
"""

import os
import shutil
import socket
import tempfile
import time
import unittest
from types import SimpleNamespace

import zmq
from zmq.utils.monitor import recv_monitor_message

from sglang.srt.managers.scheduler_components.ipc_channels import (
    SchedulerIpcChannels,
    engine_rpc_socket,
    rpc_scheduler_binds,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CONNECT_EVENTS = zmq.EVENT_CONNECT_DELAYED | zmq.EVENT_CONNECT_RETRIED


def _port_args(rpc_ipc_name: str, d: str) -> SimpleNamespace:
    def name(n: str) -> str:
        return f"ipc://{os.path.join(d, n)}"

    return SimpleNamespace(
        scheduler_input_ipc_name=name("in"),
        rpc_ipc_name=rpc_ipc_name,
        tokenizer_ipc_name=name("tok"),
        detokenizer_ipc_name=name("detok"),
        metrics_ipc_name=name("metrics"),
    )


def _free_tcp_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connect_events(sock: zmq.Socket, seconds: float) -> list[int]:
    mon = sock.get_monitor_socket(CONNECT_EVENTS)
    events = []
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if mon.poll(50):
                events.append(recv_monitor_message(mon)["event"])
    finally:
        sock.disable_monitor()
        mon.close(0)
    return events


class TestSchedulerRpcEndpoint(CustomTestCase):
    def _temp_file_path(self) -> str:
        """An existing temporary file, as PortArgs.init_new uses for an ipc endpoint."""
        with tempfile.NamedTemporaryFile(dir=self.dir, delete=False) as handle:
            return handle.name

    def _channels(
        self, rpc_ipc_name: str
    ) -> tuple[SimpleNamespace, SchedulerIpcChannels]:
        args = _port_args(rpc_ipc_name, self.dir)
        ch = SchedulerIpcChannels.create(
            port_args=args,
            is_rank_zero=True,
            skip_tokenizer_init=False,
            metrics_enabled=False,
            enable_scripted_runtime=False,
        )
        self.addCleanup(ch.recv_from_rpc.close, 0)
        self.addCleanup(ch.recv_from_tokenizer.close, 0)
        # The channels own the context of their sockets. Destroy it with linger 0,
        # so that a failed test ends cleanly and does not block at exit.
        self.addCleanup(ch.recv_from_rpc.context.destroy, 0)
        return args, ch

    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.ctx = zmq.Context()
        # destroy(linger=0) also closes a socket that a failed test left open, so a
        # wrong endpoint fails the test and does not block in term().
        self.addCleanup(self.ctx.destroy, 0)

    def _round_trip(self, args: SimpleNamespace, ch: SchedulerIpcChannels) -> None:
        engine = engine_rpc_socket(self.ctx, args)
        self.addCleanup(engine.close, 0)
        engine.setsockopt(
            zmq.SNDTIMEO, 5000
        )  # fail after 5 s, not block, without a peer
        engine.send_pyobj({"method": "ping"})
        self.assertTrue(ch.recv_from_rpc.poll(5000))
        self.assertEqual(ch.recv_from_rpc.recv_pyobj(), {"method": "ping"})
        ch.recv_from_rpc.send_pyobj("pong")
        self.assertTrue(engine.poll(5000))
        self.assertEqual(engine.recv_pyobj(), "pong")

    def test_only_the_exact_ipc_prefix_makes_the_scheduler_bind(self) -> None:
        cases = {
            "ipc:///tmp/rpc": True,
            "ipc://relative-name": True,
            "tcp://127.0.0.1:5000": False,
            "ipc:/one-slash": False,
            "ipcx://endpoint": False,
            "prefix-ipc://endpoint": False,
            "IPC://endpoint": False,
            "": False,
        }
        for name, expected in cases.items():
            with self.subTest(name):
                self.assertEqual(rpc_scheduler_binds(name), expected)

    def test_a_dealer_connected_to_an_unbound_ipc_path_retries(self) -> None:
        # the control for the next test: the old scheduler side, which the detector must see retrying
        path = self._temp_file_path()
        s = self.ctx.socket(zmq.DEALER)
        self.addCleanup(s.close, 0)
        s.connect(f"ipc://{path}")
        self.assertIn(zmq.EVENT_CONNECT_RETRIED, _connect_events(s, 0.8))

    def test_ipc_scheduler_makes_no_connect_attempts_without_an_engine(self) -> None:
        path = self._temp_file_path()
        args, ch = self._channels(f"ipc://{path}")
        self.assertTrue(rpc_scheduler_binds(args.rpc_ipc_name))
        self.assertEqual(_connect_events(ch.recv_from_rpc, 0.8), [])
        self._round_trip(args, ch)

    def test_tcp_endpoint_is_still_bound_by_the_engine(self) -> None:
        name = f"tcp://127.0.0.1:{_free_tcp_port()}"
        self.assertFalse(rpc_scheduler_binds(name))
        args, ch = self._channels(name)
        self._round_trip(args, ch)


if __name__ == "__main__":
    unittest.main()
