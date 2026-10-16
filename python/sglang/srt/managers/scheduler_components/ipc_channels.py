from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import zmq

from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.server_args import PortArgs
from sglang.srt.utils.network import get_zmq_socket

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )


def rpc_scheduler_binds(rpc_ipc_name: str) -> bool:
    """Which end of the rpc channel binds its endpoint.

    Without DP attention, PortArgs.init_new names the endpoint ipc://<temporary
    file>, new for each launch, so the scheduler binds it and the Engine connects.
    Under launch_server no Engine binds it. A scheduler that connected to the
    unbound endpoint retried at the ZeroMQ reconnect interval for the lifetime of
    the server, and each retry opens and closes a socket. With DP attention,
    PortArgs.init_new names a tcp endpoint from the host and the rpc port. Every DP
    group shares it, so the Engine keeps binding it.
    """
    return rpc_ipc_name.startswith("ipc://")


def engine_rpc_socket(context: zmq.Context, port_args: PortArgs) -> zmq.Socket:
    """The Engine's end of the rpc channel, the counterpart of the scheduler's."""
    return get_zmq_socket(
        context,
        zmq.DEALER,
        port_args.rpc_ipc_name,
        not rpc_scheduler_binds(port_args.rpc_ipc_name),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerIpcChannels:
    recv_from_tokenizer: Union[zmq.Socket, "ScriptedTokenizerRecvProxy"]
    recv_from_rpc: Optional[zmq.Socket]
    send_to_tokenizer: SenderWrapper
    send_to_detokenizer: SenderWrapper
    send_metrics_from_scheduler: Optional[zmq.Socket]

    @classmethod
    def create(
        cls,
        *,
        port_args: PortArgs,
        is_rank_zero: bool,
        skip_tokenizer_init: bool,
        metrics_enabled: bool,
        enable_scripted_runtime: bool,
    ) -> "SchedulerIpcChannels":
        context = zmq.Context(2)

        if is_rank_zero:
            recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            if enable_scripted_runtime:
                from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
                    ScriptedTokenizerRecvProxy,
                )

                recv_from_tokenizer = ScriptedTokenizerRecvProxy(
                    underlying=recv_from_tokenizer
                )
            recv_from_rpc = get_zmq_socket(
                context,
                zmq.DEALER,
                port_args.rpc_ipc_name,
                rpc_scheduler_binds(port_args.rpc_ipc_name),
            )

            send_to_tokenizer_raw = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if skip_tokenizer_init:
                # No decode work: send outputs straight to the tokenizer side
                # (MultiTokenizerRouter fans out when tokenizer_worker_num > 1).
                send_to_detokenizer_raw = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                send_to_detokenizer_raw = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            send_to_tokenizer = SenderWrapper(send_to_tokenizer_raw)
            send_to_detokenizer = SenderWrapper(send_to_detokenizer_raw)
        else:
            recv_from_tokenizer = None
            recv_from_rpc = None
            send_to_tokenizer = SenderWrapper(None)
            send_to_detokenizer = SenderWrapper(None)

        if metrics_enabled:
            send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )
        else:
            send_metrics_from_scheduler = None

        return cls(
            recv_from_tokenizer=recv_from_tokenizer,
            recv_from_rpc=recv_from_rpc,
            send_to_tokenizer=send_to_tokenizer,
            send_to_detokenizer=send_to_detokenizer,
            send_metrics_from_scheduler=send_metrics_from_scheduler,
        )
