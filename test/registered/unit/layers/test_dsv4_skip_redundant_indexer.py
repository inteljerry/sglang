"""Unit tests for the XPU skip of the C4 indexer scoring in narrow decode graphs.

In a decode graph captured at a width W of at most 4 * index_topk tokens, a row has
at most index_topk C4 entries. Top-k keeps all of them whatever the scores are, so
the backend skips the scoring and writes the in-order page indices once per forward.
"""

import itertools
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import deepseek_v4_backend as dsv4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4Metadata,
    write_sequential_c4_indices,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    C4IndexerBackendMixin,
    topk_transform_512_pytorch_vectorized,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")

INDEX_TOPK = 512
C4_PAGE_SIZE = 64  # page_size 256 // 4
NUM_C4_LAYERS = 21  # several C4 layers share the one index buffer
C4_LAYER_IDS = list(range(2, 2 * NUM_C4_LAYERS + 1, 2))
WIDTH = 4 * INDEX_TOPK  # the widest graph in which the skip holds
DEVICES = ["cpu"] + (
    ["xpu"] if hasattr(torch, "xpu") and torch.xpu.is_available() else []
)


def _make_case(
    batch_size: int, seed: int, device: str
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Per-forward metadata of a decode step replayed in a WIDTH-wide graph."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    c4_lens = torch.randint(
        0, WIDTH // 4 + 1, (batch_size,), generator=gen, dtype=torch.int32
    )
    c4_lens[0] = WIDTH // 4
    pages = WIDTH // 256
    page_table = torch.stack(
        [torch.randperm(5000, generator=gen)[:pages] for _ in range(batch_size)]
    ).to(torch.int32)
    # The attention buffer is padded to a multiple of 64 like init_flashmla_related.
    width = (INDEX_TOPK + 63) // 64 * 64
    core = SimpleNamespace(
        c4_sparse_page_indices=torch.full(
            (batch_size, width), -1, dtype=torch.int32, device=device
        ),
        c4_sparse_raw_indices=None,
        positions=torch.zeros(batch_size, dtype=torch.int64, device=device),
        swa_out_cache_loc=None,
    )
    indexer = SimpleNamespace(
        c4_seq_lens=c4_lens.to(device),
        page_table=page_table.to(device),
        c4_page_size=C4_PAGE_SIZE,
    )
    return core, indexer


def _make_backend(
    core: SimpleNamespace | None,
    indexer: SimpleNamespace | None,
    *,
    capture_width: int = WIDTH,
    skip: bool = True,
) -> DeepseekV4AttnBackend:
    """A backend built without __init__. MAX_SEQ_LEN_FOR_CAPTURE is a class
    attribute of a subclass, so it shadows an instance attribute and a read-only
    property alike."""
    backend_class = type(
        "NarrowGraphBackend",
        (DeepseekV4AttnBackend,),
        {"MAX_SEQ_LEN_FOR_CAPTURE": capture_width},
    )
    backend = object.__new__(backend_class)
    backend.forward_metadata = DSV4Metadata(
        core_attn_metadata=core, indexer_metadata=indexer
    )
    backend.token_to_kv_pool = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda loc: loc
    )
    backend.topk = 0
    backend.speculative_num_steps = 1
    backend.is_dspark_draft = False
    backend.c4_topk = INDEX_TOPK
    backend._skip_redundant_indexer = skip
    backend.debug_use_external_c4_sparse_indices = False
    backend.hisparse_coordinator = None
    backend.compressor_calls = []
    backend.forward_indexer_compressor = lambda **kw: backend.compressor_calls.append(
        kw["layer_id"]
    )
    return backend


def _decode_batch(
    batch_size: int, device: str, mode: ForwardMode = ForwardMode.DECODE
) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=mode,
        out_cache_loc=torch.zeros(batch_size, dtype=torch.int64, device=device),
        batch_size=batch_size,
    )


def _c4_indexer(layer_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer_id, index_topk=INDEX_TOPK, compressor=object()
    )


def _run_forward(
    backend: DeepseekV4AttnBackend, forward_batch: SimpleNamespace
) -> list[torch.Tensor]:
    """init_forward_metadata_in_graph, then every C4 layer's indexer, recording the
    buffer each layer's attention reads (a clone right after the indexer)."""
    backend.init_forward_metadata_in_graph(forward_batch)
    seen = []
    for layer_id in C4_LAYER_IDS:
        x = torch.zeros(forward_batch.batch_size, 8)
        backend.forward_c4_indexer(
            x=x,
            q_lora=x,
            c4_indexer=_c4_indexer(layer_id),
            forward_batch=forward_batch,
        )
        seen.append(
            backend.forward_metadata.core_attn_metadata.c4_sparse_page_indices.clone()
        )
    return seen


def _per_layer_reference(
    core: SimpleNamespace, indexer: SimpleNamespace
) -> list[torch.Tensor]:
    """What a write by every layer would leave for each layer."""
    buf = core.c4_sparse_page_indices.clone()
    seen = []
    for _ in C4_LAYER_IDS:
        write_sequential_c4_indices(
            indexer.c4_seq_lens,
            indexer.page_table,
            buf,
            indexer.c4_page_size,
            INDEX_TOPK,
            None,
        )
        seen.append(buf.clone())
    return seen


# Every condition of the predicate, set to the value that allows the skip.
PREDICATE_CONDITIONS = {
    "skip": True,
    "mode": ForwardMode.DECODE,
    "capture": True,
    "width": WIDTH,
    "external": False,
    "hisparse": None,
    "capturer": None,
}


def _fake_scoring(scored: list[int]) -> Callable[..., None]:
    def scoring(self: object, **kwargs: object) -> None:
        scored.append(kwargs["c4_indexer"].layer_id)

    return scoring


class TestSequentialIndicesMatchTopk(CustomTestCase):
    """The claim behind the skip: a row of at most topk entries gets all of them."""

    @staticmethod
    def _buffers(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.full((batch_size, INDEX_TOPK), -7, dtype=torch.int32),
            torch.full((batch_size, INDEX_TOPK), -7, dtype=torch.int32),
        )

    def test_rows_up_to_topk_equal_the_topk_transform(self):
        for batch_size, seed in itertools.product((1, 2, 4, 8), range(3)):
            with self.subTest(batch_size=batch_size, seed=seed):
                gen = torch.Generator().manual_seed(seed)
                lens = torch.randint(
                    0, INDEX_TOPK + 1, (batch_size,), generator=gen, dtype=torch.int32
                )
                lens[0] = INDEX_TOPK
                scores = torch.randn(batch_size, INDEX_TOPK + 64, generator=gen)
                page_table = torch.stack(
                    [torch.randperm(5000, generator=gen)[:8] for _ in range(batch_size)]
                ).to(torch.int32)
                want, want_raw = self._buffers(batch_size)
                got, got_raw = self._buffers(batch_size)
                topk_transform_512_pytorch_vectorized(
                    scores, lens, page_table, want, C4_PAGE_SIZE, want_raw
                )
                write_sequential_c4_indices(
                    lens, page_table, got, C4_PAGE_SIZE, INDEX_TOPK, got_raw
                )
                self.assertTrue(torch.equal(got, want))
                self.assertTrue(torch.equal(got_raw, want_raw))

    def test_row_above_topk_is_not_sequential(self):
        """Negative control: with topk + 1 entries the top-k drops the lowest score,
        so the skip would be wrong. This is why a graph may be at most 4 * topk wide."""
        lens = torch.tensor([INDEX_TOPK + 1], dtype=torch.int32)
        scores = torch.ones(1, INDEX_TOPK + 1)
        scores[0, 0] = -1.0
        page_table = torch.arange(9, dtype=torch.int32).unsqueeze(0)
        want, _ = self._buffers(1)
        got, _ = self._buffers(1)
        topk_transform_512_pytorch_vectorized(
            scores, lens, page_table, want, C4_PAGE_SIZE
        )
        write_sequential_c4_indices(
            lens, page_table, got, C4_PAGE_SIZE, INDEX_TOPK, None
        )
        self.assertFalse(torch.equal(got, want))


class TestIndexerScoringRedundant(CustomTestCase):
    """Each condition of the predicate guards a case where skipping is wrong."""

    def _redundant(self, **changes) -> bool:
        """The predicate with PREDICATE_CONDITIONS, changed by `changes`."""
        cond = {**PREDICATE_CONDITIONS, **changes}
        backend = _make_backend(
            None, None, capture_width=cond["width"], skip=cond["skip"]
        )
        backend.debug_use_external_c4_sparse_indices = cond["external"]
        backend.hisparse_coordinator = cond["hisparse"]
        with (
            patch.object(
                dsv4_backend, "get_is_capture_mode", return_value=cond["capture"]
            ),
            patch.object(
                dsv4_backend,
                "get_global_indexer_capturer",
                return_value=cond["capturer"],
            ),
        ):
            return backend._indexer_scoring_redundant(
                SimpleNamespace(forward_mode=cond["mode"]), INDEX_TOPK
            )

    def test_skips_in_a_narrow_decode_graph(self):
        self.assertTrue(self._redundant())

    def test_each_other_condition_disables_the_skip(self):
        cases = {
            "switch off": {"skip": False},
            "extend": {"mode": ForwardMode.EXTEND},
            "target verify": {"mode": ForwardMode.TARGET_VERIFY},
            "eager": {"capture": False},
            "external indices": {"external": True},
            "hisparse": {"hisparse": object()},
            "indexer capture": {"capturer": object()},
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertFalse(self._redundant(**kwargs))

    def test_width_bound_is_four_times_topk(self):
        self.assertTrue(self._redundant(width=WIDTH))
        self.assertFalse(self._redundant(width=WIDTH + 1))


class TestSequentialC4IndicesOncePerForward(CustomTestCase):
    def _patched(self, capture_mode: bool) -> tuple:
        return (
            patch.object(
                dsv4_backend, "get_is_capture_mode", return_value=capture_mode
            ),
            patch.object(
                dsv4_backend, "get_global_indexer_capturer", return_value=None
            ),
        )

    def _check_once_per_forward(self, device: str, batch_size: int, seed: int) -> None:
        core, indexer = _make_case(batch_size, seed, device)
        want = _per_layer_reference(core, indexer)
        backend = _make_backend(core, indexer)
        calls = []
        real = dsv4_backend.write_sequential_c4_indices

        def counting(*args: object, **kwargs: object) -> None:
            calls.append(1)
            return real(*args, **kwargs)

        with patch.object(dsv4_backend, "write_sequential_c4_indices", counting):
            got = _run_forward(backend, _decode_batch(batch_size, device))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(got), len(want))
        for layer, (g, w) in enumerate(zip(got, want)):
            self.assertTrue(torch.equal(g, w), f"layer {C4_LAYER_IDS[layer]}")
        # The compressor still runs in every C4 layer.
        self.assertEqual(backend.compressor_calls, C4_LAYER_IDS)

    def test_once_per_forward_equals_per_layer(self):
        cap, capturer = self._patched(True)
        with cap, capturer:
            cases = itertools.product(DEVICES, (1, 2, 4, 8), range(5))
            for device, batch_size, seed in cases:
                with self.subTest(device=device, batch_size=batch_size, seed=seed):
                    self._check_once_per_forward(device, batch_size, seed)

    def test_skip_compressor_flag_avoids_the_compressor(self):
        """The compressor is skipped when the caller asks for it, also in the skip path."""
        cap, capturer = self._patched(True)
        with cap, capturer:
            core, indexer = _make_case(2, 0, "cpu")
            backend = _make_backend(core, indexer)
            batch = _decode_batch(2, "cpu")
            backend.init_forward_metadata_in_graph(batch)
            x = torch.zeros(2, 8)
            backend.forward_c4_indexer(
                x=x,
                q_lora=x,
                c4_indexer=_c4_indexer(C4_LAYER_IDS[0]),
                forward_batch=batch,
                skip_compressor=True,
            )
        self.assertEqual(backend.compressor_calls, [])

    def test_buffer_is_not_left_unwritten(self):
        """Negative control: the buffer starts at -1 and a row with entries changes."""
        cap, capturer = self._patched(True)
        with cap, capturer:
            core, indexer = _make_case(4, 0, "cpu")
            before = core.c4_sparse_page_indices.clone()
            backend = _make_backend(core, indexer)
            got = _run_forward(backend, _decode_batch(4, "cpu"))
            self.assertFalse(torch.equal(got[0], before))

    def _assert_scores_every_layer(self, capture_mode: bool, **backend_kwargs) -> None:
        cap, capturer = self._patched(capture_mode)
        scored = []
        with (
            cap,
            capturer,
            patch.object(
                C4IndexerBackendMixin, "forward_c4_indexer", _fake_scoring(scored)
            ),
        ):
            core, indexer = _make_case(2, 0, "cpu")
            before = core.c4_sparse_page_indices.clone()
            backend = _make_backend(core, indexer, **backend_kwargs)
            got = _run_forward(backend, _decode_batch(2, "cpu"))
        self.assertEqual(scored, C4_LAYER_IDS)
        self.assertEqual(backend.compressor_calls, [])
        self.assertTrue(torch.equal(got[-1], before))

    def test_eager_forward_scores_in_every_layer(self):
        """Outside capture the skip never applies: no write at init, and every C4
        layer takes the scoring path."""
        self._assert_scores_every_layer(False)

    def test_wider_graph_scores_in_every_layer(self):
        """A graph wider than 4 * index_topk keeps scoring and writes nothing at init."""
        self._assert_scores_every_layer(True, capture_width=2 * WIDTH)

    def test_disabled_switch_scores_in_every_layer(self):
        self._assert_scores_every_layer(True, skip=False)


if __name__ == "__main__":
    unittest.main()
