"""KV-cache parity test.

Greedy decode with use_kv_cache=True must produce the same output grid as
the legacy re-encode path (use_kv_cache=False) for the same model, weights,
and task. If this ever fails, the cache implementation has drifted from the
reference; do not trust eval results from the cached path.
"""
import pytest

torch = pytest.importorskip("torch")
import numpy as np

from arc_hybrid.data.arc_loader import Pair, Task
from arc_hybrid.data.tokenize import VOCAB_SIZE
from arc_hybrid.eval.evaluate import greedy_decode
from arc_hybrid.model.hybrid import HybridModel


def _tiny_model(seed: int = 0) -> HybridModel:
    torch.manual_seed(seed)
    return HybridModel(
        d_model=64, n_heads=4, n_layers=3, d_ff=128,
        cnn_channels=16, cnn_blocks=2, max_grid=30, vocab_size=VOCAB_SIZE,
    )


def _tiny_task(seed: int = 0) -> Task:
    rng = np.random.default_rng(seed)
    def grid(h, w):
        return rng.integers(0, 5, size=(h, w), dtype=np.int8)
    return Task(
        task_id="parity-tiny",
        train=[Pair(grid(3, 3), grid(3, 3)), Pair(grid(2, 4), grid(4, 2))],
        test=[Pair(grid(3, 3), grid(3, 3))],
    )


def test_kv_cache_matches_legacy_decode():
    model = _tiny_model().eval()
    task = _tiny_task()
    out_cached = greedy_decode(model, task, max_grid=30, device="cpu",
                                max_new_tokens=64, use_kv_cache=True)
    out_legacy = greedy_decode(model, task, max_grid=30, device="cpu",
                                max_new_tokens=64, use_kv_cache=False)
    if out_cached is None and out_legacy is None:
        return
    assert out_cached is not None and out_legacy is not None, (
        f"one path returned None: cached={out_cached!r}, legacy={out_legacy!r}"
    )
    assert out_cached.shape == out_legacy.shape, (
        f"shape mismatch: cached {out_cached.shape}, legacy {out_legacy.shape}"
    )
    assert (out_cached == out_legacy).all(), (
        f"token mismatch:\ncached =\n{out_cached}\nlegacy =\n{out_legacy}"
    )


def test_prefill_logits_match_no_cache():
    """Prefill (use_cache=True with past_kv_list=None) must produce the same
    logits as the no-cache forward at every position."""
    model = _tiny_model().eval()
    from arc_hybrid.data.tokenize import collate_batch, pack_task

    task = _tiny_task()
    packed = pack_task(task, test_idx=0, include_test_output=True, max_grid=30)
    batch = collate_batch([packed], max_grid=30)

    with torch.no_grad():
        logits_no = model(**batch)
        logits_yes, kv = model(**batch, use_cache=True)

    assert torch.allclose(logits_no, logits_yes, atol=1e-5), \
        "use_cache=True prefill diverges from no-cache forward"
    assert kv is not None and len(kv) == len(model.transformer.layers), \
        f"expected {len(model.transformer.layers)} cache entries, got {len(kv) if kv else None}"
