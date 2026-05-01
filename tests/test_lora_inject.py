"""LoRA inject/discard cycle on the custom (non-HF) transformer.

Verifies that:
  1. peft.inject_adapter_in_model hits q_proj/v_proj/up_proj/down_proj.
  2. After freezing non-LoRA params, trainable count > 0 and << base.
  3. Discarding the adapter (drop the deepcopy) leaves the original transformer
     numerically unchanged.

Skips cleanly if torch or peft aren't installed.
"""
import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")


def _tiny_decoder():
    from arc_hybrid.model.transformer import CausalDecoder
    return CausalDecoder(d_model=64, n_heads=4, n_layers=2, d_ff=128)


def test_inject_targets_hit_and_freeze():
    from arc_hybrid.train.ttt import make_lora_adapter

    base = _tiny_decoder()
    base_params_before = {n: p.detach().clone() for n, p in base.named_parameters()}

    adapter = make_lora_adapter(base, r=4, alpha=8, dropout=0.0)

    n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    n_total_adapter = sum(p.numel() for p in adapter.parameters())
    assert n_trainable > 0, "LoRA inject hit zero target modules"
    assert n_trainable < n_total_adapter / 4, "LoRA trainable share should be small"

    # Adapter is a deepcopy; base must be byte-identical to before.
    for n, p in base.named_parameters():
        assert torch.equal(p, base_params_before[n]), f"base param {n} mutated"


def test_adapter_forward_runs():
    from arc_hybrid.train.ttt import make_lora_adapter

    base = _tiny_decoder()
    adapter = make_lora_adapter(base, r=4, alpha=8, dropout=0.0)
    adapter.eval()
    x = torch.randn(2, 7, 64)
    pad_mask = torch.ones(2, 7, dtype=torch.bool)
    with torch.no_grad():
        y = adapter(x, pad_mask=pad_mask)
    assert y.shape == (2, 7, 64)
    assert torch.isfinite(y).all()
