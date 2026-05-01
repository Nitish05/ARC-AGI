"""Per-task test-time training via PEFT LoRA adapters.

Per evaluation task:
  1. deepcopy base_model.transformer; inject LoRA on q_proj/v_proj/up_proj/down_proj.
  2. Freeze all non-LoRA params on the adapter; CNN/embeddings/lm_head stay frozen too.
  3. Build leave-one-out + augmented examples from the demo pairs; train for N steps.
  4. Caller swaps the trained adapter into base_model for inference, then restores.

Uses peft.inject_adapter_in_model (not get_peft_model) since the model is custom
and not a HuggingFace base model. target_modules matches by suffix on
named_modules(); a missed target would silently produce a no-op adapter, so we
verify the trainable param count is non-zero before returning.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from ..data.arc_loader import Task
from ..data.augment import augment_task
from ..data.tokenize import collate_batch, pack_task

DEFAULT_TARGETS = ("q_proj", "v_proj", "up_proj", "down_proj")


def _freeze_non_lora(module: torch.nn.Module) -> int:
    n_train = 0
    for n, p in module.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
            n_train += p.numel()
        else:
            p.requires_grad = False
    return n_train


def make_lora_adapter(
    transformer: torch.nn.Module,
    *,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = DEFAULT_TARGETS,
) -> torch.nn.Module:
    from peft import LoraConfig, inject_adapter_in_model

    adapter = deepcopy(transformer)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    inject_adapter_in_model(cfg, adapter)
    n_train = _freeze_non_lora(adapter)
    if n_train == 0:
        raise RuntimeError(
            f"LoRA inject hit no targets: target_modules={target_modules} did not match any "
            "nn.Linear under named_modules(). Verify transformer.py module names."
        )
    return adapter


@contextmanager
def swap_transformer(model, new_transformer):
    saved = model.transformer
    model.transformer = new_transformer
    try:
        yield
    finally:
        model.transformer = saved


def _ttt_examples(task: Task, rng: np.random.Generator, *, augment_cfg: dict, max_grid: int, n_examples: int):
    examples = []
    pairs = [p for p in task.train if p.output is not None]
    if len(pairs) < 2:
        return examples
    for _ in range(n_examples):
        target_idx = int(rng.integers(0, len(pairs)))
        shadow = Task(
            task_id=task.task_id,
            train=[p for i, p in enumerate(pairs) if i != target_idx],
            test=[pairs[target_idx]],
        )
        shadow = augment_task(shadow, rng, **augment_cfg)
        try:
            examples.append(pack_task(shadow, test_idx=0, include_test_output=True, max_grid=max_grid))
        except ValueError:
            continue
    return examples


def train_ttt_adapter(
    base_model,
    task: Task,
    *,
    steps: int = 100,
    lr: float = 5e-4,
    batch_size: int = 4,
    n_examples: int = 64,
    max_grid: int = 30,
    device: str = "cuda",
    seed: int = 0,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """Train a fresh LoRA adapter on the demos of `task`. Returns the adapter
    module (or None if there aren't enough demos to construct an LOO split).

    Caller is responsible for swapping the adapter into `base_model.transformer`
    for inference (use `swap_transformer`) and then restoring afterwards.
    """
    augment_cfg = dict(d8=True, color_perm=True, demo_dropout=False)
    rng = np.random.default_rng(seed)
    examples = _ttt_examples(task, rng, augment_cfg=augment_cfg, max_grid=max_grid, n_examples=n_examples)
    if not examples:
        return None

    adapter = make_lora_adapter(
        base_model.transformer, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
    ).to(device)
    optim = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=lr)

    for p in base_model.cnn.parameters():
        p.requires_grad = False
    for p in base_model.cnn_to_d.parameters():
        p.requires_grad = False
    for p in base_model.embeddings.parameters():
        p.requires_grad = False
    for p in base_model.lm_head.parameters():
        p.requires_grad = False

    with swap_transformer(base_model, adapter):
        base_model.train()
        for step in range(1, steps + 1):
            idx = rng.choice(len(examples), size=min(batch_size, len(examples)), replace=False)
            batch = collate_batch([examples[int(i)] for i in idx], max_grid=max_grid)
            batch = {k: v.to(device) for k, v in batch.items()}

            optim.zero_grad(set_to_none=True)
            logits = base_model(
                token_ids=batch["token_ids"],
                row_ids=batch["row_ids"],
                col_ids=batch["col_ids"],
                role_ids=batch["role_ids"],
                grid_in_sample=batch["grid_in_sample"],
                cell_mask=batch["cell_mask"],
                pad_mask=batch["pad_mask"],
                grids=batch["grids"],
                grid_sizes=batch["grid_sizes"],
            )
            V = logits.size(-1)
            sl = logits[:, :-1, :].reshape(-1, V)
            st = batch["token_ids"][:, 1:].reshape(-1)
            sm = batch["target_mask"][:, 1:].reshape(-1)
            if sm.sum() == 0:
                continue
            loss = F.cross_entropy(sl[sm], st[sm])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.0)
            optim.step()
    return adapter
