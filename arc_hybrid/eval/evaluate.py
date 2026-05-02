"""Greedy decoding + exact-match evaluation harness.

Two-attempt submissions:
    attempt_1 — TTA-voted (D8 majority vote)
    attempt_2 — greedy without TTA (a separate failure mode for safety)

A task is scored correct if either attempt exactly matches the ground-truth
output grid for the first test pair. Outputs Kaggle-format JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data.arc_loader import Task
from ..data.tokenize import (
    EOG,
    EOR,
    ROLE_SPECIAL,
    ROLE_TEST_OUT,
    collate_batch,
    pack_task,
)
from .voting import tta_predict


@torch.no_grad()
def greedy_decode(
    model,
    task: Task,
    *,
    test_idx: int = 0,
    max_grid: int = 30,
    device: str = "cuda",
    max_new_tokens: int | None = None,
    use_kv_cache: bool = True,
) -> np.ndarray | None:
    """KV-cached greedy decoding.

    Prefill: one full forward pass over the prefix to populate the per-layer
    K/V cache. Subsequent steps feed only the newly-generated single token,
    reusing cached past K/V so attention costs O(L) instead of O(L^2).

    The CNN runs only on the prefill step (the only call that has cells with
    grid_in_sample>=0). Generated tokens are CNN-blind by construction
    (grid_in_sample=-1), so the decode-step path skips the CNN entirely.

    Set use_kv_cache=False to fall back to the legacy re-encode-each-step path
    (e.g. for parity tests).
    """
    model.eval()
    packed = pack_task(task, test_idx=test_idx, include_test_output=False, max_grid=max_grid)

    grids_t = packed.grids.unsqueeze(0).to(device)
    grid_sizes_t = packed.grid_sizes.unsqueeze(0).to(device)

    tok = packed.token_ids.tolist()
    row = packed.row_ids.tolist()
    col = packed.col_ids.tolist()
    role = packed.role_ids.tolist()
    gis = packed.grid_in_sample.tolist()
    cmask = packed.cell_mask.tolist()

    rows: list[list[int]] = [[]]
    cur_r, cur_c = 0, 0
    if max_new_tokens is None:
        max_new_tokens = max_grid * (max_grid + 1) + 1

    if not use_kv_cache:
        return _legacy_greedy_decode(
            model, packed, grids_t, grid_sizes_t,
            tok, row, col, role, gis, cmask,
            rows, cur_r, cur_c, max_grid, device, max_new_tokens,
        )

    # ---- KV-cache path ----
    L = len(tok)
    prefill_batch = {
        "token_ids": torch.tensor([tok], dtype=torch.long, device=device),
        "row_ids": torch.tensor([row], dtype=torch.long, device=device),
        "col_ids": torch.tensor([col], dtype=torch.long, device=device),
        "role_ids": torch.tensor([role], dtype=torch.long, device=device),
        "grid_in_sample": torch.tensor([gis], dtype=torch.long, device=device),
        "cell_mask": torch.tensor([cmask], dtype=torch.bool, device=device),
        "pad_mask": torch.ones(1, L, dtype=torch.bool, device=device),
        "grids": grids_t,
        "grid_sizes": grid_sizes_t,
    }
    logits, kv_cache = model(**prefill_batch, use_cache=True)
    next_tok = int(logits[0, -1].argmax().item())

    def _emit_and_step(nt: int) -> bool:
        """Update tok/row/col/role/gis/cmask and the rows accumulator.
        Returns True if decoding should stop."""
        nonlocal cur_r, cur_c
        if nt == EOG:
            return True
        if nt == EOR:
            tok.append(EOR); row.append(cur_r); col.append(cur_c); role.append(ROLE_SPECIAL)
            gis.append(-1); cmask.append(False)
            cur_r += 1
            cur_c = 0
            if cur_r >= max_grid:
                return True
            rows.append([])
            return False
        if 0 <= nt <= 9:
            tok.append(nt); row.append(cur_r); col.append(cur_c); role.append(ROLE_TEST_OUT)
            gis.append(-1); cmask.append(True)
            rows[-1].append(nt)
            cur_c += 1
            if cur_c >= max_grid:
                tok.append(EOR); row.append(cur_r); col.append(cur_c); role.append(ROLE_SPECIAL)
                gis.append(-1); cmask.append(False)
                cur_r += 1
                cur_c = 0
                if cur_r >= max_grid:
                    return True
                rows.append([])
            return False
        return True  # unexpected token id; stop

    for _ in range(max_new_tokens):
        stop = _emit_and_step(next_tok)
        if stop:
            break
        # Build a single-token forward for the just-emitted token.
        last = -1
        step_batch = {
            "token_ids": torch.tensor([[tok[last]]], dtype=torch.long, device=device),
            "row_ids": torch.tensor([[row[last]]], dtype=torch.long, device=device),
            "col_ids": torch.tensor([[col[last]]], dtype=torch.long, device=device),
            "role_ids": torch.tensor([[role[last]]], dtype=torch.long, device=device),
            "grid_in_sample": torch.tensor([[gis[last]]], dtype=torch.long, device=device),
            "cell_mask": torch.tensor([[cmask[last]]], dtype=torch.bool, device=device),
            "pad_mask": torch.ones(1, 1, dtype=torch.bool, device=device),
            "grids": grids_t,             # unused on decode step but keeps API stable
            "grid_sizes": grid_sizes_t,
        }
        logits, kv_cache = model(**step_batch, use_cache=True, past_kv_list=kv_cache)
        next_tok = int(logits[0, -1].argmax().item())

    while rows and not rows[-1]:
        rows.pop()
    if not rows:
        return None
    W = max(len(r) for r in rows)
    if W == 0:
        return None
    out = np.zeros((len(rows), W), dtype=np.int8)
    for ri, r in enumerate(rows):
        out[ri, : len(r)] = r
    return out


def _legacy_greedy_decode(
    model, packed, grids_t, grid_sizes_t,
    tok, row, col, role, gis, cmask,
    rows, cur_r, cur_c, max_grid, device, max_new_tokens,
):
    """Re-encode-each-step greedy decoding. Slow (quadratic) but kept for parity
    testing against the cached path."""
    for _ in range(max_new_tokens):
        L = len(tok)
        b = {
            "token_ids": torch.tensor([tok], dtype=torch.long, device=device),
            "row_ids": torch.tensor([row], dtype=torch.long, device=device),
            "col_ids": torch.tensor([col], dtype=torch.long, device=device),
            "role_ids": torch.tensor([role], dtype=torch.long, device=device),
            "grid_in_sample": torch.tensor([gis], dtype=torch.long, device=device),
            "cell_mask": torch.tensor([cmask], dtype=torch.bool, device=device),
            "pad_mask": torch.ones(1, L, dtype=torch.bool, device=device),
            "grids": grids_t,
            "grid_sizes": grid_sizes_t,
        }
        logits = model(**b)
        next_tok = int(logits[0, -1].argmax().item())
        if next_tok == EOG:
            break
        if next_tok == EOR:
            tok.append(EOR); row.append(cur_r); col.append(cur_c); role.append(ROLE_SPECIAL)
            gis.append(-1); cmask.append(False)
            cur_r += 1
            cur_c = 0
            if cur_r >= max_grid:
                break
            rows.append([])
            continue
        if 0 <= next_tok <= 9:
            tok.append(next_tok); row.append(cur_r); col.append(cur_c); role.append(ROLE_TEST_OUT)
            gis.append(-1); cmask.append(True)
            rows[-1].append(next_tok)
            cur_c += 1
            if cur_c >= max_grid:
                tok.append(EOR); row.append(cur_r); col.append(cur_c); role.append(ROLE_SPECIAL)
                gis.append(-1); cmask.append(False)
                cur_r += 1
                cur_c = 0
                if cur_r >= max_grid:
                    break
                rows.append([])
            continue
        break
    while rows and not rows[-1]:
        rows.pop()
    if not rows:
        return None
    W = max(len(r) for r in rows)
    if W == 0:
        return None
    out = np.zeros((len(rows), W), dtype=np.int8)
    for ri, r in enumerate(rows):
        out[ri, : len(r)] = r
    return out


def predict_two_attempts(model, task: Task, *, max_grid: int = 30, device: str = "cuda", n_aug: int = 8):
    def gen(t: Task) -> np.ndarray | None:
        return greedy_decode(model, t, max_grid=max_grid, device=device)

    attempt_1 = tta_predict(gen, task, n_aug=n_aug)
    attempt_2 = greedy_decode(model, task, max_grid=max_grid, device=device)
    return attempt_1, attempt_2


def _exact_match(pred: np.ndarray | None, target: np.ndarray) -> bool:
    if pred is None:
        return False
    if pred.shape != target.shape:
        return False
    return bool((pred == target).all())


def evaluate_split(
    model,
    tasks: list[Task],
    *,
    use_ttt: bool = False,
    ttt_kwargs: dict | None = None,
    max_grid: int = 30,
    device: str = "cuda",
    n_aug: int = 8,
    out_dir: str | Path = "runs/eval",
    tag: str = "ttt_off",
) -> dict:
    import time
    from tqdm.auto import tqdm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_ttt:
        from ..train.ttt import swap_transformer, train_ttt_adapter

    submission: dict[str, list[dict]] = {}
    per_task: dict[str, dict] = {}
    n_correct = 0
    n_total = 0
    pbar = tqdm(tasks, desc=tag, leave=True, dynamic_ncols=True)
    for task in pbar:
        t0 = time.time()
        gt = task.test[0].output
        if use_ttt:
            adapter = train_ttt_adapter(model, task, max_grid=max_grid, device=device, **(ttt_kwargs or {}))
            if adapter is None:
                a1, a2 = predict_two_attempts(model, task, max_grid=max_grid, device=device, n_aug=n_aug)
            else:
                with swap_transformer(model, adapter):
                    a1, a2 = predict_two_attempts(model, task, max_grid=max_grid, device=device, n_aug=n_aug)
        else:
            a1, a2 = predict_two_attempts(model, task, max_grid=max_grid, device=device, n_aug=n_aug)

        correct = False
        if gt is not None:
            correct = _exact_match(a1, gt) or _exact_match(a2, gt)
            n_total += 1
            n_correct += int(correct)
        per_task[task.task_id] = {
            "correct": correct,
            "shape_a1": None if a1 is None else list(a1.shape),
            "shape_a2": None if a2 is None else list(a2.shape),
            "wall_s": round(time.time() - t0, 2),
        }
        submission[task.task_id] = [
            {
                "attempt_1": [] if a1 is None else a1.astype(int).tolist(),
                "attempt_2": [] if a2 is None else a2.astype(int).tolist(),
            }
        ]
        running_acc = (n_correct / n_total) if n_total else 0.0
        pbar.set_postfix(
            acc=f"{running_acc:.3f}",
            correct=f"{n_correct}/{n_total}",
            last=f"{per_task[task.task_id]['wall_s']:.1f}s",
            mark="✓" if correct else "·",
        )

    summary = {
        "tag": tag,
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_total) if n_total else 0.0,
    }
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"per_task_{tag}.json").write_text(json.dumps(per_task, indent=2))
    (out_dir / f"submission_{tag}.json").write_text(json.dumps(submission, indent=2))
    print(f"[{tag}] {n_correct}/{n_total} correct ({summary['accuracy']:.3f})")
    return summary
