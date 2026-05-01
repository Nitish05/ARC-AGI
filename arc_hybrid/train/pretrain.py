"""Pretraining loop for the hybrid CNN + Transformer.

Usage:
    python -m arc_hybrid.train.pretrain --config configs/pretrain_small.yaml
    python -m arc_hybrid.train.pretrain --config configs/pretrain_medium.yaml --steps 100
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.arc_loader import Task, filter_max_grid, load_split
from ..data.augment import augment_task
from ..data.tokenize import collate_batch, pack_task
from ..model.hybrid import build_hybrid_from_config
from ..utils.checkpoint import save_full, save_slim
from ..utils.config import asdict, load_config
from ..utils.logging import JsonlLogger


def gather_tasks(cfg) -> list[Task]:
    """Load every dataset listed in cfg.data that exists on disk.

    Missing paths are skipped silently — Colab notebooks may opt into a subset
    (e.g. ARC-AGI-2 only) by simply not downloading RE-ARC.
    """
    # Both arcprize/ARC-AGI-2 and fchollet/ARC-AGI lay out JSONs under <repo>/data/{training,evaluation}.
    # Keep the flattened paths too in case the user copies just the JSON folders.
    # ARC-AGI-2 evaluation/ is HELD OUT — that is the benchmark we score against in notebook 02,
    # so training on it would inflate eval accuracy. ARC-AGI-1 eval is fine to use as training
    # data since we don't score against it.
    arc2 = Path(cfg.data.arc_agi_2_path)
    arc1 = Path(cfg.data.arc_agi_1_path)
    candidates = [
        (arc2 / "data" / "training", "*.json"),
        (arc2 / "training", "*.json"),
        (arc1 / "data" / "training", "*.json"),
        (arc1 / "data" / "evaluation", "*.json"),
        (arc1 / "training", "*.json"),
        (arc1 / "evaluation", "*.json"),
        (Path(cfg.data.re_arc_path), "*.json"),
    ]
    tasks: list[Task] = []
    for root, pat in candidates:
        if root.exists():
            loaded = load_split(root, pat)
            tasks.extend(loaded)
            print(f"  + {root}: {len(loaded)} tasks")
    return filter_max_grid(tasks, cfg.model.max_grid_size)


def make_iterator(tasks: list[Task], cfg, rng: np.random.Generator):
    aug_cfg = dict(
        d8=cfg.data.augment.d8,
        color_perm=cfg.data.augment.color_perm,
        demo_dropout=cfg.data.augment.demo_dropout,
    )
    while True:
        task = tasks[int(rng.integers(0, len(tasks)))]
        all_pairs = [p for p in task.train + task.test if p.output is not None]
        if len(all_pairs) < 2:
            continue
        target_idx = int(rng.integers(0, len(all_pairs)))
        shadow = Task(
            task_id=task.task_id,
            train=[p for i, p in enumerate(all_pairs) if i != target_idx],
            test=[all_pairs[target_idx]],
        )
        shadow = augment_task(shadow, rng, **aug_cfg)
        try:
            yield pack_task(
                shadow, test_idx=0, include_test_output=True, max_grid=cfg.model.max_grid_size
            )
        except ValueError:
            continue


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    pct = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * pct))


def _model_forward(model, batch):
    return model(
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


def _shifted_loss(logits: torch.Tensor, token_ids: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    V = logits.size(-1)
    shift_logits = logits[:, :-1, :].reshape(-1, V)
    shift_targets = token_ids[:, 1:].reshape(-1)
    shift_tmask = target_mask[:, 1:].reshape(-1)
    if shift_tmask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(shift_logits[shift_tmask], shift_targets[shift_tmask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.train.steps = args.steps

    torch.manual_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)

    # Hopper / Ampere matmul knobs. bf16 path is unaffected; this only nudges any
    # fp32 fallbacks (norms, accumulators) onto TF32 tensor cores.
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"gpu: {gpu_name}")

    print("loading tasks...")
    tasks = gather_tasks(cfg)
    print(f"total {len(tasks)} tasks (filtered to max_grid={cfg.model.max_grid_size})")
    if not tasks:
        raise RuntimeError("no tasks found; check data paths in config")

    model = build_hybrid_from_config(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M")

    # torch.compile gives a meaningful win by collapsing many small kernels
    # (this model has lots of them — small CNN + per-cell gathers). On
    # Hopper/Blackwell (sm >= 9.0), Inductor's Triton templates often beat
    # stock cudnn, so max-autotune is worth its long warmup. On Ampere
    # (A100, sm 8.0), cudnn is mature enough that Triton usually loses,
    # which makes autotune pure overhead — drop to 'default' there.
    if args.device.startswith("cuda"):
        try:
            cap = torch.cuda.get_device_capability(0)
            compile_mode = "max-autotune-no-cudagraphs" if cap[0] >= 9 else "default"
            model = torch.compile(model, mode=compile_mode)
            print(f"torch.compile: enabled (mode={compile_mode}, sm={cap[0]}.{cap[1]})")
        except Exception as e:
            print(f"torch.compile skipped: {e!r}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95),
    )
    start_step = 1
    if args.resume:
        from ..utils.checkpoint import load_into

        ckpt = load_into(args.resume, model=model, optimizer=optimizer, map_location=args.device)
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"resumed from {args.resume} at step {start_step}")

    out_dir = Path(cfg.logging.out_dir)
    logger = JsonlLogger(out_dir / "train.jsonl")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.train.precision, torch.float32)
    use_amp = amp_dtype is not torch.float32
    scaler = torch.amp.GradScaler("cuda") if amp_dtype is torch.float16 else None
    autocast_device = "cuda" if args.device.startswith("cuda") else "cpu"

    it = make_iterator(tasks, cfg, rng)
    model.train()
    for step in range(start_step, cfg.train.steps + 1):
        batch = collate_batch([next(it) for _ in range(cfg.train.batch_tasks)], max_grid=cfg.model.max_grid_size)
        batch = {k: v.to(args.device) for k, v in batch.items()}

        lr = cosine_lr(step, cfg.train.warmup_steps, cfg.train.steps, cfg.train.lr, cfg.train.lr_min)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(autocast_device, dtype=amp_dtype, enabled=use_amp):
            logits = _model_forward(model, batch)
            loss = _shifted_loss(logits, batch["token_ids"], batch["target_mask"])

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        if step % cfg.logging.log_every == 0:
            logger.log(step=step, lr=lr, loss=float(loss.detach().item()))
        if step % cfg.logging.ckpt_every == 0:
            save_full(out_dir / f"ckpt_{step}.pt", model=model, optimizer=optimizer,
                      scheduler=None, step=step, config=asdict(cfg))
            save_slim(out_dir / f"slim_{step}.pt", model=model, config=asdict(cfg))

    save_slim(out_dir / "slim_final.pt", model=model, config=asdict(cfg))
    logger.close()


if __name__ == "__main__":
    main()
