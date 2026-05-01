# ARC-AGI-2 Hybrid CNN + Transformer with TTT-LoRA

Hybrid CNN + Transformer for the ARC-AGI-2 benchmark, with test-time training (TTT) on per-task LoRA adapters. Trained on Google Colab Pro.

## Architecture (one-line)

CNN patch encoder fused with per-cell color/row/col embeddings, fed as tokens to a small (~30-80M) decoder-only transformer trained from scratch. At evaluation time, a fresh LoRA adapter is injected onto a deepcopied transformer per task and fine-tuned on the task's demonstrations.

See `/home/nitish/.claude/plans/foamy-dazzling-swing.md` for the full design.

## Quickstart (local, smoke test)

```bash
pip install -r requirements.txt
pytest tests/                          # shape + LoRA inject/discard tests
python -m arc_hybrid.train.pretrain --config configs/pretrain_small.yaml --steps 100
```

## Colab

Open `notebooks/01_pretrain_colab.ipynb` for full pretraining (Drive-mounted), then `notebooks/02_ttt_eval_colab.ipynb` for TTT evaluation.

## Layout

```
arc_hybrid/         # source package
  data/             # ARC/RE-ARC loading, augmentations, tokenization
  model/            # CNN, embeddings, transformer, hybrid wrapper
  train/            # pretrain loop, per-task TTT loop
  eval/             # exact-match eval, TTA voting, Kaggle JSON
  utils/            # checkpoint, logging
configs/            # YAML hyperparameter sets
notebooks/          # Colab pretrain + eval notebooks
tests/              # pytest unit + smoke tests
runs/               # checkpoints + results (gitignored)
```

## Status

v1: research-first; clean Colab pipeline producing a TTT-on > TTT-off ablation on ARC-AGI-2 eval.
v2 (deferred): Kaggle hardening — offline ckpts, vendored PEFT, per-task time budget guard.
