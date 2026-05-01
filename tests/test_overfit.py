"""Single-task overfit smoke test.

Feeds one ARC task repeatedly; the model must drive loss to ~0 and predict the
held-out test pair correctly. If this fails, every later metric is suspect.

Filled in once the pretrain loop (Step 6) and data layer (Step 2) exist.
"""
import pytest

pytest.importorskip("torch")


@pytest.mark.skip(reason="implemented alongside arc_hybrid/train/pretrain.py")
def test_single_task_overfits():
    pass
