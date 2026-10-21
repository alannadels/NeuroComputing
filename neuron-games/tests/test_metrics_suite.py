"""The evaluation suite: detailed test metrics, and every metric logged for every model."""
from __future__ import annotations

import pytest
import torch

from neuron_games.config import NGRunConfig
from neuron_games.model import FCSNN
from neuron_games.test_metrics import (calibration_metrics, classification_metrics,
                                       full_test_evaluation, temporal_metrics)
from neuron_games.timing import peak_memory_mib, reset_peak_memory
from neuron_games.train import train


def test_classification_metrics_hand_computed() -> None:
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    pred = torch.tensor([0, 1, 1, 1, 0, 2])
    m = classification_metrics(pred, target, 3)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["per_class_accuracy"] == pytest.approx([0.5, 1.0, 0.5])
    assert m["per_class_precision"] == pytest.approx([0.5, 2 / 3, 1.0])
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    f1 = [0.5, 0.8, 2 / 3]
    assert m["per_class_f1"] == pytest.approx(f1)
    assert m["macro_f1"] == pytest.approx(sum(f1) / 3)


def test_calibration_perfect_and_overconfident() -> None:
    target = torch.tensor([0, 1, 0, 1])
    sure = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    c = calibration_metrics(sure, target.clone(), target)
    assert c["ece"] == pytest.approx(0.0) and c["brier"] == pytest.approx(0.0)
    # always 90% sure of class 0, right half the time -> ECE 0.4
    over = torch.tensor([[0.9, 0.1]] * 4)
    c2 = calibration_metrics(over, torch.zeros(4, dtype=torch.long), target)
    assert c2["ece"] == pytest.approx(0.4)
    assert c2["brier"] == pytest.approx(((0.1 ** 2 + 0.1 ** 2) * 2 + (0.9 ** 2 + 0.9 ** 2) * 2) / 4)


def test_temporal_metrics_decision_step() -> None:
    # 3 examples, T = 4, 2 classes; cumulative counts per step
    counts = torch.tensor([
        [[0, 1], [0, 2], [0, 3], [0, 4]],   # class 1 from step 1 -> decision step 1
        [[1, 0], [1, 1], [1, 2], [1, 3]],   # 0, tie(0), 1, 1 -> last change at step 3
        [[0, 0], [0, 0], [0, 0], [0, 0]],   # never spikes -> class 0 throughout
    ])
    target = torch.tensor([1, 1, 1])
    m = temporal_metrics(counts, target)
    assert m["accuracy_over_time"] == pytest.approx([1 / 3, 1 / 3, 2 / 3, 2 / 3])
    assert m["decision_step"]["all"]["mean"] == pytest.approx((1 + 3 + 1) / 3)
    assert m["fraction_decided_by_step"] == pytest.approx([2 / 3, 2 / 3, 1.0, 1.0])
    assert m["no_output_spike_fraction"] == pytest.approx(1 / 3)


def test_peak_memory_is_none_off_cuda() -> None:
    start = reset_peak_memory(torch.device("cpu"))
    assert start is None
    assert peak_memory_mib(torch.device("cpu"), start) == {"allocated": None, "increase": None,
                                                           "reserved": None}


def _loaders(n_batches=3, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    batches = [(torch.rand(B, 1, 28, 28, generator=g), torch.randint(0, 10, (B,), generator=g))
               for _ in range(n_batches)]

    class L(list):
        batch_size = B
    return {"train": L(batches), "val": L(batches[:2]), "test": L(batches)}


SHARED_KEYS = ["epoch_phi", "epoch_components", "epoch_grad_norm", "epoch_update_norm",
               "epoch_credit_per_neuron", "epoch_phi_ref", "epoch_phi_ref_components",
               "epoch_train_time_s", "epoch_val_time_s", "epoch_diag_time_s",
               "epoch_train_peak_mem_mib", "epoch_val_peak_mem_mib", "phi_per_batch",
               "reward_per_batch"]


@pytest.mark.parametrize("condition", ["bp_anchor", "three_factor", "ng_full"])
def test_every_training_metric_for_every_model(condition, tmp_path) -> None:
    cfg = NGRunConfig.for_condition(condition, n_hidden=16, num_steps=6, epochs=2, device="cpu",
                                    diag_batches=1)
    h = train(cfg, _loaders(), torch.device("cpu"), tmp_path)["history"]
    for key in SHARED_KEYS:
        assert key in h and len(h[key]) > 0, f"{condition}: {key} missing"
    # credit is logged from the second step on, for the hidden layer, for every model
    assert h["epoch_credit_per_neuron"][-1][0] is not None
    assert len(h["epoch_credit_per_neuron"][-1][0]) == 16
    assert h["epoch_credit_per_neuron"][-1][1] is None
    ref = h["epoch_phi_ref_components"][-1]
    assert {"J", "V", "E_met", "E_bar"} <= set(ref)
    assert all(isinstance(x, float) for x in h["epoch_update_norm"][-1])


def test_full_test_evaluation_shapes() -> None:
    cfg = NGRunConfig.for_condition("three_factor", n_hidden=16, num_steps=6, device="cpu",
                                    eval_steps=(2, 4))
    torch.manual_seed(0)
    net = FCSNN(n_hidden=16)
    ev = full_test_evaluation(net, _loaders(), cfg, torch.device("cpu"))
    d = ev["detailed"]
    assert d["accuracy"] == pytest.approx(ev["metrics"]["accuracy"])
    assert len(d["temporal"]["accuracy_over_time"]) == 6
    assert len(d["confusion_matrix"]) == 10
    assert [s["num_steps"] for s in ev["sweep"]] == [2, 4, 6]
    assert ev["timing"]["n_images"] == 24
