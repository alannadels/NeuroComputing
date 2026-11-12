"""scripts/credit_diagnosis.py: credit variants, removal test and summary table."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import credit_diagnosis as cd  # noqa: E402
from neuron_games.config import NGRunConfig  # noqa: E402
from neuron_games.model import build_net  # noqa: E402


def _setup(readout: str = "leaky", layers: int = 2):
    torch.manual_seed(0)
    cfg = NGRunConfig.for_condition("bp_anchor", n_in=12, n_hidden=20, n_out=4, readout=readout,
                                    num_steps=6, encoding="direct", n_hidden_layers=layers)
    net = build_net(cfg)
    with torch.no_grad():
        for layer in net.layers:
            layer.fc.weight.mul_(3.0)
    loader = [(torch.rand(9, 12), torch.randint(0, 4, (9,))) for _ in range(2)]
    return cfg, net, loader


def test_credit_variants_shapes_and_values() -> None:
    cfg, net, loader = _setup()
    credits = cd.credit_variants(net, loader, cfg, torch.device("cpu"))
    assert set(credits) == set(cd.VARIANTS)
    for c in credits.values():
        assert c.shape == (20,)
        assert torch.all(c >= 0)
    # Activity is the mean spike count per example, bounded by the number of steps.
    assert torch.all(credits["activity"] <= cfg.num_steps)


def test_credit_variants_none_without_leaky_readout() -> None:
    cfg, net, loader = _setup(readout="spike")
    assert cd.credit_variants(net, loader, cfg, torch.device("cpu")) is None
    assert cd.diagnose(net, loader, cfg, torch.device("cpu")) is None


def test_diagnose_keys_and_summary() -> None:
    cfg, net, loader = _setup()
    out = cd.diagnose(net, loader, cfg, torch.device("cpu"), n_random=3)
    assert out["layer"] == len(net.layers) - 2
    assert out["n_neurons"] == 20
    assert out["k"] == [1, 2, 5]
    assert len(out["random_drop"]) == 3
    assert set(out["spearman"]) == {"act_x_signal~activity", "act_x_signal~signal", "activity~signal"}
    for v in cd.VARIANTS:
        rec = out["variants"][v]
        assert len(rec["top_drop"]) == len(rec["margin"]) == 3
        for t, r, m in zip(rec["top_drop"], out["random_drop"], rec["margin"]):
            assert abs((t - r["mean"]) - m) < 1e-9

    record = {"run_dir": "x", "dataset": "toy", "n_classes": 4, "model": "bp_anchor", **out}
    table = cd.summarize([record, {"run_dir": "y", "skipped": "readout is not leaky"}])
    assert "| toy | 4 | bp_anchor | 1 |" in table
    assert "Skipped 1 run(s)" in table


def test_limit_and_expand(tmp_path) -> None:
    loader = [1, 2, 3]
    assert cd._limit(loader, 0) is loader
    assert cd._limit(loader, 2) == [1, 2]
    (tmp_path / "a.pt").write_text("")
    (tmp_path / "b.pt").write_text("")
    got = cd._expand([str(tmp_path / "*.pt"), str(tmp_path / "a.pt")])
    assert [p.name for p in got] == ["a.pt", "b.pt"]
