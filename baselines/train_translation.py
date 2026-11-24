"""Train the encoder-decoder Transformer on Multi30k (English -> German by default).

    python train_translation.py --epochs 20 --no-download

Recipe: Adam (betas 0.9/0.98), linear warmup then inverse square root decay,
label smoothing 0.1, gradient clipping 1.0. The checkpoint with the lowest
validation loss is kept, reloaded, and used for greedy decoding on the test
split. BLEU is tokenized and lowercased (see common/text_data.py); it uses
sacrebleu when installed and a built-in corpus BLEU-4 otherwise.

Writes results/multi30k/seq2seq_transformer/<tag>/{config.json, results.json,
best.pt, samples.txt}.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from common.text_data import PAD_ID, build_multi30k_loaders
from models.seq2seq_transformer import Seq2SeqTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--src-lang", default="en")
    p.add_argument("--tgt-lang", default="de")
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-len", type=int, default=None, help="cap training sequence length")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None, help="stop early after this many steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4, help="peak learning rate after warmup")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--ff-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--tag", default=None, help="run folder name (default: timestamp + seed)")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def corpus_bleu(hyps: list[str], refs: list[str]) -> float:
    """Corpus BLEU-4 on pre-tokenized strings (sacrebleu tokenize="none" when available)."""
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none").score
    except ImportError:
        pass
    match, total = [0] * 4, [0] * 4
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        h, r = h.split(), r.split()
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, 5):
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            match[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            total[n - 1] += max(0, len(h) - n + 1)
    if hyp_len == 0 or min(match) == 0:
        return 0.0
    log_prec = sum(math.log(m / t) for m, t in zip(match, total)) / 4
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100 * bp * math.exp(log_prec)


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None, clip_grad=None,
              max_steps=None, step=0) -> tuple[float, float, int]:
    """One pass; trains when optimizer is given. Returns (loss per token, token accuracy, step)."""
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, tokens = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            if train and max_steps is not None and step >= max_steps:
                break
            b = to_device(batch, device)
            logits = model(b["src"], b["tgt_in"], b["src_pad_mask"], b["tgt_pad_mask"])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), b["tgt_out"].reshape(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                scheduler.step()
                step += 1
            keep = b["tgt_out"] != PAD_ID
            n = int(keep.sum())
            loss_sum += loss.item() * n
            correct += int(((logits.argmax(-1) == b["tgt_out"]) & keep).sum())
            tokens += n
    return loss_sum / max(1, tokens), correct / max(1, tokens), step


@torch.no_grad()
def translate(model, loader, tgt_vocab, device) -> list[str]:
    """Greedy translations in dataset order, as space-joined tokens."""
    model.eval()
    hyps = [""] * len(loader.dataset)
    for batch in loader:
        b = to_device(batch, device)
        out = model.greedy_decode(b["src"], b["src_pad_mask"])
        for i, ids in zip(batch["index"].tolist(), out.cpu()):
            hyps[i] = " ".join(tgt_vocab.decode(ids))
    return hyps


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}"
    run_dir = Path(args.results_dir) / "multi30k" / "seq2seq_transformer" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, vocabs = build_multi30k_loaders(
        args.data_dir, args.batch_size, args.eval_batch_size, args.seed,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", download=not args.no_download,
        src_lang=args.src_lang, tgt_lang=args.tgt_lang, min_freq=args.min_freq, max_len=args.max_len,
    )
    model = Seq2SeqTransformer(
        len(vocabs["src"]), len(vocabs["tgt"]), dim=args.dim, heads=args.heads,
        enc_layers=args.enc_layers, dec_layers=args.dec_layers, ff_dim=args.ff_dim, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    config = {**vars(args), "device": str(device), "n_params": n_params,
              "src_vocab": len(vocabs["src"]), "tgt_vocab": len(vocabs["tgt"])}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"params={n_params} vocab src={len(vocabs['src'])} tgt={len(vocabs['tgt'])} "
          f"device={device} run_dir={run_dir}", flush=True)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=args.label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    warmup = max(1, args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, math.sqrt(warmup / (s + 1))))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_epoch, step = float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, step = run_epoch(model, loaders["train"], loss_fn, device, optimizer, scheduler,
                                          args.clip_grad, args.max_steps, step)
        va_loss, va_acc, _ = run_epoch(model, loaders["val"], eval_loss_fn, device)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)
        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), run_dir / "best.pt")
        print(f"epoch {epoch:3d} step {step:6d} train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} perplexity {math.exp(va_loss):.1f} acc {va_acc:.3f} | "
              f"{time.time() - t0:.0f}s", flush=True)
        if args.max_steps is not None and step >= args.max_steps:
            break
    train_time = time.time() - t0

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    te_loss, te_acc, _ = run_epoch(model, loaders["test"], eval_loss_fn, device)
    test_ds = loaders["test"].dataset
    hyps = translate(model, loaders["test"], vocabs["tgt"], device)
    refs = test_ds.references()
    bleu = corpus_bleu(hyps, refs)

    with open(run_dir / "samples.txt", "w", encoding="utf-8") as f:
        for i in range(min(20, len(hyps))):
            f.write(f"source:     {test_ds.pairs[i][0]}\nreference:  {refs[i]}\nhypothesis: {hyps[i]}\n\n")
    results = {
        "best_epoch": best_epoch, "best_val_loss": best_val, "test_loss": te_loss,
        "test_perplexity": math.exp(te_loss), "test_token_accuracy": te_acc, "test_bleu": bleu,
        "bleu_note": "tokenized, lowercased, greedy decoding", "train_time_s": train_time,
        "steps": step, "history": history,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"best epoch {best_epoch} | test loss {te_loss:.3f} perplexity {math.exp(te_loss):.1f} "
          f"token accuracy {te_acc:.3f} BLEU {bleu:.2f} -> {run_dir}", flush=True)
    for i in range(3):
        print(f"  source: {test_ds.pairs[i][0]}\n  reference: {refs[i]}\n  hypothesis: {hyps[i]}")


if __name__ == "__main__":
    main()
