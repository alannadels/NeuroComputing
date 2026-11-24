"""Multi30k translation loader for sequence-to-sequence Transformer training.

Multi30k (task 1, Flickr30k image descriptions) pairs English and German
sentences: 29,000 train / 1,014 validation / 1,000 test (test_2016_flickr).
The raw gzip files come from the official multi30k GitHub repository and are
cached once in <data_dir>/multi30k/ (see prefetch_data.py), so offline GPU
jobs can run with download=False.

Default direction is English -> German, matching the transformer experiment
plan. Tokenization is word level (lowercase, words and punctuation split);
each vocabulary is built from the training split only, keeping tokens seen at
least `min_freq` times. Special token ids are fixed:
    <pad>=0, <unk>=1, <bos>=2, <eos>=3

Each batch is a dict:
    src           (B, S) long   source ids followed by <eos>, right padded
    tgt_in        (B, T) long   <bos> + target ids  (decoder input, teacher forcing)
    tgt_out       (B, T) long   target ids + <eos>  (labels; ignore_index=PAD_ID)
    src_pad_mask  (B, S) bool   True at padding (nn.Transformer key_padding_mask)
    tgt_pad_mask  (B, T) bool   True at padding
    index         (B,)   long   position of the pair in its split (for BLEU references)
"""
from __future__ import annotations

import gzip
import re
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


MULTI30K_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/{name}.{lang}.gz"
MULTI30K_SPLITS = {"train": "train", "val": "val", "test": "test_2016_flickr"}
MULTI30K_SIZES = {"train": 29000, "val": 1014, "test": 1000}
MULTI30K_LANGS = ("en", "de")

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = (PAD, UNK, BOS, EOS)
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(4)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenizer: words (Unicode aware, so umlauts stay) and single punctuation marks."""
    return _TOKEN_RE.findall(text.lower())


def download_multi30k(data_dir: str | Path, download: bool = True) -> Path:
    """Make sure <data_dir>/multi30k/{split}.{lang} exist (plain text, one sentence per line)."""
    root = Path(data_dir) / "multi30k"
    root.mkdir(parents=True, exist_ok=True)
    for name in MULTI30K_SPLITS.values():
        for lang in MULTI30K_LANGS:
            path = root / f"{name}.{lang}"
            if path.is_file():
                continue
            if not download:
                raise FileNotFoundError(
                    f"{path} not found and download=False; run prefetch_data.py on a node with internet")
            url = MULTI30K_URL.format(name=name, lang=lang)
            with urllib.request.urlopen(url, timeout=60) as resp:
                text = gzip.decompress(resp.read()).decode("utf-8")
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
    return root


def read_split(root: Path, split: str, src_lang: str, tgt_lang: str) -> list[tuple[str, str]]:
    name = MULTI30K_SPLITS[split]
    src = (root / f"{name}.{src_lang}").read_text(encoding="utf-8").splitlines()
    tgt = (root / f"{name}.{tgt_lang}").read_text(encoding="utf-8").splitlines()
    if len(src) != len(tgt):
        raise ValueError(f"{split}: {len(src)} {src_lang} lines but {len(tgt)} {tgt_lang} lines")
    if len(src) != MULTI30K_SIZES[split]:
        raise ValueError(f"{split}: expected {MULTI30K_SIZES[split]} pairs, found {len(src)}")
    return [(s.strip(), t.strip()) for s, t in zip(src, tgt)]


class Vocab:
    """Token <-> id map with the four special tokens at ids 0..3."""

    def __init__(self, tokens: list[str]):
        self.itos = list(SPECIALS) + [t for t in tokens if t not in SPECIALS]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, sentences: list[list[str]], min_freq: int = 2, max_size: int | None = None) -> "Vocab":
        counts = Counter(tok for sent in sentences for tok in sent)
        # Most frequent first, ties broken alphabetically so the vocabulary is deterministic.
        ranked = sorted((t for t, c in counts.items() if c >= min_freq), key=lambda t: (-counts[t], t))
        if max_size is not None:
            ranked = ranked[:max(0, max_size - len(SPECIALS))]
        return cls(ranked)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids, strip_specials: bool = True) -> list[str]:
        """Ids -> tokens, stopping at the first <eos> when strip_specials is set."""
        out = []
        for i in (ids.tolist() if torch.is_tensor(ids) else ids):
            if strip_specials:
                if i == EOS_ID:
                    break
                if i in (PAD_ID, BOS_ID):
                    continue
            out.append(self.itos[i])
        return out


class TranslationDataset(Dataset):
    """Pre-tokenized (source ids, target ids) pairs; raw sentences kept in `.pairs` for BLEU."""

    def __init__(self, pairs: list[tuple[str, str]], src_vocab: Vocab, tgt_vocab: Vocab,
                 max_len: int | None = None):
        self.pairs = pairs
        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
        self.examples = []
        for src, tgt in pairs:
            s, t = src_vocab.encode(tokenize(src)), tgt_vocab.encode(tokenize(tgt))
            if max_len is not None:  # leave room for <eos>/<bos> so batches never exceed max_len
                s, t = s[:max_len - 1], t[:max_len - 1]
            self.examples.append((torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        src, tgt = self.examples[i]
        return src, tgt, i

    def references(self) -> list[str]:
        """Target sentences in the model's own token space (lowercased, punctuation and hyphens split).

        Score with sacrebleu.corpus_bleu(hyps, [refs], tokenize="none"), where each hypothesis is
        " ".join(tgt_vocab.decode(ids)). Raw references would be deflated by casing, since the
        model only ever produces lowercase tokens.
        """
        return [" ".join(tokenize(t)) for _, t in self.pairs]


def _pad(seqs: list[torch.Tensor]) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)


def collate_translation(batch) -> dict[str, torch.Tensor]:
    srcs, tgts, idx = zip(*batch)
    eos, bos = torch.tensor([EOS_ID]), torch.tensor([BOS_ID])
    src = _pad([torch.cat([s, eos]) for s in srcs])
    tgt_in = _pad([torch.cat([bos, t]) for t in tgts])
    tgt_out = _pad([torch.cat([t, eos]) for t in tgts])
    return {
        "src": src, "tgt_in": tgt_in, "tgt_out": tgt_out,
        "src_pad_mask": src == PAD_ID, "tgt_pad_mask": tgt_out == PAD_ID,
        "index": torch.tensor(idx, dtype=torch.long),
    }


def build_multi30k_loaders(
    data_dir: str | Path,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    download: bool = True,
    src_lang: str = "en",
    tgt_lang: str = "de",
    min_freq: int = 2,
    max_vocab: int | None = None,
    max_len: int | None = None,
) -> tuple[dict[str, DataLoader], dict[str, Vocab]]:
    """Returns ({"train", "val", "test"} loaders, {"src", "tgt"} vocabularies).

    Multi30k ships its own validation split, so there is no val_size split here.
    max_len, when set, caps training sequence length including <bos>/<eos>.
    """
    if {src_lang, tgt_lang} != set(MULTI30K_LANGS):
        raise ValueError(f"src_lang/tgt_lang must be one each of {MULTI30K_LANGS}")
    if max_len is not None and max_len < 2:
        raise ValueError("max_len must be >= 2 (room for <bos>/<eos> plus one token)")
    root = download_multi30k(data_dir, download=download)
    pairs = {split: read_split(root, split, src_lang, tgt_lang) for split in MULTI30K_SPLITS}

    train_src = [tokenize(s) for s, _ in pairs["train"]]
    train_tgt = [tokenize(t) for _, t in pairs["train"]]
    vocabs = {"src": Vocab.build(train_src, min_freq, max_vocab),
              "tgt": Vocab.build(train_tgt, min_freq, max_vocab)}
    # max_len truncates training pairs only, so validation and test inputs stay complete.
    sets = {split: TranslationDataset(p, vocabs["src"], vocabs["tgt"], max_len if split == "train" else None)
            for split, p in pairs.items()}

    # Same convention as common/data.py: shuffling RNG separated from the global seed.
    train_gen = torch.Generator().manual_seed(seed + 1)
    common = dict(collate_fn=collate_translation, num_workers=num_workers, pin_memory=pin_memory)
    loaders = {
        "train": DataLoader(sets["train"], batch_size=batch_size, shuffle=True, drop_last=True,
                            generator=train_gen, **common),
        "val": DataLoader(sets["val"], batch_size=eval_batch_size, shuffle=False, drop_last=False, **common),
        "test": DataLoader(sets["test"], batch_size=eval_batch_size, shuffle=False, drop_last=False, **common),
    }
    return loaders, vocabs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download Multi30k and print loader statistics.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    loaders, vocabs = build_multi30k_loaders(args.data_dir, batch_size=128, eval_batch_size=256,
                                             seed=20260923, download=not args.no_download, pin_memory=False)
    print(f"vocab sizes: src={len(vocabs['src'])} tgt={len(vocabs['tgt'])}")
    for split, loader in loaders.items():
        ds = loader.dataset
        lens = [len(t) for _, t in ds.examples]
        unk = sum(int((t == UNK_ID).sum()) for _, t in ds.examples) / max(1, sum(lens))
        print(f"{split}: {len(ds)} pairs, {len(loader)} batches, mean target length "
              f"{sum(lens) / len(lens):.1f}, target unknown-token rate {unk:.2%}")
    batch = next(iter(loaders["train"]))
    print({k: tuple(v.shape) for k, v in batch.items()})
    i = int(batch["index"][0])
    print("source:", loaders["train"].dataset.pairs[i][0])
    print("decoded target:", " ".join(vocabs["tgt"].decode(batch["tgt_out"][0])))
