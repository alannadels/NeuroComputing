"""Encoder-decoder Transformer for translation (Vaswani et al. 2017, "Attention Is All You Need").

Token embeddings (scaled by sqrt(dim)) + sinusoidal position encoding ->
torch.nn.Transformer (pre-norm encoder and decoder stacks) -> linear output
layer over the target vocabulary. Consumes the batches of
common/text_data.py:

    logits = model(batch["src"], batch["tgt_in"], batch["src_pad_mask"], batch["tgt_pad_mask"])
    # (B, T, tgt_vocab), compare with batch["tgt_out"]

greedy_decode() generates translations one token at a time for BLEU.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from common.text_data import BOS_ID, EOS_ID, PAD_ID


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        dim: int = 256,
        heads: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 3,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.dim = dim
        self.src_emb = nn.Embedding(src_vocab, dim, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(tgt_vocab, dim, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(dim, max_len)
        self.drop = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=dim, nhead=heads, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True, norm_first=True,
        )
        self.out = nn.Linear(dim, tgt_vocab)

    def _embed(self, emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pos(emb(ids) * math.sqrt(self.dim)))

    @staticmethod
    def causal_mask(size: int, device) -> torch.Tensor:
        # True above the diagonal = position may not attend to later tokens.
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), 1)

    def encode(self, src, src_pad_mask):
        return self.transformer.encoder(self._embed(self.src_emb, src), src_key_padding_mask=src_pad_mask)

    def decode(self, tgt_in, memory, src_pad_mask, tgt_pad_mask=None):
        h = self.transformer.decoder(
            self._embed(self.tgt_emb, tgt_in), memory,
            tgt_mask=self.causal_mask(tgt_in.size(1), tgt_in.device),
            tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask,
        )
        return self.out(h)

    def forward(self, src, tgt_in, src_pad_mask, tgt_pad_mask):
        return self.decode(tgt_in, self.encode(src, src_pad_mask), src_pad_mask, tgt_pad_mask)

    @torch.no_grad()
    def greedy_decode(self, src, src_pad_mask, extra_len: int = 50) -> torch.Tensor:
        """Returns (B, L) generated ids (no <bos>); each row stops after its first <eos>."""
        memory = self.encode(src, src_pad_mask)
        max_steps = src.size(1) + extra_len
        ys = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_steps):
            next_tok = self.decode(ys, memory, src_pad_mask)[:, -1].argmax(-1)
            next_tok = next_tok.masked_fill(done, PAD_ID)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            done |= next_tok == EOS_ID
            if done.all():
                break
        return ys[:, 1:]
