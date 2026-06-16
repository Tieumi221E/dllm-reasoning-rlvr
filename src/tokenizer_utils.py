"""Thin wrapper around tokenizers.Tokenizer exposing an AutoTokenizer-compatible interface."""

from __future__ import annotations
import os
from tokenizers import Tokenizer as _Tokenizer


class KGTokenizer:
    SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]"]

    def __init__(self, tok: _Tokenizer) -> None:
        self._tok = tok
        self.pad_token_id  = tok.token_to_id("[PAD]")
        self.bos_token_id  = tok.token_to_id("[BOS]")
        self.eos_token_id  = tok.token_to_id("[EOS]")
        self.mask_token_id = tok.token_to_id("[MASK]")
        self.unk_token_id  = tok.token_to_id("[UNK]")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special = {self.pad_token_id, self.bos_token_id, self.eos_token_id,
                       self.mask_token_id, self.unk_token_id}
            ids = [i for i in ids if i not in special]
        # tokenizer.json has decoder=null, so the library joins token strings
        # with spaces and leaves Ġ (U+0120, GPT-2 BPE space prefix) as-is.
        # Fix: join raw token strings directly (no spaces), then replace Ġ→space.
        tokens = [self._tok.id_to_token(i) for i in ids]
        text = ''.join(t for t in tokens if t is not None)
        return text.replace('Ġ', ' ').strip()

    def get_vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def save_pretrained(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._tok.save(os.path.join(directory, "tokenizer.json"))

    @classmethod
    def from_file(cls, path: str) -> "KGTokenizer":
        return cls(_Tokenizer.from_file(path))
