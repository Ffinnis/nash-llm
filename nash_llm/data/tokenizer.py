from __future__ import annotations

import tiktoken


BYTE_EOS = 256
BYTE_PAD = 257
BYTE_BOS = 258
BYTE_VOCAB_SIZE = 259


class Tokenizer:
    def __init__(self, representation: str = "bytes", encoding_name: str = "gpt2"):
        self.representation = representation
        self.encoding_name = encoding_name
        self._enc = None
        if representation == "tiktoken":
            self._enc = tiktoken.get_encoding(encoding_name)
        elif representation != "bytes":
            raise ValueError(
                f"Unsupported tokenizer representation '{representation}'. "
                "Expected one of: bytes, tiktoken"
            )

    @property
    def vocab_size(self) -> int:
        if self.representation == "bytes":
            return BYTE_VOCAB_SIZE
        assert self._enc is not None
        return self._enc.n_vocab

    @property
    def eot_token(self) -> int:
        if self.representation == "bytes":
            return BYTE_EOS
        assert self._enc is not None
        return self._enc.eot_token

    @property
    def pad_token(self) -> int:
        if self.representation == "bytes":
            return BYTE_PAD
        return 0

    @property
    def bos_token(self) -> int:
        if self.representation == "bytes":
            return BYTE_BOS
        return self.eot_token

    def encode(self, text: str) -> list[int]:
        if self.representation == "bytes":
            return list(text.encode("utf-8"))
        assert self._enc is not None
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list[int]) -> str:
        if self.representation == "bytes":
            byte_values = bytes(token for token in tokens if 0 <= token < 256)
            return byte_values.decode("utf-8", errors="replace")
        assert self._enc is not None
        return self._enc.decode(tokens)

    def metadata(self) -> dict[str, int | str | None]:
        return {
            "representation": self.representation,
            "encoding_name": self.encoding_name if self.representation == "tiktoken" else None,
            "vocab_size": self.vocab_size,
            "eos_token": self.eot_token,
            "bos_token": self.bos_token,
            "pad_token": self.pad_token,
        }
