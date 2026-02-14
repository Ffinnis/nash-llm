import tiktoken


class Tokenizer:
    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot_token(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)
