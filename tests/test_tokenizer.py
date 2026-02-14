from nash_llm.data.tokenizer import Tokenizer


class TestTokenizer:
    def setup_method(self):
        self.tok = Tokenizer()

    def test_encode_returns_list_of_ints(self):
        ids = self.tok.encode("Hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_decode_roundtrip(self):
        text = "The quick brown fox jumps over the lazy dog."
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_vocab_size(self):
        assert self.tok.vocab_size == 50257

    def test_eot_token(self):
        assert isinstance(self.tok.eot_token, int)
