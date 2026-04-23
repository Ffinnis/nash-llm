import torch
import pytest
from nash_llm.model.attention import MultiHeadAttention
from nash_llm.config import ModelConfig


class TestMultiHeadAttention:
    def setup_method(self):
        self.cfg = ModelConfig(d_model=64, n_heads=4, max_seq_len=32, dropout=0.0)
        self.attn = MultiHeadAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(2, 16, 64)
        out = self.attn(x)
        assert out.shape == (2, 16, 64)

    def test_causal_mask(self):
        torch.manual_seed(42)
        self.attn.eval()
        x = torch.randn(1, 8, 64)
        out_full = self.attn(x)

        x_modified = x.clone()
        x_modified[0, 7, :] = torch.randn(64)
        out_modified = self.attn(x_modified)

        # Positions 0-6 should be identical (causal = no future leaking)
        assert torch.allclose(out_full[0, :7], out_modified[0, :7], atol=1e-6)

    def test_head_dim_consistency(self):
        bad_cfg = ModelConfig(d_model=65, n_heads=4)
        with pytest.raises(ValueError):
            MultiHeadAttention(bad_cfg)

    def test_grouped_query_attention_output_shape(self):
        attn = MultiHeadAttention(ModelConfig(d_model=64, n_heads=4, n_kv_heads=2, max_seq_len=32, dropout=0.0))
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_grouped_query_attention_projection_shapes(self):
        attn = MultiHeadAttention(ModelConfig(d_model=64, n_heads=4, n_kv_heads=2, max_seq_len=32, dropout=0.0))
        assert attn.q_proj.weight.shape == (64, 64)
        assert attn.k_proj.weight.shape == (32, 64)
        assert attn.v_proj.weight.shape == (32, 64)

    def test_grouped_query_attention_requires_divisible_heads(self):
        with pytest.raises(ValueError, match="must divide model.n_heads"):
            ModelConfig(d_model=64, n_heads=6, n_kv_heads=4)

    def test_rope_buffers_created_by_default(self):
        assert self.attn.position_embedding == "rope"
        assert self.attn.rope_cos.shape == (32, 8)
        assert self.attn.rope_sin.shape == (32, 8)

    def test_rope_requires_even_head_dim(self):
        bad_cfg = ModelConfig(d_model=60, n_heads=4, position_embedding="rope")
        with pytest.raises(ValueError, match="even head_dim"):
            MultiHeadAttention(bad_cfg)

    def test_learned_position_embedding_skips_rope_buffers(self):
        attn = MultiHeadAttention(ModelConfig(d_model=64, n_heads=4, max_seq_len=32, position_embedding="learned"))
        assert not hasattr(attn, "rope_cos")
        assert not hasattr(attn, "rope_sin")
