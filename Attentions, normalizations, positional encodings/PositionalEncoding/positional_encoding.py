import torch
import torch.nn as nn

'''
Rotary Position Embedding(RoPE)

Articles - RoFormer: Enhanced Transformer with Rotary Position Embedding
https://arxiv.org/pdf/2104.09864.pdf


Code from Google Gemma model HaggingFace repo
https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma
'''


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                    self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


'''
inv_freq - calculates the inverse frequencies (inv_freq) used in Rotary Positional Embedding. 
It represents the following mathematical formula:
torch.arange(0,dim,2) generates a sequence of integers starting from 0 up to dim
dim (exclusive), with a step of 2. 
This sequence represents the even indices for the dimensions of the embedding.

This sequence is then divided by dim - the embedding dimension, to scale these indices.

Formula raises the base to the power of these scaled indices. 
The base is a hyperparameter (often a large number like 10000) that controls 
the rate at which the positional encoding frequencies decrease.

Taking the inverse of this, gives the inverse frequencies. 
These are used in calculating the sine and cosine positional encodings, 
allowing the model to encode positional information through rotation effectively.

The formula effectively spreads out the frequencies across the dimensions of the embeddings 
in such a way that adjacent dimensions have slightly different frequencies, 
and this pattern repeats for all even dimensions. 
This spread helps the model to differentiate positions across the sequence length effectively.

"GPT 4.0"
'''
