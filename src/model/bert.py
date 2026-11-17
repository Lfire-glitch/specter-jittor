"""Minimal BERT-base encoder in Jittor, mirroring HuggingFace `bert-base-uncased`
layout so that weights can be copied 1:1 (Jittor nn.Linear is [out, in], same as torch).

Reference: BERT (1810.04805); architecture config from allenai/specter config.json
(hidden 768, 12 layers, 12 heads, intermediate 3072, gelu exact, layer_norm_eps 1e-12).
"""
import math

import jittor as jt
from jittor import Module, nn


def gelu(x):
    """Exact GELU (erf based), matching HF hidden_act='gelu'."""
    return 0.5 * x * (1.0 + jt.erf(x / math.sqrt(2.0)))


class BertEmbeddings(Module):
    def __init__(self, vocab_size, hidden=768, max_position=512, type_vocab=2, eps=1e-12):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden)
        self.position_embeddings = nn.Embedding(max_position, hidden)
        self.token_type_embeddings = nn.Embedding(type_vocab, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)
        self.dropout = nn.Dropout(0.1)

    def execute(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = jt.zeros_like(input_ids)
        b, l = input_ids.shape
        pos = jt.arange(l, dtype='int32').unsqueeze(0) + jt.zeros((b, 1), dtype='int32')
        x = self.word_embeddings(input_ids) + self.position_embeddings(pos)+ self.token_type_embeddings(token_type_ids)
        return self.dropout(self.LayerNorm(x))


class BertSelfAttention(Module):
    def __init__(self, hidden=768, heads=12):
        super().__init__()
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(0.1)

    def split_heads(self, x):
        b, l, _ = x.shape
        return x.reshape(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def execute(self, hidden_states, attention_mask):
        q = self.split_heads(self.query(hidden_states))
        k = self.split_heads(self.key(hidden_states))
        v = self.split_heads(self.value(hidden_states))
        scores = (q @ k.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        scores = scores + attention_mask
        probs = self.dropout(nn.softmax(scores, dim=-1))
        ctx = probs @ v
        b, h, l, d = ctx.shape
        return ctx.permute(0, 2, 1, 3).reshape(b, l, h * d)


class BertAttention(Module):
    def __init__(self, hidden=768, heads=12):
        super().__init__()
        self.self_attn = BertSelfAttention(hidden, heads)
        self.output = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def execute(self, hidden_states, attention_mask):
        ctx = self.self_attn(hidden_states, attention_mask)
        return self.LayerNorm(self.dropout(self.output(ctx)) + hidden_states)


class BertLayer(Module):
    def __init__(self, hidden=768, heads=12, intermediate=3072):
        super().__init__()
        self.attention = BertAttention(hidden, heads)
        self.intermediate = nn.Linear(hidden, intermediate)
        self.output = nn.Linear(intermediate, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def execute(self, hidden_states, attention_mask):
        h = self.attention(hidden_states, attention_mask)
        inter = gelu(self.intermediate(h))
        return self.LayerNorm(self.dropout(self.output(inter)) + h)


class BertModel(Module):
    def __init__(self, vocab_size=31116, hidden=768, layers=12, heads=12,
                 intermediate=3072, max_position=512):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden, max_position)
        self.layers = nn.ModuleList([BertLayer(hidden, heads, intermediate) for _ in range(layers)])
        self.pooler = nn.Linear(hidden, hidden)

    def execute(self, input_ids, attention_mask=None, token_type_ids=None, return_hidden=False):
        if attention_mask is None:
            attention_mask = jt.ones_like(input_ids).float32()
        mask = (1.0 - attention_mask.float32()) * -10000.0
        mask = mask.unsqueeze(1).unsqueeze(2)
        h = self.embeddings(input_ids, token_type_ids)
        hiddens = [h]
        for layer in self.layers:
            h = layer(h, mask)
            hiddens.append(h)
        pooled = jt.tanh(self.pooler(h[:, 0]))
        if return_hidden:
            return h, pooled, hiddens
        return h, pooled
