# -*- coding: utf-8 -*-
"""bert.py 逐行精读版（与原文件等价，仅为阅读加了中文注释与形状标注）

记号约定（全文统一）:
  B = batch size（一篇论文一条序列）
  L = 序列长度（token 数，含 [CLS]/[SEP]/[PAD]，上限 512）
  H = hidden size = 768
  A = attention heads = 12
  D = 每个头的维度 = H/A = 64
  F = FFN 中间维度 = 3072
  V = 词表大小 = 31090（scivocab）

BERT-base 每层的数据流（先记住这张图，代码就是它的逐行翻译）:

  输入 ids (B,L)
    -> Embedding 三表相加 + LayerNorm          (B,L,H)
    -> 循环 12 次 BertLayer:
         自注意力: q,k,v 投影 -> 打分 -> softmax -> 加权求和 (B,L,H)
         残差+LN:  LN(dropout(注意力输出) + 输入)             (B,L,H)
         FFN:     H->F (gelu) -> H                            (B,L,H)
         残差+LN:  LN(dropout(FFN输出) + 输入)                (B,L,H)
    -> pooler: 取 [CLS] 位置, 过一层 Linear+tanh              (B,H)

关键数学（对照论文 1810.04805 / SciBERT 1910.06064）:
  Attention(Q,K,V) = softmax(QK^T / sqrt(D)) V
  LayerNorm(x)     = gamma * (x - mean) / sqrt(var + eps) + beta
  GELU(x)          = x * Phi(x), Phi 为标准正态 CDF（用 erf 精确实现）
"""
import math

import jittor as jt
from jittor import Module, nn


def gelu(x):
    """精确版 GELU（基于误差函数 erf），与 HF 的 hidden_act='gelu' 完全一致。

    SciBERT checkpoint 的 config 里写的就是 gelu（精确版），
    若用 tanh 近似版，每层会有 1e-3 量级偏差 —— M1 对齐就会失败。
    """
    return 0.5 * x * (1.0 + jt.erf(x / math.sqrt(2.0)))


class BertEmbeddings(Module):
    """把 token id 序列变成初始隐状态 (B,L,H)。

    三个嵌入表相加（这是 BERT 的输入表示）:
      word_embeddings     词本身是谁          (V, H)
      position_embeddings 它在序列第几个位置  (512, H)
      token_type_embeddings 它属于句子A还是B  (2, H)
    然后 LayerNorm + dropout。
    """
    def __init__(self, vocab_size, hidden=768, max_position=512, type_vocab=2, eps=1e-12):
        super().__init__()
        # nn.Embedding(V,H) 本质是一张 (V,H) 的查表矩阵: embedding(ids) 按 id 取行
        self.word_embeddings = nn.Embedding(vocab_size, hidden)
        # 位置上限 512: 所以 MAX_LEN=512 是硬约束, 超长没有位置向量可用
        self.position_embeddings = nn.Embedding(max_position, hidden)
        # 句子对任务用: 标题段=0, 摘要段=1
        self.token_type_embeddings = nn.Embedding(type_vocab, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)
        self.dropout = nn.Dropout(0.1)

    def execute(self, input_ids, token_type_ids=None):
        # 没有句子对时全当句子A
        if token_type_ids is None:
            token_type_ids = jt.zeros_like(input_ids)
        b, l = input_ids.shape
        # 位置 id = [0,1,2,...,L-1], 对 batch 内每条序列都一样
        # unsqueeze(0) 让 (L,) -> (1,L); 与 (B,1) 的零相加是广播技巧的写法,
        # 目的是得到 (B,L) 的形状与 ids 对齐
        pos = jt.arange(l, dtype='int32').unsqueeze(0) + jt.zeros((b, 1), dtype='int32')
        # 三个查表结果都是 (B,L,H), 直接逐元素相加
        x = self.word_embeddings(input_ids) + self.position_embeddings(pos) \
            + self.token_type_embeddings(token_type_ids)
        # LayerNorm 把每个 token 的 768 维向量标准化, 稳定后续计算
        return self.dropout(self.LayerNorm(x))


class BertSelfAttention(Module):
    """多头自注意力: 让每个 token"看"序列里所有 token, 汇总信息。

    直觉: 对 token i, 它的输出 = 全序列 token 的 value 向量的加权和,
    权重由"我的 query 与你的 key 有多匹配"决定。
    12 个头各看 64 维子空间, 学不同类型的关系(指代/共现/句法...)。
    """
    def __init__(self, hidden=768, heads=12):
        super().__init__()
        self.num_heads = heads                 # A = 12
        self.head_dim = hidden // heads        # D = 64
        # 三个 [out,in]=[H,H] 的线性层: 把同一个输入分别变成 q/k/v
        # （注意: 每个投影内部已含 12 个头的划分, 后面 reshape 拆开）
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(0.1)

    def split_heads(self, x):
        """(B,L,H) -> (B,A,L,D): 把 768 维切成 12 个 64 维的头。

        reshape(B,L,12,64) 先按头分组, permute 把头维提到序列维前面,
        这样每个头独立做自己的 (L,L) 注意力。
        """
        b, l, _ = x.shape
        return x.reshape(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def execute(self, hidden_states, attention_mask):
        # 三步投影 + 分头:  各 (B,A,L,D)
        q = self.split_heads(self.query(hidden_states))
        k = self.split_heads(self.key(hidden_states))
        v = self.split_heads(self.value(hidden_states))
        # 注意力打分: (B,A,L,D) @ (B,A,D,L) -> (B,A,L,L)
        # scores[b,h,i,j] = 第 h 个头里, token i 对 token j 的关注程度(未归一化)
        # 除以 sqrt(D)=8: 防止内积过大把 softmax 推进梯度极小的饱和区
        scores = (q @ k.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        # attention_mask 形状 (B,1,1,L): 真实 token 处为 0, [PAD] 处为 -10000
        # 加进 scores 后, softmax 会把 PAD 列的概率压到 ~0 —— 这就是掩码的实现
        scores = scores + attention_mask
        # 沿最后一维(L)归一化: 每个 token 对所有 token 的注意力概率和为 1
        probs = self.dropout(nn.softmax(scores, dim=-1))
        # 加权求和: (B,A,L,L) @ (B,A,L,D) -> (B,A,L,D)
        ctx = probs @ v
        # 合并头: (B,A,L,D) -> (B,L,A,D) -> (B,L,H), 回到标准形状
        b, h, l, d = ctx.shape
        return ctx.permute(0, 2, 1, 3).reshape(b, l, h * d)


class BertAttention(Module):
    """自注意力 + 输出投影 + 残差连接 + LayerNorm（一个完整注意力子层）。

    残差: 输出 = LN(dropout(投影(注意力结果)) + 输入)
    残差让深层网络可训练(梯度有捷径), LN 稳定每层数值范围。
    """
    def __init__(self, hidden=768, heads=12):
        super().__init__()
        self.self_attn = BertSelfAttention(hidden, heads)
        self.output = nn.Linear(hidden, hidden)   # 多头拼接后的混合投影
        self.LayerNorm = nn.LayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def execute(self, hidden_states, attention_mask):
        ctx = self.self_attn(hidden_states, attention_mask)
        return self.LayerNorm(self.dropout(self.output(ctx)) + hidden_states)


class BertLayer(Module):
    """一个完整 Transformer 层 = 注意力子层 + FFN 子层(各带残差+LN)。

    FFN: H -> F(gelu) -> H, 即 768 -> 3072 -> 768
    注意力负责"token 之间交换信息", FFN 负责"每个 token 独立地变换特征",
    两者交替 12 次, 信息在序列内充分混合。
    """
    def __init__(self, hidden=768, heads=12, intermediate=3072):
        super().__init__()
        self.attention = BertAttention(hidden, heads)
        self.intermediate = nn.Linear(hidden, intermediate)   # (F,H): 升维
        self.output = nn.Linear(intermediate, hidden)         # (H,F): 降维
        self.LayerNorm = nn.LayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def execute(self, hidden_states, attention_mask):
        h = self.attention(hidden_states, attention_mask)     # 子层1: 注意力
        inter = gelu(self.intermediate(h))                    # FFN 前半: (B,L,F)
        # 子层2: FFN 输出 + 残差 + LN
        return self.LayerNorm(self.dropout(self.output(inter)) + h)


class BertModel(Module):
    """完整 BERT-base 编码器。

    execute 返回:
      last_hidden (B,L,H): 最后一层每个 token 的向量
      pooled      (B,H):   [CLS] 过 Linear+tanh 的"句子级"向量
      (return_hidden=True 时额外返回 hiddens: 13 层隐状态列表, 供对齐验证用)
    """
    def __init__(self, vocab_size=31116, hidden=768, layers=12, heads=12,
                 intermediate=3072, max_position=512):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden, max_position)
        # ModuleList 保证每层的参数都被注册, 可以被 save/load
        self.layers = nn.ModuleList([BertLayer(hidden, heads, intermediate)
                                     for _ in range(layers)])
        self.pooler = nn.Linear(hidden, hidden)

    def execute(self, input_ids, attention_mask=None, token_type_ids=None,
                return_hidden=False):
        # 默认所有位置都是真实 token (不掩任何位置)
        if attention_mask is None:
            attention_mask = jt.ones_like(input_ids).float32()
        # 构造"加性掩码": 真实 token=0, PAD=-10000
        # 形状 (B,1,1,L): 留出两个 1, 广播到注意力的 (B,A,L,L) 打分矩阵上
        mask = (1.0 - attention_mask.float32()) * -10000.0
        mask = mask.unsqueeze(1).unsqueeze(2)

        h = self.embeddings(input_ids, token_type_ids)   # (B,L,H) 第 0 层
        hiddens = [h]
        for layer in self.layers:                        # 12 层循环
            h = layer(h, mask)
            hiddens.append(h)

        # 取 [CLS](位置0)的向量做句子级表示: (B,H) -> Linear -> tanh
        # pooler 权重是 NSP 任务训练出来的; 下游任务一般用 last_hidden[:,0]
        pooled = jt.tanh(self.pooler(h[:, 0]))
        if return_hidden:
            return h, pooled, hiddens
        return h, pooled

# ============ 精读检查清单 ============
# 读完后你应该能不看代码回答:
# 1. 一个 (B,L) 的 ids 进去, 每个张量流经 Embedding/单头注意力/合并头/FFN 时
#    的形状分别是什么? (B,L,H) -> (B,A,L,D) -> (B,A,L,L) -> (B,A,L,D)
#    -> (B,L,H) -> (B,L,F) -> (B,L,H)
# 2. 掩码为什么用 -10000 而不是 0? 掩码的形状为什么留两个 1?
# 3. 除以 sqrt(64)=8 防的是什么?
# 4. 残差+LN 出现在哪两个地方? 为什么 M1 对齐对 LayerNorm 的 eps 敏感?
# 5. pooler 和 [CLS] 隐状态有什么区别? 我们编码论文用的是哪个?
#    (答案: 用 last_hidden[:,0], 即最后一层 [CLS] 位置, 不过 pooler)
