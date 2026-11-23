"""CCQGen/CCExpand 技术演示共用的纯确定性协议(L1 已 Jittor 化三个算子)。

诚实边界: 只复现 2601.00567 中「概念索引如何调度生成与匹配」的纯
确定性算子, 不复现生成质量或检索增益。4 处 LLM 调用(core topic/phrase
选择、4.2 query 生成、5.1 snippet 生成)全部用 fixture/stub 替代;
概念索引是受控人造向量, 非 MAG 431416 节点 taxonomy; 不新增 retriever、
不引测试集、不跑 R@K/NDCG/MAP。定级: 接口/机制级技术演示。
L1 边界(#20): l2norm/subtree_scores/coverage_pi 的张量算子改 jt.Var,
Jittor 1.3.11.0 强制 float32, 故显式转 float32 入图、numpy 收尾重归一;
argsort / default_rng / json / npz 等仪器层仍保留 numpy 语义不改。
"""

import numpy as np
# L1: 仅三个纯算子入图; use_cuda=0 与 13 回归同 CPU 口径
import jittor as jt
jt.flags.use_cuda = 0 
# float32 入图(matmul 只支持 float32, 混 float64 会编译失败)
def _to_jt(x):
    return jt.array(np.asarray(x, dtype=np.float32))
# 出图回 numpy float64, 供 argsort/default_rng/json 等仪器层继续用
def _from_jt(v):
    return np.asarray(v.numpy(), dtype=np.float64)
# 等价 np.maximum(v, eps), eps 常量也必须 float32
def _jt_clamp_min(v, eps):
    e = jt.array(np.full(v.shape, float(eps), dtype=np.float32))
    return jt.where(v > e, v, e)
# 按行 L2 归一化(jt.Var 算子), 输出仍回 np.ndarray
def l2norm(x, eps=1e-12):
    v = _to_jt(x)
    sq = (v * v).sum(-1)
    norm = (sq ** 0.5).reshape((-1, 1))
    den = _jt_clamp_min(norm, eps)
    return _from_jt(v / den)
# 3.1 s(d,c): 子树 phrase 均值 -> jt.Var matmul -> cosine; 广播回 phrase 用 numpy
def subtree_scores(doc_emb, phrase_emb, groups):
    d = l2norm(doc_emb)
    p = l2norm(phrase_emb)
    centers = l2norm(np.stack([p[g].mean(axis=0) for g in groups]))
    a, b = _to_jt(d), _to_jt(centers.T)
    try:
        node_v = a @ b
    except Exception:
        node_v = jt.matmul(a, b) 
    node = _from_jt(node_v)
    phrase = np.zeros((d.shape[0], p.shape[0]), dtype=np.float64)
    for c, g in enumerate(groups):
        phrase[:, g] = node[:, c:c + 1]
    return node, phrase
def distinctiveness(df,n_docs):
    df=np.clip(np.asarray(df,dtype=np.float64),1.0,float(n_docs))
    return np.log(float(n_docs)/df)
def top_fraction_mask(score,frac):
    score=np.asarray(score,dtype=np.float64)
    k=int(np.ceil(frac*score.size))
    idx=np.argsort(-score)[:k]
    mask=np.zeros(score.size,dtype=bool)
    mask[idx]=True
    return mask
def truncate_distribution(pi,k_p,mask=None):
    pi=np.asarray(pi,dtype=np.float64).copy()
    if mask is not None:
        pi[~np.asarray(mask,dtype=bool)]=0.0
    if k_p is not None and k_p<pi.size:
        idx=np.argsort(-pi)[:int(k_p)]
        keep=np.zeros(pi.size,dtype=bool)
        keep[idx]=True
        pi[~keep]=0.0
    s=pi.sum()
    if s<=0.0:
        raise ValueError("truncate_distribution: all mass is zero")
    pi=pi/s
    return pi,np.flatnonzero(pi>0.0)
# 4.1 Eq.1; 末尾 numpy 重归一保证 sum 精确=1, 避免 float32 击穿 1e-12 断言
def coverage_pi(y_d, y_q, eps=1e-3):
    yd, yq = _to_jt(y_d), _to_jt(y_q)
    gap = _jt_clamp_min(yd - yq, eps)
    pi = _from_jt(gap / gap.sum())
    return pi / pi.sum()
def sample_round(pi,q,rng,blocked=None):
    pi=np.asarray(pi,dtype=np.float64).copy()
    if blocked is not None:
        pi[np.asarray(blocked,dtype=bool)]=0.0
    s=pi.sum()
    if s<=0.0:
        return np.empty(0,dtype=np.int64)
    pi=pi/s
    q=int(min(q,np.count_nonzero(pi>0.0)))
    if q<=0:
        return np.empty(0,dtype=np.int64)
    return np.asarray(rng.choice(pi.size,size=q,replace=False,p=pi),dtype=np.int64)
# 4.1 多轮闭环: 每轮 floor(k_p/M) 个, 累积 y_q 使覆盖并集单调上升
def coverage_rounds(y_d, k_p, rounds, rng, eps=1e-3, allowed=None):
    y_d = np.asarray(y_d, dtype=np.float64)
    m = int(rounds)
    q = int(k_p // m)
    if q <= 0:
        raise ValueError("k_p // rounds must be positive")
    allow = np.ones(y_d.size, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    y_q = np.zeros_like(y_d)
    used = np.zeros(y_d.size, dtype=bool)
    rec = []
    for t in range(m):
        pi = coverage_pi(y_d, y_q, eps)
        pi[~allow] = 0.0
        pi = pi / pi.sum()
        pick = sample_round(pi, q, rng, used)
        y_q[pick] = 1.0
        used[pick] = True
        rec.append({
            "round": t + 1,
            "pi_sum": float(pi.sum()),
            "picked": pick.tolist(),
            "union": int(used.sum()),
        })
    return rec, y_q, used
def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / s
# 4.3 Eq.4 融合项 z(s_text)+z(s_concept)
def fuse_scores(s_text, s_concept):
    return zscore(s_text) + zscore(s_concept)
def consistency_filter(s_text, s_concept, top_n):
    fused = fuse_scores(s_text, s_concept)
    keep = np.argsort(-fused)[:int(top_n)]
    return keep, fused
# 5.2 Eq.6 s*=argmax_s sim(q,s)
def best_snippet(sims):
    return int(np.argmax(np.asarray(sims, dtype=np.float64)))
# max 保峰值 / mean 会稀释; 对照论文表5反例机制
def snippet_evidence(sims, agg):
    sims = np.asarray(sims, dtype=np.float64)
    if agg == "max":
        return float(sims.max())
    if agg == "mean":
        return float(sims.mean())
    raise ValueError("agg must be 'max' or 'mean'")
# 5.2 Eq.7 rel=(1-alpha)*a+alpha*b
def relevance(a, b, alpha):
    return float((1.0 - float(alpha)) * float(a) + float(alpha) * float(b))
# 4.2 q=LLM([P;C]) 的 fixture 替代(纯模板, 不调模型)
def stub_query(phrases, tag=""):
    return "Q[%s]: " % tag + ", ".join(str(x) for x in phrases)
# 5.1 s=LLM(inst,d,q) 的 fixture 替代
def stub_snippet(doc_id, query, concepts):
    return ("D%s about %s. " % (doc_id, ", ".join(str(x) for x in concepts)) +
            "It relates to %s. " % query +
            "Evidence sentence three. Evidence sentence four.")
