import numpy as np
def l2norm(x,eps=1e-12):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),eps)
def subtree_scores(doc_emb,phrase_emb,groups):
    d=l2norm(doc_emb)
    p=l2norm(phrase_emb)
    centers=l2norm(np.stack([p[g].mean(axis=0) for g in groups]))
    node=d@centers.T
    phrase=np.zeros((d.shape[0],p.shape[0]),dtype=np.float64)
    for c,g in enumerate(groups):
        phrase[:,g]=node[:,c:c+1]
    return node ,phrase
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
def coverage_pi(y_d,y_q,eps=1e-3):
    y_d=np.asarray(y_d,dtype=np.float64)
    y_q=np.asarray(y_q,dtype=np.float64)
    gap=np.maximum(y_d-y_q,eps)
    return gap/gap.sum()
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
def fuse_scores(s_text, s_concept):
    return zscore(s_text) + zscore(s_concept)
def consistency_filter(s_text, s_concept, top_n):
    fused = fuse_scores(s_text, s_concept)
    keep = np.argsort(-fused)[:int(top_n)]
    return keep, fused
def best_snippet(sims):
    return int(np.argmax(np.asarray(sims, dtype=np.float64)))
def snippet_evidence(sims, agg):
    sims = np.asarray(sims, dtype=np.float64)
    if agg == "max":
        return float(sims.max())
    if agg == "mean":
        return float(sims.mean())
    raise ValueError("agg must be 'max' or 'mean'")
def relevance(a, b, alpha):
    return float((1.0 - float(alpha)) * float(a) + float(alpha) * float(b))
def stub_query(phrases, tag=""):
    return "Q[%s]: " % tag + ", ".join(str(x) for x in phrases)
def stub_snippet(doc_id, query, concepts):
    return ("D%s about %s. " % (doc_id, ", ".join(str(x) for x in concepts)) +
            "It relates to %s. " % query +
            "Evidence sentence three. Evidence sentence four.")
