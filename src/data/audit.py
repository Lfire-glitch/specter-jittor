import numpy as np
def analytic_base(labels):
    cnt=np.bincount(labels).astype(np.float64)
    n=int(cnt.sum())
    return float((cnt*cnt).sum()-n)/(n*(n-1))
def analytic_base_in(labels):
    cnt=np.bincount(labels).astype(np.float64)
    p=cnt/cnt.sum()
    return float((p*p).sum())
def mc_base(labels, trials, rng):
    n = len(labels)
    a = rng.integers(0, n, trials)
    b = rng.integers(0, n, trials)
    keep = a != b
    return float((labels[a[keep]] == labels[b[keep]]).mean())
def rate_from_top_idx(labels, top_idx):
    return float((labels[top_idx] == labels[:, None]).mean())
def topk_same_rate(E, labels, top_k=10, block_size=256):

    n = E.shape[0]
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    top_idx = np.empty((n, top_k), dtype=np.int64)
    for s in range(0, n, block_size):
        e = min(s + block_size, n)
        sim = En[s:e] @ En.T
        for j, qi in enumerate(range(s, e)):
            sim[j, qi] = -2.0
            top_idx[qi] = np.argsort(-sim[j])[:top_k]
    return rate_from_top_idx(labels, top_idx), top_idx
def permutation_lifts(labels, top_idx, base_excl, trials, rng):
    n = len(labels)
    vals = np.empty(trials, np.float64)
    for t in range(trials):
        lab = labels[rng.permutation(n)]
        vals[t] = rate_from_top_idx(lab, top_idx) / base_excl
    return vals
def audit_labels(E, labels, rng, top_k=10, block_size=256,mc_trials=200000, perm_trials=1000):
    labels = np.asarray(labels)
    rate, top_idx = topk_same_rate(E, labels, top_k, block_size)
    base_excl = analytic_base(labels)
    base_incl = analytic_base_in(labels)
    mc = mc_base(labels, mc_trials, rng)
    lift = rate / base_excl
    vals = permutation_lifts(labels, top_idx, base_excl, perm_trials, rng)
    p_lo = float(np.percentile(vals, 2.5))
    p_hi = float(np.percentile(vals, 97.5))
    return {
        "n": int(E.shape[0]),
        "classes": int(len(set(labels.tolist()))),
        "top_k": top_k,
        "rate": rate,
        "base_excl": base_excl,
        "base_incl": base_incl,
        "mc_base": mc,
        "lift": float(lift),
        "lift_incl": float(rate / base_incl),
        "perm": {
            "trials": perm_trials,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p2_5": p_lo,
            "p97_5": p_hi,
            "obs_percentile": float((vals < lift).mean()),
        },
        "in_ci_95": bool(p_lo <= lift <= p_hi),
    }