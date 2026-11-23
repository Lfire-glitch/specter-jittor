import json
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.concept import (
    best_snippet, consistency_filter, coverage_pi, coverage_rounds,
    distinctiveness, relevance, snippet_evidence, stub_query, stub_snippet,
    subtree_scores, top_fraction_mask, truncate_distribution,
)
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
SEED = 42
N_DOCS, N_PHR, DIM = 8, 120, 16
N_NODES = 30
TOP_FRAC = 0.20
K_P = 20
M = 4
EPS = 1e-3
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
def build_fixture():
    rng = np.random.default_rng(SEED)
    phrase_emb = rng.normal(size=(N_PHR, DIM))
    groups = [list(range(i * 4, i * 4 + 4)) for i in range(N_NODES)]
    doc_emb = np.stack([
        phrase_emb[groups[i % N_NODES]].mean(axis=0) + 0.05 * rng.normal(size=DIM)
        for i in range(N_DOCS)
    ])
    df = np.array([N_DOCS] * 96 + [1, 2, 3] * 8, dtype=np.float64)
    return doc_emb, phrase_emb, groups, df
def main():
    doc_emb, phrase_emb, groups, df = build_fixture()
    rng = np.random.default_rng(SEED + 1)
    node_s, phrase_s = subtree_scores(doc_emb, phrase_emb, groups)
    disc = distinctiveness(df, N_DOCS)
    mask = top_fraction_mask(disc, TOP_FRAC)
    raw = np.clip(phrase_s[0], 0.0, None) + 1e-6
    pi0, kept = truncate_distribution(raw, K_P, mask)
    allowed = pi0 > 0.0
    y_d = np.zeros(N_PHR, dtype=np.float64)
    y_d[kept] = 1.0
    universal_zero = bool((pi0[~mask] == 0.0).all() and (pi0[mask] >= 0.0).all())
    idx_ok = (
        node_s.shape == (N_DOCS, N_NODES)
        and phrase_s.shape == (N_DOCS, N_PHR)
        and bool(np.isfinite(node_s).all() and np.isfinite(phrase_s).all())
        and int(mask.sum()) == int(np.ceil(TOP_FRAC * N_PHR))
        and len(kept) <= K_P
        and abs(pi0.sum() - 1.0) < 1e-12
        and universal_zero
        and bool((y_d[~allowed] == 0.0).all())
    )
    rec, y_q, used = coverage_rounds(y_d, K_P, M, rng, EPS, allowed)
    unions = [r["union"] for r in rec]
    pi_sums = [r["pi_sum"] for r in rec]
    first = np.asarray(rec[0]["picked"], dtype=np.int64)
    y_q1 = np.zeros_like(y_d)
    y_q1[first] = 1.0
    pi_before = coverage_pi(y_d, np.zeros_like(y_d), EPS)
    pi_before[~allowed] = 0.0
    pi_before = pi_before / pi_before.sum()
    pi_after = coverage_pi(y_d, y_q1, EPS)
    pi_after[~allowed] = 0.0
    pi_after = pi_after / pi_after.sum()
    rest = np.flatnonzero(allowed & (y_q1 == 0.0))
    cov_ok = (
        all(abs(s - 1.0) < 1e-12 for s in pi_sums)
        and all(unions[i] < unions[i + 1] for i in range(M - 1))
        and int(used.sum()) == K_P
        and bool((used & ~allowed).sum() == 0)
        and float(pi_after[first].mean()) < float(pi_before[first].mean())
        and float(pi_after[rest].mean()) > float(pi_before[rest].mean())
    )
    q_stub = stub_query(first.tolist(), tag="concept-aligned")
    s_text = np.array([0.90, 0.50, 0.20], dtype=np.float64)
    s_concept = np.array([0.10, 0.95, 0.30], dtype=np.float64)
    keep, fused = consistency_filter(s_text, s_concept, top_n=1)
    text_top1 = int(np.argmax(s_text))
    fused_top1 = int(keep[0])
    filt_ok = (
        text_top1 == 0
        and fused_top1 == 1
        and float(fused[1]) > float(fused[0])
    )
    base = 0.3313
    sims = np.array([0.3124, 0.3412], dtype=np.float64)
    b_mean = snippet_evidence(sims, "mean")
    b_max = snippet_evidence(sims, "max")
    s_star = best_snippet(sims)
    alpha_scan = []
    for a in ALPHAS:
        alpha_scan.append({
            "alpha": float(a),
            "rel_mean": relevance(base, b_mean, a),
            "rel_max": relevance(base, b_max, a),
        })
    rel_ok = (
        b_max > base > b_mean
        and abs(relevance(base, b_max, 0.0) - relevance(base, b_mean, 0.0)) < 1e-12
        and relevance(base, b_max, 0.5) > base
        and relevance(base, b_mean, 0.5) < base
        and s_star == 1
    )
    snippet_stub = stub_snippet("D0", q_stub, first.tolist()[:3])
    checks = [
        {"name": "index: subtree/distinctiveness/top20/kp", "ok": bool(idx_ok)},
        {"name": "coverage: pi normalized + union monotone + covered down", "ok": bool(cov_ok)},
        {"name": "filter: z-score rescues concept-aligned query", "ok": bool(filt_ok)},
        {"name": "relevance: argmax > base > mean", "ok": bool(rel_ok)},
    ]
    ok = all(c["ok"] for c in checks)
    report = {
        "seed": SEED,
        "config": {
            "n_docs": N_DOCS, "n_phrases": N_PHR, "dim": DIM,
            "n_nodes": N_NODES, "top_frac": TOP_FRAC, "k_p": K_P,
            "rounds": M, "eps": EPS, "alphas": ALPHAS,
        },
        "index": {
            "node_shape": list(node_s.shape),
            "phrase_shape": list(phrase_s.shape),
            "top20_kept": int(mask.sum()),
            "kp_kept": int(len(kept)),
            "kept": kept.tolist(),
            "pi_sum": float(pi0.sum()),
            "universal_high_df_zero_mass": universal_zero,
        },
        "coverage": {
            "rounds": rec,
            "unions": unions,
            "pi_sums": pi_sums,
            "covered_pi_before": float(pi_before[first].mean()),
            "covered_pi_after": float(pi_after[first].mean()),
            "uncovered_pi_before": float(pi_before[rest].mean()),
            "uncovered_pi_after": float(pi_after[rest].mean()),
            "final_union": int(used.sum()),
        },
        "filter": {
            "s_text": s_text.tolist(),
            "s_concept": s_concept.tolist(),
            "fused": fused.tolist(),
            "text_top1": text_top1,
            "fused_top1": fused_top1,
            "keep": keep.tolist(),
        },
        "relevance": {
            "base": base,
            "snippet_sims": sims.tolist(),
            "mean": b_mean,
            "max": b_max,
            "best_snippet": s_star,
            "alpha_scan": alpha_scan,
        },
        "stubs": {
            "query": q_stub,
            "snippet": snippet_stub,
        },
        "checks": checks,
        "verdict": "PASS" if ok else "FAIL",
        "note": (
            "纯确定性技术演示: 受控人造概念索引 + LLM fixture/stub, "
            "不新增 retriever、不引测试集、不跑 R@K/NDCG/MAP、不微调、"
            "不报检索增益。只断言调度机制方向, 绝对数字不与论文比较、不外推。"
        ),
    }
    with open(os.path.join(LOG_DIR, "concept_demo.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("[index] nodes=%d phrases=%d top20=%d kp=%d kept=%d pi_sum=%.6f" %
          (N_NODES, N_PHR, int(mask.sum()), K_P, len(kept), pi0.sum()), flush=True)
    print("[coverage] unions=%s covered_pi %.5f->%.5f uncovered_pi %.5f->%.5f" %
          (unions, pi_before[first].mean(), pi_after[first].mean(),
           pi_before[rest].mean(), pi_after[rest].mean()), flush=True)
    print("[filter] text_top1=%d fused_top1=%d fused=%s" %
          (text_top1, fused_top1, [round(float(x), 4) for x in fused]), flush=True)
    print("[relevance] mean=%.4f base=%.4f max=%.4f s*=%d" %
          (b_mean, base, b_max, s_star), flush=True)
    print("VERDICT:", report["verdict"], flush=True)
    print("[ok] logs/concept_demo.json", flush=True)
    sys.exit(0 if ok else 1)
if __name__ == "__main__":
    main()
