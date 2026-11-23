"""M6 / 17 —— 受控两层 planted 社区图 + Leiden 两级 CPM 分区。

真实引文图不可得(截断片段 / 4421 孤儿), 故构造已知真值的两层随机连边图:
L1=30 个大社区, 每 L1 内含 8 个 L2 子社区; 连边概率 P_IN>P_MID>P_OUT
形成密度梯度。真跑 leidenalg.CPMVertexPartition 全局切 L1、每个 L1 内
subgraph 再切 L2, gamma 按「社区数最接近真值规模」扫描选取。
产出 communities.npz(标签) + logs/m6_communities.json(统计), 均被忽略。
诚实边界: planted 图非论文 OpenAlex 引文社区, ARI 只作仪器自检, 绝对数字不外推。
"""

import json
import os
from collections import defaultdict

import igraph as ig
import leidenalg as la
import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, 'data', 'scidocs')
OUT_DIR = os.path.join(ROOT, 'data', 'audit')
LOG_DIR = os.path.join(ROOT, 'logs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 42
N_L1 = 30         
N_L2_PER_L1 = 8    
P_IN = 0.15         
P_MID = 0.02       
P_OUT = 0.002       
W_IN, W_MID, W_OUT = 1.0, 0.5, 0.2   
MIN_SPLIT = 50      

# CPM 的 gamma 是社区内密度阈值, 与图规模相关, 需扫描不能照搬论文
GAMMAS = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
GAMMA2_CANDIDATES = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
TARGET_L2 = N_L1 * N_L2_PER_L1  


# 随机抽两篇同社区的概率(允许抽到自己): sum((n_c/n)^2); 18 改用排除自身版
def rand_base(sizes, n):
    s = np.asarray(sizes, 'float64') / n
    return float((s * s).sum())


# 生成已知真值的 l1_true/l2_true: 30 个 L1, 每个均分 8 个 L2
def build_planted(rng, n):
    l1_true = np.repeat(np.arange(N_L1), int(np.ceil(n / N_L1)))[:n].copy()
    rng.shuffle(l1_true)
    l2_true = np.empty(n, dtype=np.int64)
    nxt = 0
    for c in range(N_L1):
        members = np.where(l1_true == c)[0]
        rng.shuffle(members)
        parts = np.array_split(members, N_L2_PER_L1)
        for p in parts:
            l2_true[p] = nxt
            nxt += 1
    return l1_true, l2_true


# L2 = 每个大 L1 内部再切 CPM; 层次切分而非全局重切
def hierarchical_l2(g, l1, gamma2, min_split):
    n = len(l1)
    labels = np.empty(n, dtype=np.int64)
    nxt = 0
    n_split = 0
    for c in sorted(set(l1.tolist())):
        members = np.where(l1 == c)[0]
        if len(members) >= min_split:
            sub = g.subgraph(members.tolist())
            w = [float(x) for x in sub.es['weight']]
            p2 = la.find_partition(sub, la.CPMVertexPartition,
                                   weights=w, resolution_parameter=gamma2)
            n_split += 1
            for sc in p2:
                for v in sc:
                    # subgraph 的局部下标 v 必须经 members 映射回全局
                    labels[members[v]] = nxt
                nxt += 1
        else:
            for v in members:
                labels[v] = nxt
            nxt += 1
    return labels, n_split


# 三级密度: 同 L2 最密 / 同 L1 跨 L2 中等 / 跨 L1 最稀
def build_edges(rng, n, l1_true, l2_true):
    ew = {}

    def put(a, b, w):
        if a == b:
            return
        k = (a, b) if a < b else (b, a)
        ew[k] = ew.get(k, 0.0) + w

    by_l2 = defaultdict(list)
    for i in range(n):
        by_l2[l2_true[i]].append(i)
    for members in by_l2.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                if rng.random() < P_IN:
                    put(members[x], members[y], W_IN)

    by_l1 = defaultdict(list)
    for i in range(n):
        by_l1[l1_true[i]].append(i)
    for members in by_l1.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                if l2_true[members[x]] != l2_true[members[y]] and rng.random() < P_MID:
                    put(members[x], members[y], W_MID)

    n_cross = int(n * (n - 1) / 2 * P_OUT)
    a = rng.integers(0, n, n_cross)
    b = rng.integers(0, n, n_cross)
    for x, y in zip(a, b):
        if l1_true[x] != l1_true[y] and rng.random() < 0.5:
            put(int(x), int(y), W_OUT)
    return ew


def main():
    ids = json.load(open(os.path.join(S, 'scibert-paper_ids-light.json')))
    n = len(ids)
    rng = np.random.default_rng(SEED)
    l1_true, l2_true = build_planted(rng, n)
    ew = build_edges(rng, n, l1_true, l2_true)
    keys = list(ew.keys())
    weights = [float(ew[k]) for k in keys]
    g = ig.Graph(n=n, edges=keys, directed=False)
    g.es['weight'] = weights
    comp = g.connected_components()
    print('[graph] n=%d edges=%d mean_deg=%.2f orphans=%d comps=%d largest=%d' %
          (n, len(keys), 2.0 * len(keys) / n,
           sum(1 for v in g.vs if g.degree(v) == 0), len(comp),
           max(len(c) for c in comp)), flush=True)
    print('[truth] L1=%d L2=%d  L2_mean=%.1f  base_L1=%.5f base_L2=%.5f' %
          (len(set(l1_true)), len(set(l2_true)), n / len(set(l2_true)),
           rand_base(np.bincount(l1_true), n), rand_base(np.bincount(l2_true), n)), flush=True)

    scan = []
    for gamma in GAMMAS:
        p = la.find_partition(g, la.CPMVertexPartition,
                              weights='weight', resolution_parameter=gamma)
        sizes = np.bincount(np.asarray(p.membership))
        scan.append({'gamma': gamma, 'communities': len(sizes),
                     'max_pct': float(sizes.max() / n),
                     'base': rand_base(sizes, n)})
        print('[scan] gamma=%.0e  comm=%d  max=%.1f%%  base=%.5f' %
              (gamma, len(sizes), 100.0 * sizes.max() / n,
               rand_base(sizes, n)), flush=True)
    gamma1 = min(scan, key=lambda r: abs(r['communities'] - N_L1))['gamma']
    p1 = la.find_partition(g, la.CPMVertexPartition,
                           weights='weight', resolution_parameter=gamma1)
    l1 = np.asarray(p1.membership, dtype=np.int64)
    print('[L1] gamma1=%.0e  comm=%d  ARI_vs_truth=%.4f' %
          (gamma1, len(set(l1)), adjusted_rand_score(l1_true, l1)), flush=True)

    g2_scan = []
    for gamma2_try in GAMMA2_CANDIDATES:
        lab, _ = hierarchical_l2(g, l1, gamma2_try, MIN_SPLIT)
        g2_scan.append({'gamma2': gamma2_try, 'l2_communities': int(len(set(lab.tolist()))),
                        'ari': float(adjusted_rand_score(l2_true, lab))})
        print('[g2scan] gamma2=%.0e  L2_comm=%d  ARI=%.4f' %
              (gamma2_try, len(set(lab.tolist())),
               adjusted_rand_score(l2_true, lab)), flush=True)
    # 取舍点: 按「社区数对齐 240」选 gamma2(而非 ARI 最优), 见 docs/08
    gamma2 = min(g2_scan, key=lambda r: abs(r['l2_communities'] - TARGET_L2))['gamma2']
    l2, n_split = hierarchical_l2(g, l1, gamma2, MIN_SPLIT)
    print('[L2] gamma2=%.0e  comm=%d  split_L1=%d  ARI_vs_truth=%.4f' %
          (gamma2, len(set(l2.tolist())), n_split,
           adjusted_rand_score(l2_true, l2)), flush=True)
    s1, s2 = np.bincount(l1), np.bincount(l2)
    print('[base] L1=%.5f  L2=%.5f  (随机两篇同社区概率)' %
          (rand_base(s1, n), rand_base(s2, n)), flush=True)

    np.savez(os.path.join(OUT_DIR, 'communities.npz'),
             ids=np.array(ids), l1=l1, l2=l2,
             l1_true=l1_true, l2_true=l2_true,
             gamma1=gamma1, gamma2=gamma2)
    stats = {
        'n': n, 'edges': len(keys), 'mean_degree': 2.0 * len(keys) / n,
        'n_l1_true': int(len(set(l1_true))), 'n_l2_true': int(len(set(l2_true))),
        'gamma_scan': scan, 'gamma1': gamma1,
        'gamma2_scan': g2_scan, 'gamma2': gamma2, 'target_l2': TARGET_L2,
        'min_split': MIN_SPLIT, 'n_split_l1': n_split,
        'l1_communities': int(len(s1)), 'l2_communities': int(len(s2)),
        'l1_ari_vs_truth': float(adjusted_rand_score(l1_true, l1)),
        'l2_ari_vs_truth': float(adjusted_rand_score(l2_true, l2)),
        'base_l1': rand_base(s1, n), 'base_l2': rand_base(s2, n),
        # 诚实边界随产物落盘, 防止把合成图误当真实引文图
        'note': 'planted partition, NOT a real citation graph; absolute numbers not comparable',
    }
    json.dump(stats, open(os.path.join(LOG_DIR, 'm6_communities.json'), 'w'), indent=2)
    print('[ok] data/audit/communities.npz + logs/m6_communities.json', flush=True)


if __name__ == '__main__':
    main()
