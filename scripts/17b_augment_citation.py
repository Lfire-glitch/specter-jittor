"""M6 / 17b —— 引文增强(direct + BC + CC)算子与三层消融。

论文 3.2 的增强图来自 OpenAlex 全表 2.5B 条引文边; 本地真实
references/cited_by 是截断片段, 实测 BC=1 边 / CC=8 边 / 合图 4421 孤儿,
Leiden 在任何 gamma 下只能切出单例社区。本脚本用「受控合成引文」把
BC/CC 倒排 + Salton 加权 + 三层叠加这条因果链端到端真跑一遍,
并做 direct / +BC / +BC+CC 三层消融, 让两级 CPM 有可分区的输入。
产出 augmented.npz + logs/m6_augment_ablation.json, 均被忽略。
诚实边界: 合成 token 按真值社区生成, ARI=1 是设计出的可分性,
不代表分区算法能力或论文效果; 合成引文 NOT OpenAlex。
"""

import json
import os
from collections import defaultdict
import igraph as ig
import leidenalg as la
import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "data", "audit")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(A, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
SEED = 42
N2, P2 = 8, 6   
N1, P1 = 8, 3           
NN, PN = 60, 2         
MIN_SHARED, CAP = 3, 500
P_DIRECT = 0.25      
GAMMAS1 = [1e-4, 1e-3, 1e-2]
GAMMAS2 = [1e-3, 1e-2, 1e-1]
TARGET_L1, TARGET_L2 = 30, 240
MIN_SPLIT = 50
# 三级 token 池: L2 私有(最密) / L1 私有(中) / 全局噪声(最稀), 造密度梯度
def build_synthetic(ids,l1_true,l2_true,idx):
    n=len(ids)
    rng=np.random.default_rng(SEED)
    by_l2=defaultdict(list)
    for i in range(n):
        by_l2[int(l2_true[i])].append(i)
    refs=[set() for _ in range(n)]
    cits=[set() for _ in range(n)]
    for i in range(n):
        c1,c2=int(l1_true[i]),int(l2_true[i])
        for t in rng.choice(N2, P2, replace=False):
            refs[i].add("R2:%d:%d" % (c2, t))
        for t in rng.choice(N2, P2, replace=False):
            cits[i].add("C2:%d:%d" % (c2, t))
        for t in rng.choice(N1, P1, replace=False):
            refs[i].add("R1:%d:%d" % (c1, t))
        for t in rng.choice(N1, P1, replace=False):
            cits[i].add("C1:%d:%d" % (c1, t))
        for t in rng.choice(NN, PN, replace=False):
            refs[i].add("RN:%d" % t)
        for t in rng.choice(NN, PN, replace=False):
            cits[i].add("CN:%d" % t)
        if rng.random()<P_DIRECT:
            j=int(rng.choice(by_l2[c2]))
            if j!=i:
                refs[i].add(ids[j])
                cits[j].add(ids[i])
    # 只用池外 id 做倒排: 池内 id 归 direct 层, 混入会污染三层消融
    ext_refs=[set(x for x in s if x not in idx) for s in refs]
    ext_cits=[set(x for x in s if x not in idx) for s in cits]
    return refs,cits,ext_refs,ext_cits
# direct 层: refs|cits 中落在语料内的 id 建无向边(最稀疏)
def direct_edges(refs,cits,idx,n):
    e={}
    for i in range(n):
        for x in refs[i]|cits[i]:
            j=idx.get(x)
            if j is None or j==i:
                continue
            a,b=(i,j)if i<j else (j,i)
            e[(a,b)]=e.get((a,b),0.0)+1.0
    return e
# BC/CC 共用倒排: 外部 id -> 共享该 id 的论文对, sh>=min_shared 才建边
def inverted(sets,cap,min_shared):
    inv=defaultdict(list)
    for i,s in enumerate(sets):
        for x in s:
            inv[x].append(i)
    hot=sum(1 for v in inv.values() if len(v)>cap)
    pair=defaultdict(int)
    for x,vs in inv.items():
        if len(vs)>cap or len(vs)<min_shared:
            continue
        for a in range(len(vs)):
            for b in range(a+1,len(vs)):
                i,j=(vs[a],vs[b]) if vs[a]<vs[b] else (vs[b],vs[a])
                pair[(i,j)]+=1
    e={}
    for (i,j),sh in pair.items():
        if sh<min_shared:
            continue
        # Salton 余弦权重 sh/sqrt(|S_i||S_j|), 消除参考文献数量差异
        den=(len(sets[i])*(len(sets[j])))**0.5
        if den>0:
            e[(i,j)]=sh/den
    return e,len(inv),hot
def hierarchical(g,l1,gamma2,min_split,n):
    labels=np.empty(n,np.int64)
    nxt,n_split=0,0
    for c in sorted(set(l1.tolist())):
        members=np.where(l1==c)[0]
        if len(members)>=min_split:
            sub=g.subgraph(members.tolist())
            w=[float(x) for x in sub.es["weight"]]
            p2=la.find_partition(sub,la.CPMVertexPartition,weights=w,resolution_parameter=gamma2)
            n_split+=1
            for sc in p2:
                for v in sc:
                    labels[members[v]]=nxt
                nxt+=1
        else:
            for v in members:
                labels[v]=nxt
            nxt+=1
    return labels,n_split
def make_graph(ew,n):
    keys=list(ew.keys())
    w=[float(ew[k]) for k in keys]
    g=ig.Graph(n=n,edges=keys,directed=False)
    g.es["weight"]=w
    return g,w
def main():
    z = np.load(os.path.join(A, "communities.npz"), allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    l1_true, l2_true = z["l1_true"], z["l2_true"]
    n = len(ids)
    idx = {p: i for i, p in enumerate(ids)}
    refs, cits, ext_refs, ext_cits = build_synthetic(ids, l1_true, l2_true, idx)
    layers, inv_info = {}, {}
    e = direct_edges(refs, cits, idx, n)
    layers["direct"] = dict(e)
    # +BC: 共同参考文献边叠加到同一 edge_w dict
    bc, nb, hb = inverted(ext_refs, CAP, MIN_SHARED)
    inv_info["BC"] = {"inv_keys": nb, "hot": hb, "edges": len(bc)}
    for k, v in bc.items():
        e[k] = e.get(k, 0.0) + v
    layers["+BC"] = dict(e)
    # +CC: 共同被引边; 与 BC 是同一函数、不同输入
    cc, nc, hc = inverted(ext_cits, CAP, MIN_SHARED)
    inv_info["CC"] = {"inv_keys": nc, "hot": hc, "edges": len(cc)}
    for k, v in cc.items():
        e[k] = e.get(k, 0.0) + v
    # 三层各存一次快照, 保证消融互不串改
    layers["+BC+CC"] = dict(e)
    report = {
        "n": n, "seed": SEED,
        "params": {"N2": N2, "P2": P2, "N1": N1, "P1": P1, "NN": NN, "PN": PN,
                   "min_shared": MIN_SHARED, "cap": CAP, "p_direct": P_DIRECT,
                   "gammas1": GAMMAS1, "gammas2": GAMMAS2,
                   "target_l1": TARGET_L1, "target_l2": TARGET_L2,
                   "min_split": MIN_SPLIT},
        "inverted": inv_info, "layers": [],
    }
    aug = {}
    for tag, ew in layers.items():
        g, w = make_graph(ew, n)
        comp = g.connected_components()
        row = {
            "layer": tag, "edges": len(ew),
            "mean_degree": 2.0 * len(ew) / n,
            "orphans": int(sum(1 for v in g.vs if g.degree(v) == 0)),
            "components": len(comp), "largest": max(len(c) for c in comp),
            "weight_sum": float(sum(w)),
        }
        best1 = None
        for g1 in GAMMAS1:
            p1 = la.find_partition(g, la.CPMVertexPartition,
                                   weights="weight", resolution_parameter=g1)
            l1 = np.asarray(p1.membership)
            k1 = len(set(l1.tolist()))
            if best1 is None or abs(k1 - TARGET_L1) < abs(best1[0] - TARGET_L1):
                best1 = (k1, g1, l1)
        row["l1"] = {"gamma": best1[1], "communities": best1[0],
                     "ari_vs_truth": float(adjusted_rand_score(l1_true, best1[2]))}
        best2 = None
        for g2 in GAMMAS2:
            l2, ns = hierarchical(g, best1[2], g2, MIN_SPLIT, n)
            k2 = len(set(l2.tolist()))
            a2 = float(adjusted_rand_score(l2_true, l2))
            if best2 is None or abs(k2 - TARGET_L2) < abs(best2[0] - TARGET_L2):
                best2 = (k2, g2, a2, ns, l2)
        row["l2"] = {"gamma": best2[1], "communities": best2[0],
                     "ari_vs_truth": best2[2], "split_l1": best2[3]}
        report["layers"].append(row)
        if tag == "+BC+CC":
            aug = {"l1_aug": best1[2], "l2_aug": best2[4],
                   "keys": list(ew.keys()), "weights": w}
        print("[%s] edges=%d orphans=%d comps=%d | L1 comm=%d ARI=%.4f | L2 comm=%d ARI=%.4f" %
              (tag, row["edges"], row["orphans"], row["components"],
               row["l1"]["communities"], row["l1"]["ari_vs_truth"],
               row["l2"]["communities"], row["l2"]["ari_vs_truth"]), flush=True)
    np.savez(os.path.join(A, "augmented.npz"),
             l1_aug=aug["l1_aug"], l2_aug=aug["l2_aug"],
             edge_a=np.array([k[0] for k in aug["keys"]]),
             edge_b=np.array([k[1] for k in aug["keys"]]),
             weights=np.array(aug["weights"]))
    # 诚实边界随日志落盘
    report["note"] = ("synthetic citations, NOT OpenAlex; ARI=1 is designed "
                      "separability, not algorithm capability or paper result")
    json.dump(report, open(os.path.join(LOG_DIR, "m6_augment_ablation.json"), "w"), indent=2)
    print("[ok] data/audit/augmented.npz + logs/m6_augment_ablation.json", flush=True)
if __name__ == "__main__":
    main()
