import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'training')
LOGS = os.path.join(ROOT, 'logs')

EMB_FILE = os.path.join(DATA, 'train_emb.npy')
IDS_FILE = os.path.join(DATA, 'train_ids.json')
CIT_FILE = os.path.join(DATA, 'triplets.jsonl')
OUT_FILE = os.path.join(DATA, 'triplets_knn_ring.jsonl')
STAT_FILE = os.path.join(LOGS, 'm4_ring_stats.json')

k_Pos_lo,k_Pos_hi=20,25
k_Neg_lo,k_Neg_hi=3998,4000
keep_pos=False
seed=42
Block=256

def l2norm(e):
    return e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
def ring_indices(order,self_i,lo,hi):
    others=order[order!=self_i]
    return others[lo-1:hi]
def main():
    emb=np.load(EMB_FILE).astype(np.float32)
    ids=json.load(open(IDS_FILE))
    n=len(ids)
    assert emb.shape == (n, 768), f'emb/ids 不对齐: {emb.shape} vs {n}'
    assert k_Neg_hi < n, f'k- 上界 {k_Neg_hi} 超出池大小 {n}'
    print(f'[pool] {n} papers, emb={emb.shape}', flush=True)
    row={p:i for i,p in enumerate(ids)}
    citation=[json.loads(line) for line in open(CIT_FILE)]
    qs=[t['q'] for t in citation]
    assert all(q in row for q in qs), '存在不在检索池里的 q'
    print(f'[q] {len(qs)} queries (frozen from citation table)', flush=True)
    E=l2norm(emb)
    rng=np.random.default_rng(seed)
    q_idx=np.array([row[q] for q in qs],dtype=np.int64)
    out=[]
    cos_pos,cos_neg,cos_cit=[],[],[]
    n_pos_same,n_neg_same=0,0
    for s in range(0,len(qs),Block):
        block=q_idx[s:s+Block]
        sim=E[block]@E.T
        for j,qi in enumerate(block):
            order=np.argsort(-sim[j])
            pos_band=ring_indices(order,qi,k_Pos_lo,k_Pos_hi)
            neg_band=ring_indices(order,qi,k_Neg_lo,k_Neg_hi)
            t=citation[s+j]
            if keep_pos:
                pos=t['pos']
            else:
                pos=ids[int(rng.choice(pos_band))]
            negs=[ids[int(i)] for i in neg_band]
            assert qi not in pos_band and qi not in neg_band
            assert len(set(pos_band.tolist()) & set(neg_band.tolist())) == 0
            out.append({'q': ids[int(qi)], 'pos': pos, 'negs': negs})
            cos_pos.append(float(sim[j, pos_band].mean()))
            cos_neg.append(float(sim[j, neg_band].mean()))
            cos_cit.append(float(sim[j, [row[p] for p in t['negs']]].mean()))
            n_pos_same += int(pos == t['pos'])
            n_neg_same += len(set(negs) & set(t['negs']))
        print(f'[ring] {min(s + Block, len(qs))}/{len(qs)}', flush=True)
    with open(OUT_FILE, 'w') as f:
        for t in out:
            f.write(json.dumps(t) + '\n')

    stats = {
        'pool': n,
        'k_pos': [k_Pos_lo, k_Pos_hi],
        'k_neg': [k_Neg_lo, k_Neg_hi],
        'keep_citation_pos': keep_pos,
        'rows': len(out),
        'negs_per_row': len(out[0]['negs']),
        'cos_pos_band_mean': float(np.mean(cos_pos)),
        'cos_neg_band_mean': float(np.mean(cos_neg)),
        'cos_citation_neg_mean': float(np.mean(cos_cit)),
        'pos_same_as_citation': n_pos_same,
        'neg_same_as_citation': n_neg_same,
    }
    json.dump(stats, open(STAT_FILE, 'w'), indent=2)
    print(f'[ok] {OUT_FILE}  ({len(out)} rows)', flush=True)
    print(f'[stats] cos: pos_band={stats["cos_pos_band_mean"]:.4f}  '
          f'neg_band={stats["cos_neg_band_mean"]:.4f}  '
          f'citation_neg={stats["cos_citation_neg_mean"]:.4f}', flush=True)
    ok = stats['cos_pos_band_mean'] > stats['cos_neg_band_mean']
    print(f'VERDICT: {"PASS (方向对齐)" if ok else "CHECK"}', flush=True)
if __name__=="__main__":
    main()