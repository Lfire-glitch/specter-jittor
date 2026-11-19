"""13: 接口化回归校验 —— ①/② 归档数字不变; M5 fmt 接口形状成立。

闸位 (全部来自代码输出, 不引用记忆里的数字):
  A. PaperEncoder.encode(fmt=None, normalize=False) 对 light 前 48 篇的输出
     与 scibert-embeddings-light.npy 前 48 行逐位一致 (cos>0.9999, max_abs<1e-4)
  A2. normalize=True 行范数≈1
  B. build_triplets('citation') 与 triplets.jsonl 逐行逐字段相等
  C. fmt='xxx' 抛 ValueError (M5 后接口边界; 旧版是 fmt='clf' 抛 NotImplementedError)
  D. build_triplets('knn_ring') 与 triplets_knn_ring.jsonl 逐行逐字段相等
  E. fmt 分支形状: fmt=None 与 fmt='clf' 输出流可区分 + 有限 + 确定性
     (E* 为记录项, 不作判据: cos(clf,rgn) 未训练控制码下≈1, 格式尚不可分)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model.encoder import PaperEncoder
from src.data.triplets import build_triplets

D = os.path.join(ROOT, 'data', 'scidocs')
T = os.path.join(ROOT, 'data', 'training')


def cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    checks = []

    enc = PaperEncoder()                     # 默认 scibert.npz, max_len=512, bs=16
    meta = json.load(open(os.path.join(D, 'paper_metadata_mag_mesh.json')))
    ids = json.load(open(os.path.join(D, 'scibert-paper_ids-light.json')))[:48]

    # A: 逐位回归 (与 07b 同权重/同分批/同句子对; fmt=None 路径零改动)
    e = enc.encode([meta[p] for p in ids])
    ref = np.load(os.path.join(D, 'scibert-embeddings-light.npy'))[:48]
    c, m = cos(e, ref), float(np.abs(e - ref).max())
    print(f'[A ] encode vs 07b  cos={c:.10f}  max_abs={m:.2e}')
    checks.append(('A', c > 0.9999 and m < 1e-4))

    en = enc.encode([meta[p] for p in ids], normalize=True)
    nrm = float(np.linalg.norm(en, axis=1).mean())
    print(f'[A2] normalize=True  mean_row_norm={nrm:.4f}')
    checks.append(('A2', abs(nrm - 1.0) < 1e-3))

    # B: 采样表冻结, 逐行逐字段相等
    t = build_triplets('citation')
    raw = [json.loads(l) for l in open(os.path.join(T, 'triplets.jsonl'))]
    same = (t == raw)
    print(f'[B ] citation == triplets.jsonl  {same}  ({len(t)} rows)')
    checks.append(('B', same and len(t) == len(raw)))

    # C: fmt 接口边界 (M5 后非法格式必须拒绝)
    raised = False
    try:
        enc.encode([meta[ids[0]]], fmt='xxx')
    except ValueError:
        raised = True
    print(f'[C ] fmt=xxx raises ValueError  {raised}')
    checks.append(('C', raised))

    # D: knn_ring 分支读落盘表, 逐行逐字段相等 (产物一致, 非过程一致)
    ring = build_triplets('knn_ring')
    raw_ring = [json.loads(l) for l in open(os.path.join(T, 'triplets_knn_ring.jsonl'))]
    same_ring = (ring == raw_ring)
    print(f'[D ] knn_ring == triplets_knn_ring.jsonl  {same_ring}  ({len(ring)} rows)')
    checks.append(('D', same_ring and len(ring) == len(raw_ring) and len(ring) > 0))

    # E: M5 fmt 分支形状。用同一小 batch 单独 encode, 避免与 A 的 padding 口径混淆。
    #    只判「输出流确实改变 + 有限 + 确定性」; 不要求 clf/rgn 可分(需训练, 见 E*)。
    sub = [meta[p] for p in ids][:8]
    e_ref = enc.encode(sub)
    e_clf = enc.encode(sub, fmt='clf')
    e_rgn = enc.encode(sub, fmt='rgn')
    e_clf2 = enc.encode(sub, fmt='clf')
    c_nc = cos(e_ref, e_clf)
    c_cr = cos(e_clf, e_rgn)
    finite = bool(np.isfinite(e_clf).all() and np.isfinite(e_rgn).all())
    det = bool(np.array_equal(e_clf, e_clf2))
    print(f'[E ] cos(None,clf)={c_nc:.4f}  finite={finite}  deterministic={det}')
    print(f'[E*] cos(clf,rgn)={c_cr:.4f}  (记录项, 不作判据: 控制码未训练, 格式尚不可分)')
    checks.append(('E', c_nc < 0.9 and finite and det))

    print('VERDICT:', 'PASS' if all(v for _, v in checks) else 'FAIL')
    sys.exit(0 if all(v for _, v in checks) else 1)


if __name__ == '__main__':
    main()
