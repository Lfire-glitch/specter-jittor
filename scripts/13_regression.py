"""13: 接口化回归校验 —— ① 收敛后, 02/03 已归档数字一个不差; ② 环带表可读。

四道闸 (全部来自代码输出, 不引用记忆里的数字):
  A. PaperEncoder.encode(fmt=None, normalize=False) 对 light 前 48 篇的输出
     与 scibert-embeddings-light.npy 前 48 行逐位一致 (cos>0.9999, max_abs<1e-4)
  B. build_triplets('citation') 与 triplets.jsonl 逐行逐字段相等
  C. fmt='clf' 抛 NotImplementedError (接口形状存在); normalize=True 行范数≈1
  D. build_triplets('knn_ring') 与 triplets_knn_ring.jsonl 逐行逐字段相等
     (读落盘表, 不重写采样逻辑 —— 与 citation 分支对称)
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

    # A: 逐位回归 (与 07b 同权重/同分批/同句子对)
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

    # C: fmt 接口形状存在但未实现
    raised = False
    try:
        enc.encode([meta[ids[0]]], fmt='clf')
    except NotImplementedError:
        raised = True
    print(f'[C ] fmt=clf raises NotImplementedError  {raised}')
    checks.append(('C', raised))

    # D: knn_ring 分支读落盘表, 逐行逐字段相等 (产物一致, 非过程一致)
    ring = build_triplets('knn_ring')
    raw_ring = [json.loads(l) for l in open(os.path.join(T, 'triplets_knn_ring.jsonl'))]
    same_ring = (ring == raw_ring)
    print(f'[D ] knn_ring == triplets_knn_ring.jsonl  {same_ring}  ({len(ring)} rows)')
    checks.append(('D', same_ring and len(ring) == len(raw_ring) and len(ring) > 0))

    print('VERDICT:', 'PASS' if all(v for _, v in checks) else 'FAIL')
    sys.exit(0 if all(v for _, v in checks) else 1)


if __name__ == '__main__':
    main()
