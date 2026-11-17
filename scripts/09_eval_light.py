import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.eval.classification import knn_classify, macro_f1

D = os.path.join(ROOT, 'data', 'scidocs')
K = 5
def main():
    ids = json.load(open(os.path.join(D, 'scibert-paper_ids-light.json')))
    emb = np.load(os.path.join(D, 'scibert-embeddings-light.npy'))
    row = {p: i for i, p in enumerate(ids)}     
    light = json.load(open(os.path.join(D, 'light_ids.json')))
    assert emb.shape == (len(ids), 768), '嵌入与 id 清单长度不一致'

    for task in ('mag', 'mesh'):
        pool = light[f'{task}_pool']
        query = light[f'{task}_query']
        pool_ids = list(pool)
        query_ids = list(query)
        pool_e = emb[[row[p] for p in pool_ids]]
        query_e = emb[[row[p] for p in query_ids]]
        pool_y = np.array([int(pool[p]) for p in pool_ids])
        query_y = np.array([int(query[p]) for p in query_ids])
        majority = macro_f1(query_y, np.full_like(query_y,
                         np.bincount(pool_y).argmax()))
        rng = np.random.default_rng(42)
        random_f1 = macro_f1(query_y, rng.choice(pool_y, len(query_y)))
        pred = knn_classify(pool_e, pool_y, query_e, k=K)
        f1 = macro_f1(query_y, pred)
        print(f'[{task}] classes={len(set(pool_y.tolist()))} '
              f'pool={len(pool_y)} query={len(query_y)} k={K}')
        print(f'  majority F1 = {majority:.3f}   random F1 = {random_f1:.3f}')
        print(f'  kNN F1      = {f1:.3f}   '
              f'(lift = +{f1 - majority:.3f})')


if __name__ == '__main__':
    main()