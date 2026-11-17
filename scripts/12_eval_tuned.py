"""M2d: 闭环验收 —— 基线 vs 训练后, 同一 1,200 篇子集, 同一把 kNN 尺.

基线嵌入: scibert-embeddings-light.npy 的对应行 (昨天已产出, 不重算)
训练嵌入: scibert-embeddings-sub-tuned.npy (07c 刚产出)
子集构成: light_ids 的 mag/mesh 各 pool 前 300 + query 前 300
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.eval.classification import knn_classify, macro_f1

D = os.path.join(ROOT, 'data', 'scidocs')
N_EACH = 300
K = 5


def main():
    base_ids = json.load(open(os.path.join(D, 'scibert-paper_ids-light.json')))
    base_emb = np.load(os.path.join(D, 'scibert-embeddings-light.npy'))
    row = {p: i for i, p in enumerate(base_ids)}

    sub_ids = json.load(open(os.path.join(D, 'scibert-paper_ids-sub-tuned.json')))
    tuned = np.load(os.path.join(D, 'scibert-embeddings-sub-tuned.npy'))
    srow = {p: i for i, p in enumerate(sub_ids)}
    light = json.load(open(os.path.join(D, 'light_ids.json')))

    for task in ('mag', 'mesh'):
        pool = list(light[f'{task}_pool'])[:N_EACH]
        query = list(light[f'{task}_query'])[:N_EACH]
        pool_y = np.array([int(light[f'{task}_pool'][p]) for p in pool])
        query_y = np.array([int(light[f'{task}_query'][p]) for p in query])

        results = {}
        for name, pe, qe in [
                ('baseline',
                 base_emb[[row[p] for p in pool]],
                 base_emb[[row[p] for p in query]]),
                ('tuned',
                 tuned[[srow[p] for p in pool]],
                 tuned[[srow[p] for p in query]])]:
            pred = knn_classify(pe, pool_y, qe, k=K)
            results[name] = macro_f1(query_y, pred)

        d = results['tuned'] - results['baseline']
        print(f'[{task}] baseline={results["baseline"]:.3f} '
              f'tuned={results["tuned"]:.3f}  delta={"+" if d>=0 else ""}{d:.3f}')

    json.dump(results, open(os.path.join(ROOT, 'logs', 'm2_eval.json'), 'w'),
              indent=2)


if __name__ == '__main__':
    main()
