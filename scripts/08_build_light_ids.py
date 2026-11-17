"""Build the light validation subset v3 (CPU 轻量验证, 不跑全量 48k).

v3 变更: 放弃自建链接对(metadata 的 references 指向语料外的论文);
轻量评测 = MAG 分类 + MeSH 分类, 两个任务共用 kNN + macro-F1 实现.
v1 的 cite qrels 依赖未下载的 paper_metadata_view_cite_read.json, 同样弃用.

产出 data/scidocs/light_ids.json:
  mag_pool/mag_query   : mag  train/test 各随机抽样 {pid: label}
  mesh_pool/mesh_query : mesh train/test 各随机抽样 {pid: label}
  all_ids              : 四组去重排序的并集, 编码脚本按此清单跑
seed=42 固定抽样, 保证可复现。
"""
import csv
import json
import os
import random

random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'data', 'scidocs')
META = os.path.join(D, 'paper_metadata_mag_mesh.json')


def read_csv(path):
    with open(path) as f:
        return [(row['pid'], row['class_label']) for row in csv.DictReader(f)]


def main():
    meta = json.load(open(META))
    sub = {}
    for name, split, n in [('mag', 'train', 1500), ('mag', 'test', 1000),
                           ('mesh', 'train', 1000), ('mesh', 'test', 1000)]:
        rows = [(p, c) for p, c in read_csv(os.path.join(D, name, f'{split}.csv'))
                if p in meta]                # 59 篇不在 metadata, 过滤掉
        sub[f'{name}_{"pool" if split == "train" else "query"}'] = dict(
            random.sample(rows, n))

    all_ids = sorted({p for d in sub.values() for p in d})
    json.dump({**sub, 'all_ids': all_ids},
              open(os.path.join(D, 'light_ids.json'), 'w'))
    print(f'mag pool/query = {len(sub["mag_pool"])}/{len(sub["mag_query"])}, '
          f'mesh pool/query = {len(sub["mesh_pool"])}/{len(sub["mesh_query"])}, '
          f'total_unique = {len(all_ids)}')


if __name__ == '__main__':
    main()
