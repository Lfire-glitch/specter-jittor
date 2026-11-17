"""M2d: 用 tuned 权重重编码一个 1,200 篇的小评测子集 (轻量闭环).

子集 = light_ids 的 mag/mesh 各取 pool 前 300 + query 前 300 (seed 已固定的抽样序).
基线侧同一批论文直接从已有 scibert-embeddings-light.npy 取行, 无需重算.
输出: data/scidocs/scibert-embeddings-sub-tuned.npy + 对应 ids
"""
import json
import os
import sys
import time

import numpy as np
import jittor as jt
from transformers import BertTokenizer

jt.flags.use_cuda = 0
os.environ['HF_HUB_OFFLINE'] = '1'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model.bert import BertModel as BertModelJt
from src.model.hf_loader import load_npz

META  = os.path.join(ROOT, 'data', 'scidocs', 'paper_metadata_mag_mesh.json')
LIGHT = os.path.join(ROOT, 'data', 'scidocs', 'light_ids.json')
VOCAB = os.path.join(ROOT, 'data', 'raw', 'scibert_x', 'vocab.txt')
NPZ   = os.environ.get('ENCODE_NPZ',
                       os.path.join(ROOT, 'models', 'scibert', 'scibert-tuned.npz'))
OUT   = os.path.join(ROOT, 'data', 'scidocs', 'scibert-embeddings-sub-tuned.npy')
IDS   = os.path.join(ROOT, 'data', 'scidocs', 'scibert-paper_ids-sub-tuned.json')
CKPT  = os.path.join(ROOT, 'data', 'scidocs', 'encode-sub-tuned.ckpt.npz')

N_EACH = 300
BATCH = 16
MAX_LEN = 512


def main():
    light = json.load(open(LIGHT))
    pids = sorted(list(light['mag_pool'])[:N_EACH] + list(light['mag_query'])[:N_EACH]
                  + list(light['mesh_pool'])[:N_EACH] + list(light['mesh_query'])[:N_EACH])
    meta = json.load(open(META))
    print(f'[data] {len(pids)} papers, weights={os.path.basename(NPZ)}', flush=True)

    start = 0
    emb = np.zeros((len(pids), 768), dtype=np.float32)
    if os.path.exists(CKPT):
        z = np.load(CKPT, allow_pickle=True)
        saved = z['ids'].tolist()
        if saved == pids[:len(saved)]:
            emb[:len(saved)] = z['emb']
            start = len(saved)
            print(f'[resume] from {start}', flush=True)

    tok = BertTokenizer(VOCAB, do_lower_case=True)
    model = BertModelJt(vocab_size=31090)
    load_npz(model, NPZ, verbose=False)
    model.eval()

    t0 = time.time()
    with jt.no_grad():
        for s in range(start, len(pids), BATCH):
            batch = pids[s:s + BATCH]
            titles = [meta[p]['title'] or '' for p in batch]
            abstracts = [meta[p]['abstract'] or '' for p in batch]
            enc = tok(titles, abstracts, padding=True,
                      truncation='only_second', max_length=MAX_LEN,
                      return_tensors='np')
            ids  = jt.array(enc['input_ids'].astype('int32'))
            mask = jt.array(enc['attention_mask'].astype('float32'))
            tt   = jt.array(enc['token_type_ids'].astype('int32'))
            h, _ = model(ids, mask, tt)
            emb[s:s + len(batch)] = h[:, 0].numpy()
            done = s + len(batch)
            if done % 160 == 0 or done == len(pids):
                dt = time.time() - t0
                print(f'[enc] {done}/{len(pids)}  {dt:.0f}s', flush=True)
            if done % 400 < BATCH:
                np.savez(CKPT, ids=np.array(pids[:done], dtype=object),
                         emb=emb[:done])

    np.save(OUT, emb)
    json.dump(pids, open(IDS, 'w'))
    if os.path.exists(CKPT):
        os.remove(CKPT)
    print(f'[ok] {emb.shape} -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
