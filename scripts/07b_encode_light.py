"""Step S2-light: 只编码 light_ids.json 清单里的 5449 篇 (CPU 轻量验证).

与 07 相比只有两处不同:
  1. pids 来自 light_ids.json 的 all_ids (而不是全部 metadata 的 key)
  2. 输出文件名带 -light 后缀, 不会覆盖全量产物
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
NPZ   = os.path.join(ROOT, 'models', 'scibert', 'scibert.npz')
EMB_OUT = os.path.join(ROOT, 'data', 'scidocs', 'scibert-embeddings-light.npy')
IDS_OUT = os.path.join(ROOT, 'data', 'scidocs', 'scibert-paper_ids-light.json')

BATCH = 16
MAX_LEN = 512


def main():
    meta = json.load(open(META))
    pids = json.load(open(LIGHT))['all_ids']
    print(f'[data] {len(pids)} papers to encode')

    tok = BertTokenizer(VOCAB, do_lower_case=True)
    model = BertModelJt(vocab_size=31090)
    load_npz(model, NPZ, verbose=False)
    model.eval()

    emb = np.zeros((len(pids), 768), dtype=np.float32)
    t0 = time.time()
    with jt.no_grad():
        for s in range(0, len(pids), BATCH):
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
            if done % (BATCH * 100) == 0 or done == len(pids):
                dt = time.time() - t0
                eta = (len(pids) - done) / max(done / dt, 1e-9)
                print(f'[enc] {done}/{len(pids)}  {dt:.0f}s  '
                      f'eta {eta/60:.0f}min', flush=True)

    np.save(EMB_OUT, emb)
    json.dump(pids, open(IDS_OUT, 'w'))
    print(f'[ok] embeddings {emb.shape} -> {EMB_OUT}')


if __name__ == '__main__':
    main()
