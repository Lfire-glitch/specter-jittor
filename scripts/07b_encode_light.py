"""Step S2-light v2: 编码 light_ids.json 清单 (CPU 轻量验证, 支持断点续跑).

v2 新增:
  - 每 CKPT_EVERY 篇落盘 .ckpt.npz, 崩溃后自动从断点继续
  - ENCODE_NPZ 环境变量: 换用其他权重 (如 M2 训练后的), 实现"同集重编码"
  - ENCODE_TAG 环境变量: 输出文件名加后缀, 不覆盖基线产物
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

BATCH = 16
MAX_LEN = 512
CKPT_EVERY = 800
WEIGHTS = os.environ.get('ENCODE_NPZ', NPZ)
OUT_TAG = os.environ.get('ENCODE_TAG', '')


def main():
    D = os.path.join(ROOT, 'data', 'scidocs')
    emb_out = os.path.join(D, f'scibert-embeddings-light{OUT_TAG}.npy')
    ids_out = os.path.join(D, f'scibert-paper_ids-light{OUT_TAG}.json')
    ckpt = os.path.join(D, f'encode-light{OUT_TAG}.ckpt.npz')

    meta = json.load(open(META))
    pids = json.load(open(LIGHT))['all_ids']
    print(f'[data] {len(pids)} papers, weights={os.path.basename(WEIGHTS)}',
          flush=True)

    # ---- 断点恢复 ----
    start = 0
    emb = np.zeros((len(pids), 768), dtype=np.float32)
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        saved = z['ids'].tolist()
        if saved == pids[:len(saved)]:
            emb[:len(saved)] = z['emb']
            start = len(saved)
            print(f'[resume] from {start}/{len(pids)}', flush=True)
        else:
            print('[resume] ckpt id 不匹配, 从头开始', flush=True)

    tok = BertTokenizer(VOCAB, do_lower_case=True)
    model = BertModelJt(vocab_size=31090)
    load_npz(model, WEIGHTS, verbose=False)
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

            if done % (BATCH * 100) == 0 or done == len(pids):
                dt = time.time() - t0
                eta = (len(pids) - done) / max(done / dt, 1e-9)
                print(f'[enc] {done}/{len(pids)}  {dt:.0f}s  '
                      f'eta {eta/60:.0f}min', flush=True)
            if done % CKPT_EVERY < BATCH:
                np.savez(ckpt, ids=np.array(pids[:done], dtype=object),
                         emb=emb[:done])

    np.save(emb_out, emb)
    json.dump(pids, open(ids_out, 'w'))
    if os.path.exists(ckpt):
        os.remove(ckpt)                       # 全量完成后清掉断点文件
    print(f'[ok] embeddings {emb.shape} -> {emb_out}', flush=True)


if __name__ == '__main__':
    main()
