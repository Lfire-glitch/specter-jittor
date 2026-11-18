"""M4 工程样板（助手跑，不需手打）：用 tuned 权重把 8421 篇训练语料编码成 kNN 检索池嵌入。

产出（data/training/ 下，.gitignore 已排除，不入库）：
    train_emb.npy    (N, 768) float32
    train_ids.json   长度 N 的有序 id 列表，与 emb 行一一对应
断点：每 800 篇写 encode-train-pool.ckpt.npz，WSL 崩溃后重跑自动 resume。

与 07c 的差异：语料换成 OpenAlex 训练语料 papers.json；权重换成 scibert-tuned.npz；
max_len 默认 128（邻域次序对截断长度不敏感，换取 CPU 速度）。
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
from src.model.bert import BertModel
from src.model.hf_loader import load_npz

DATA = os.path.join(ROOT, 'data', 'training')
VOCAB = os.path.join(ROOT, 'data', 'raw', 'scibert_x', 'vocab.txt')
NPZ = os.path.join(ROOT, 'models', 'scibert', 'scibert-tuned.npz')
EMB = os.path.join(DATA, 'train_emb.npy')
IDS_F = os.path.join(DATA, 'train_ids.json')
CKPT = os.path.join(DATA, 'encode-train-pool.ckpt.npz')

MAX_LEN = int(os.environ.get('POOL_MAX_LEN', '128'))
BATCH = int(os.environ.get('POOL_BATCH', '32'))
LIMIT = int(os.environ.get('POOL_LIMIT', '0'))  # 0 = 全量 8421；冒烟用 200
SAVE_EVERY = 800


def main():
    papers = json.load(open(os.path.join(DATA, 'papers.json')))
    pids = sorted(papers)
    if LIMIT:
        pids = pids[:LIMIT]
    print(f'[data] {len(pids)} papers  max_len={MAX_LEN} batch={BATCH}', flush=True)

    emb = np.zeros((len(pids), 768), dtype=np.float32)
    start = 0
    if os.path.exists(CKPT) and not LIMIT:
        z = np.load(CKPT, allow_pickle=True)
        saved = z['ids'].tolist()
        if saved == pids[:len(saved)]:
            emb[:len(saved)] = z['emb']
            start = len(saved)
            print(f'[resume] from {start}', flush=True)

    tok = BertTokenizer(VOCAB, do_lower_case=True)
    model = BertModel(vocab_size=31090)
    load_npz(model, NPZ, verbose=False)
    model.eval()

    t0 = time.time()
    with jt.no_grad():
        for s in range(start, len(pids), BATCH):
            chunk = pids[s:s + BATCH]
            titles = [papers[p].get('title') or '' for p in chunk]
            abstracts = [papers[p].get('abstract') or '' for p in chunk]
            # longest_first：允许截两段中较长的那段。only_second 只截 abstract，
            # 遇到「title 本身就顶到 max_length + abstract 很短」的样本会抛
            # "Sequence to truncate too short"。检索池是全池自洽的中间量，
            # 不参与任何归档数字，统一用标准策略即可。
            enc = tok(titles, abstracts, padding=True,
                      truncation='longest_first', max_length=MAX_LEN,
                      return_tensors='np')
            ids = jt.array(enc['input_ids'].astype('int32'))
            mask = jt.array(enc['attention_mask'].astype('float32'))
            tt = jt.array(enc['token_type_ids'].astype('int32'))
            h, _ = model(ids, mask, tt)
            emb[s:s + len(chunk)] = h[:, 0].numpy()

            done = s + len(chunk)
            if not LIMIT and (done % SAVE_EVERY < BATCH or done == len(pids)):
                np.savez(CKPT, ids=np.array(pids[:done], dtype=object),
                         emb=emb[:done])
            if done % 400 < BATCH or done == len(pids):
                el = time.time() - t0
                rate = max(done - start, 1)
                print(f'[{done}/{len(pids)}] {el:.1f}s  {el / rate:.3f}s/paper  '
                      f'eta={el / rate * (len(pids) - done) / 60:.1f}min', flush=True)

    if not LIMIT:
        np.save(EMB, emb)
        json.dump(pids, open(IDS_F, 'w'))
        print(f'[ok] {EMB} {emb.shape}  {time.time() - t0:.1f}s', flush=True)
    else:
        print(f'[smoke] {len(pids)} papers in {time.time() - t0:.1f}s '
              f'({(time.time() - t0) / len(pids):.3f}s/paper)', flush=True)


if __name__ == '__main__':
    main()
