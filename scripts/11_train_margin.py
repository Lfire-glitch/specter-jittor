import json
import os
import random
import sys
import time
import numpy as np
import jittor as jt
from transformers import BertTokenizer

jt.flags.use_cuda=0
os.environ['HF_HUB_OFFLINE']='1'
random.seed(42)
np.random.seed(42)
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
from src.model.bert import BertModel as BertModelJt
from src.model.hf_loader import load_npz, build_mapping
DATA  = os.path.join(ROOT, 'data', 'training')
VOCAB = os.path.join(ROOT, 'data', 'raw', 'scibert_x', 'vocab.txt')
NPZ   = os.path.join(ROOT, 'models', 'scibert', 'scibert.npz')
OUT   = os.path.join(ROOT, 'models', 'scibert', 'scibert-tuned.npz')
LOG   = os.path.join(ROOT, 'logs', 'm2_train.json')
Steps=600
Triples=4
lr=2e-5
margin=1.0
max_len=256
def encode(model,tok,papers,pids):
    titles=[papers[p]['title'] for p in pids]
    abstracts=[papers[p]['abstract'] for p in pids]
    enc=tok(titles,abstracts,padding=True,truncation='only_second',max_length=max_len,return_tensors='np')
    ids=jt.array(enc['input_ids'].astype('int32'))
    mask=jt.array(enc['attention_mask'].astype('float32'))
    tt=jt.array(enc['token_type_ids'].astype('int32'))
    h,_pooled=model(ids,mask,tt)
    return h[:,0]
def main():
    papers=json.load(open(os.path.join(DATA,'papers.json')))
    triplets=[json.loads(l) for l in open(os.path.join(DATA,'triplets.jsonl'))]
    print(f'[data] {len(triplets)} triplets, {len(papers)} papers', flush=True)
    tok = BertTokenizer(VOCAB, do_lower_case=True)
    model = BertModelJt(vocab_size=31090)
    load_npz(model, NPZ, verbose=False)
    model.train()      
    opt = jt.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist=[]
    t0=time.time()
    for step in range(1,Steps+1):
        batch=random.sample(triplets,Triples)
        qs=[t['q'] for t in batch]
        ps=[t['pos'] for t in batch]
        ns=[random.choice(t['negs']) for t in batch]
        emb=encode(model,tok,papers,qs+ps+ns)
        eq,ep,en=emb[:Triples],emb[Triples:2*Triples],emb[2*Triples:]
        d_pos=((eq-ep)**2).sum(1)**0.5
        d_neg=((eq-en)**2).sum(1)**0.5
        loss=jt.nn.relu(margin+d_pos-d_neg).mean()
        opt.backward(loss)
        opt.clip_grad_norm(1.0)
        opt.step()

        # 周期性落盘: 防 VM 崩溃丢失全部训练
        if step % 150 == 0:
            mp = build_mapping(model)
            np.savez(OUT, **{k: np.ascontiguousarray(v.numpy(), dtype=np.float32)
                             for k, v in mp.items()})
            json.dump(hist, open(LOG, 'w'), indent=2)

        if step%10==0 or step==1:
            lp, ln = d_pos.numpy().mean(), d_neg.numpy().mean()
            sat = float((d_pos < d_neg).numpy().mean())   # 满足率
            hist.append({'step': step, 'loss': float(loss.item()),
                         'd_pos': float(lp), 'd_neg': float(ln), 'sat': sat})
            print(f'[{step}/{Steps}] loss={loss.item():.3f} '
                  f'd_pos={lp:.2f} d_neg={ln:.2f} sat={sat:.2f} '
                  f'({(time.time()-t0)/step:.2f}s/step)', flush=True)
    mp = build_mapping(model)
    np.savez(OUT, **{k: np.ascontiguousarray(v.numpy(), dtype=np.float32)
                    for k, v in mp.items()})
    json.dump(hist, open(LOG, 'w'), indent=2)
    print(f'[ok] tuned weights -> {OUT}', flush=True)
    print(f'[ok] history -> {LOG}', flush=True)


if __name__ == '__main__':
    main()
