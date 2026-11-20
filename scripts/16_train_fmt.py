import json
import math
import os
import sys
import time
import numpy as np
import jittor as jt
from jittor import nn
from transformers import BertTokenizer
jt.flags.use_cuda=0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.bert import BertModel
from src.model.hf_loader import load_npz
WEIGHTS = os.path.join(ROOT, 'models', 'scibert', 'scibert.npz')
VOCAB = os.path.join(ROOT, 'data', 'raw', 'scibert_x', 'vocab.txt')
DATA = os.path.join(ROOT, 'data', 'training', 'fmt_train.jsonl')
OUT = os.path.join(ROOT, 'models', 'scibert', 'scibert-fmt.npz')
CTRL_CLF, CTRL_RGN = 2, 3
N_CLASSES = 19
MAX_LEN = int(os.environ.get('FMT_MAXLEN', '96'))
BATCH = int(os.environ.get('FMT_BATCH', '8'))
STEPS = int(os.environ.get('FMT_STEPS', '60'))
LR = float(os.environ.get('FMT_LR', '3e-2'))
LAMBDA = 1.0
SEED = 42

def insert_ctrl(arr,value):
    col=np.full((arr.shape[0],1),value,dtype=arr.dtype)
    return np.concatenate([arr[:,:1],col,arr[:,1:]],axis=1)[:,:MAX_LEN]
def vec_cos(a,b):
    a=a.astype('float64').ravel()
    b=b.astype('float64').ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def rows_cos(a,b):
    num=(a*b).sum(1)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12
    return float((num/den).mean())
def main():
    rec=[json.loads(l) for l in open(DATA)]
    tok=BertTokenizer(VOCAB,do_lower_case=True)
    model=BertModel(vocab_size=31090)
    load_npz(model,WEIGHTS,verbose=False)
    W=model.embeddings.word_embeddings.weight
    head_clf=nn.Linear(768,N_CLASSES)
    head_rgn=nn.Linear(768,1)
    opt=jt.optim.Adam(
        [W]+list(head_clf.parameters())+list(head_rgn.parameters()),
        lr=LR,weight_decay=0
    )
    keep=np.zeros((31090,1),'float32')
    keep[CTRL_CLF]=1.0
    keep[CTRL_RGN]=1.0
    keep_j=jt.array(keep)
    lr_all=np.array([math.log1p(r['rgn']) for r in rec])
    rgn_mu,rgn_sd=float(lr_all.mean()),float(lr_all.std()+1e-6)
    def enc_batch(batch):
        e=tok([r['title'] for r in batch],[r['abstract'] for r in batch],
              padding='max_length',
              truncation='longest_first',
              max_length=MAX_LEN,
              return_tensors='np')
        ids_c=jt.array(insert_ctrl(e['input_ids'],CTRL_CLF).astype('int32'))
        ids_r=jt.array(insert_ctrl(e['input_ids'],CTRL_RGN).astype('int32'))
        mask = jt.array(insert_ctrl(e['attention_mask'], 1).astype('float32'))
        tt = jt.array(insert_ctrl(e['token_type_ids'], 0).astype('int32'))
        return ids_c,ids_r,mask,tt
    W0=W.numpy()[[CTRL_CLF,CTRL_RGN]].copy()
    print(f'[init] cos(W2,W3) = {vec_cos(W0[0], W0[1]):.4f}', flush=True)
    rng=np.random.default_rng(SEED)
    t0=time.time()
    for step in range(STEPS):
        idx=rng.choice(len(rec),BATCH,replace=False)
        batch=[rec[i] for i in idx]
        ids_c,ids_r,mask,tt=enc_batch(batch)
        hc,_=model(ids_c,mask,tt)
        hr,_=model(ids_r,mask,tt)
        logit=head_clf(hc[:,1])
        pred=head_rgn(hr[:,1])
        y=jt.array(np.array([r['clf'] for r in batch],'int32'))
        tgt=jt.array(np.array(
            [(math.log1p(r['rgn'])-rgn_mu)/rgn_sd for r in batch],
            'float32'
        ).reshape(-1,1))
        loss_c=nn.cross_entropy_loss(logit,y)
        loss_r=nn.mse_loss(pred,tgt)
        loss=loss_c+LAMBDA*loss_r
        opt.backward(loss)
        g=W.opt_grad(opt)
        g.assign(g*keep_j)
        opt.step()
        jt.sync_all()
        if (step+1)%10 == 0 :
            Wc=W.numpy()
            print(f'[{step+1}/{STEPS}] loss_c={loss_c.item():.3f} '
                  f'loss_r={loss_r.item():.3f} '
                  f'cos(W2,W3)={vec_cos(Wc[CTRL_CLF], Wc[CTRL_RGN]):.4f}  '
                  f'{time.time()-t0:.1f}s', flush=True)
    Wf = W.numpy()
    print(f'\n[final] cos(W2,W3) {vec_cos(W0[0], W0[1]):.4f} -> '
          f'{vec_cos(Wf[CTRL_CLF], Wf[CTRL_RGN]):.4f}', flush=True)
    print(f'[final] dW2={np.abs(Wf[CTRL_CLF]-W0[0]).max():.4f} '
          f'dW3={np.abs(Wf[CTRL_RGN]-W0[1]).max():.4f}', flush=True)
    arrs = {k: v.copy() for k, v in np.load(WEIGHTS).items()}
    key = 'embeddings.word_embeddings.weight'
    arrs[key][CTRL_CLF] = Wf[CTRL_CLF]
    arrs[key][CTRL_RGN] = Wf[CTRL_RGN]
    np.savez(OUT, **arrs)
    print(f'[save] {OUT}  keys={len(arrs)}', flush=True)
    from src.model.encoder import PaperEncoder
    probe = [{'title': r['title'], 'abstract': r['abstract']} for r in rec[:8]]
    enc = PaperEncoder(weights=OUT, max_len=MAX_LEN, batch_size=8)
    e_none = enc.encode(probe, fmt=None)
    e_clf = enc.encode(probe, fmt='clf')
    e_rgn = enc.encode(probe, fmt='rgn')
    print(f'\n[verify] cos(None,clf) = {rows_cos(e_none, e_clf):.4f}', flush=True)
    print(f'[verify] cos(None,rgn) = {rows_cos(e_none, e_rgn):.4f}', flush=True)
    print(f'[verify] cos(clf,rgn)  = {rows_cos(e_clf, e_rgn):.4f}  '
          f'(基线 0.9992, 期望明显下降)', flush=True)
    print(f'[time] {time.time()-t0:.1f}s total', flush=True)
if __name__=='__main__':
    main()