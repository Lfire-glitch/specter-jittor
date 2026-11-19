import os
import numpy as np
import jittor as jt
from transformers import BertTokenizer
from .bert import BertModel
from .hf_loader import load_npz

jt.flags.use_cuda=0

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS = os.path.join(ROOT, 'models', 'scibert', 'scibert.npz')
DEFAULT_VOCAB = os.path.join(ROOT, 'data', 'raw', 'scibert_x', 'vocab.txt')
CTRL_IDS = {'clf': 2, 'rgn': 3}

class PaperEncoder:
    def __init__(self,weights=DEFAULT_WEIGHTS,vocab=DEFAULT_VOCAB,max_len=512,batch_size=16):
        self.tok=BertTokenizer(vocab,do_lower_case=True)
        self.model=BertModel(vocab_size=31090)
        load_npz(self.model,weights,verbose=False)
        self.model.eval()
        self.max_len=max_len
        self.batch_size=batch_size
    @staticmethod
    def _as_pairs(texts):
        pairs=[]
        for t in texts:
            if isinstance(t,str):
                pairs.append((t,''))
            elif isinstance(t,dict):
                pairs.append((t.get('title') or '',t.get('abstract') or ''))
            else:
                pairs.append((t[0] or '',t[1] or ''))
        return pairs
    def _insert_ctrl(self, arr, value):
        col = np.full((arr.shape[0], 1), value, dtype=arr.dtype)
        out = np.concatenate([arr[:, :1], col, arr[:, 1:]], axis=1)
        return out[:, :self.max_len]
    def encode(self,texts,fmt=None,normalize=False):
        if fmt is not None and fmt not in CTRL_IDS:
            raise ValueError(
                f'unknown fmt: {fmt!r}, expected None or one of {sorted(CTRL_IDS)}')
        pairs=self._as_pairs(texts)
        out=np.zeros((len(pairs),768),dtype=np.float32)

        with jt.no_grad():
            for s in range(0,len(pairs),self.batch_size):
                chunk=pairs[s:s+self.batch_size]
                titles=[t for t,_ in chunk]
                abstracts=[a for _,a in chunk]
                enc=self.tok(titles,abstracts,padding=True,
                truncation='only_second',
                max_length=self.max_len,
                return_tensors='np')
                if fmt is None:
                    ids_np = enc['input_ids']
                    mask_np = enc['attention_mask']
                    tt_np = enc['token_type_ids']
                else:
                    ids_np = self._insert_ctrl(enc['input_ids'], CTRL_IDS[fmt])
                    mask_np = self._insert_ctrl(enc['attention_mask'], 1)
                    tt_np = self._insert_ctrl(enc['token_type_ids'], 0)
                ids=jt.array(ids_np.astype('int32'))
                mask=jt.array(mask_np.astype('float32'))
                tt=jt.array(tt_np.astype('int32'))
                h,_pooled=self.model(ids,mask,tt)
                out[s:s+len(chunk)]=h[:, 1 if fmt is not None else 0].numpy()
        if normalize:
            out/=np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-12)
        return out
