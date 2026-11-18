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
    def encode(self,texts,fmt=None,normalize=False):
        if fmt is not None:
            raise NotImplementedError(
                f'fmt={fmt!r}: M5 (SPECTER2 双格式) 尚未实现; '
                f'fmt=None 为当前唯一通路')
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
                ids=jt.array(enc['input_ids'].astype('int32'))
                mask=jt.array(enc['attention_mask'].astype('float32'))
                tt=jt.array(enc['token_type_ids'].astype('int32'))
                h,_pooled=self.model(ids,mask,tt)
                out[s:s+len(chunk)]=h[:,0].numpy()
        if normalize:
            out/=np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-12)
        return out
