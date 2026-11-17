import os
import sys
import numpy as np
import torch

Root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Bin=os.path.join(Root,'data','raw','scibert_x','pytorch_model.bin')
out_dir=os.path.join(Root,'models','scibert')
out=os.path.join(out_dir,'scibert.npz')

def main():
    sd=torch.load(Bin,map_location='cpu',weights_only=True)
    arrs={}
    n_skip_cls=0
    for k,v in sd.items():
        if k.startswith('cls.'):
            n_skip_cls+=1
            continue
        if k.startswith('bert.'):
            k=k[len('bert.'):]
        arrs[k]=v.detach().numpy().astype(np.float32)
    os.makedirs(out_dir,exist_ok=True)
    np.savez(out,**arrs)
    size=os.path.getsize(out)/1e6
    n_params=sum(a.size for a in arrs.values())/1e6
    print(f'[ok] skipped cls.* : {n_skip_cls} tensors')
    print(f'[ok] saved {len(arrs)} tensors -> {out} ({size:.1f} MB)')
    print(f'[ok] encoder params: {n_params:.2f}M')

    assert arrs['embeddings.word_embeddings.weight'].shape == (31090, 768)
    assert arrs['encoder.layer.0.attention.self.query.weight'].shape == (768, 768)
    assert arrs['encoder.layer.11.output.dense.weight'].shape == (768, 3072)
    assert 'pooler.dense.weight' in arrs
    print('[ok] shape sanity checks passed')
if __name__ == '__main__':
    main()
