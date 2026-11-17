import json
import os
import sys
import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer
import jittor as jt
jt.flags.use_cuda = 0
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CFG   = os.path.join(ROOT, 'data', 'raw', 'scibert_x', 'bert_config.json')
BIN   = os.path.join(ROOT, 'data', 'raw', 'scibert_x', 'pytorch_model.bin')
NPZ   = os.path.join(ROOT, 'models', 'scibert', 'scibert.npz')
VOCAB = os.path.join(ROOT, 'data', 'raw', 'scibert_x', 'vocab.txt')
LOG   = os.path.join(ROOT, 'logs', 'scibert_align.json')
def cosine(a,b):
    a=a.ravel().astype(np.float64)
    b=b.ravel().astype(np.float64)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def main():
    cfg =json.load(open(CFG))
    jm=BertModelJt(
        vocab_size=cfg['vocab_size'], hidden=cfg['hidden_size'],
        layers=cfg['num_hidden_layers'], heads=cfg['num_attention_heads'],
        intermediate=cfg['intermediate_size'],
    )
    from src.model.hf_loader import load_npz
    load_npz(jm,NPZ)
    jm.eval()

    ref_cfg = BertConfig(
        vocab_size=cfg['vocab_size'], hidden_size=cfg['hidden_size'],
        num_hidden_layers=cfg['num_hidden_layers'],
        num_attention_heads=cfg['num_attention_heads'],
        intermediate_size=cfg['intermediate_size'],
        layer_norm_eps=1e-12,
    )
    ref = BertModel(ref_cfg)
    sd = torch.load(BIN, map_location='cpu', weights_only=True)
    sd = {k[len('bert.'):]: v for k, v in sd.items() if k.startswith('bert.')}
    ref.load_state_dict(sd, strict=True) 
    ref.eval()

    tok=BertTokenizer(VOCAB,do_lower_case=True)
    texts = [
        'we propose a neural model for citation intent classification.',
        'gene expression in breast cancer cells was measured by rna-seq.',
    ]
    enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=64)
    ids = enc['input_ids'].numpy().astype('int32')
    mask = enc['attention_mask'].numpy().astype('float32')
    with torch.no_grad():
        out=ref(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask'],
            token_type_ids=enc['token_type_ids'],
            output_hidden_states=True,
        )
    th_hiddens=out.hidden_states
    with jt.no_grad():
        jh,jp,jt_hiddens=jm.execute(
            jt.array(ids),jt.array(mask),return_hidden=True)
    report = {'layers': []}
    print(f'{"layer":>5} {"cos":>10} {"max_abs":>10}')
    for i, (ta, ja) in enumerate(zip(th_hiddens, jt_hiddens)):
        ta = ta.numpy()
        ja = ja.numpy()
        c = cosine(ta, ja)
        m = float(np.abs(ta - ja).max())
        report['layers'].append({'layer': i, 'cos': c, 'max_abs': m})
        print(f'{i:>5} {c:>10.8f} {m:>10.2e}')

    ok = all(l['cos'] >= 0.9999 for l in report['layers'])
    report['verdict'] = 'PASS' if ok else 'FAIL'
    print(f'VERDICT: {report["verdict"]}')

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    json.dump(report, open(LOG, 'w'), indent=2)
    print(f'[saved] {LOG}')

from src.model.bert import BertModel as BertModelJt

if __name__ == '__main__':
    main()