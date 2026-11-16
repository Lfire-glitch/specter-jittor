"""M1 gate: numerically align the Jittor BERT against the PyTorch/HF reference.

Pass criteria (from 00_复现对接_Jittor.md):
  - every layer hidden state cosine > 0.9999
  - final [CLS] cosine > 0.999
No training is involved here; PyTorch is used only as a reference forward.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402  (reference only)
from transformers import AutoModel, AutoTokenizer  # noqa: E402

import jittor as jt  # noqa: E402
from src.model.bert import BertModel  # noqa: E402
from src.model.hf_loader import load_npz  # noqa: E402

jt.flags.use_cuda = 0
MDL = os.path.join(ROOT, 'models', 'specter')

TEXTS = [
    'SciBERT: A Pretrained Language Model for Scientific Text',
    'SPECTER: Document-level Representation Learning using Citation-informed Transformers. '
    'We propose a new method for learning document representations that leverages citation information.',
    'Attention Is All You Need',
]


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    tok = AutoTokenizer.from_pretrained(MDL, local_files_only=True)
    enc = tok(TEXTS, padding=True, truncation=True, max_length=64, return_tensors='np')
    ids = enc['input_ids'].astype('int32')
    mask = enc['attention_mask'].astype('float32')
    print(f'[input] shape={ids.shape} tokens/seq={mask.sum(1).tolist()}')

    # ---------- reference ----------
    ref = AutoModel.from_pretrained(MDL, local_files_only=True).eval()
    with torch.no_grad():
        out = ref(torch.from_numpy(ids.astype('int64')),
                  attention_mask=torch.from_numpy(mask.astype('int64')),
                  output_hidden_states=True)
    th_hidden = [h.numpy() for h in out.hidden_states]
    th_pool = out.pooler_output.numpy()
    print(f'[torch] hidden_states={len(th_hidden)} last={th_hidden[-1].shape}')

    # ---------- jittor ----------
    model = BertModel(vocab_size=31116)
    load_npz(model, os.path.join(MDL, 'specter.npz'))
    model.eval()
    with jt.no_grad():
        _, pooled, hiddens = model(jt.array(ids), jt.array(mask), return_hidden=True)
        jt.sync_all()
        jh = [h.numpy() for h in hiddens]
        jpool = pooled.numpy()
    print(f'[jittor] hidden_states={len(jh)} last={jh[-1].shape}')

    # ---------- compare ----------
    rows = []
    for i, (a, b) in enumerate(zip(jh, th_hidden)):
        rows.append({'layer': i, 'cos': cos(a, b), 'max_abs': float(np.abs(a - b).max())})
    cls_cos = cos(jpool, th_pool)

    print('\n layer        cos        max_abs')
    for r in rows:
        print(f' {r["layer"]:>5}   {r["cos"]:.8f}   {r["max_abs"]:.2e}')
    print(f'\n [CLS] pooled cosine = {cls_cos:.8f}')

    ok = all(r['cos'] > 0.9999 for r in rows) and cls_cos > 0.999
    print(f'\n M1 {"PASS" if ok else "FAIL"}  (layers>0.9999, cls>0.999)')

    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    with open(os.path.join(ROOT, 'logs', 'm1_align.json'), 'w') as f:
        json.dump({'layers': rows, 'cls_cos': cls_cos, 'pass': ok}, f, indent=2)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
