"""Sanity check: does our Jittor encoder reproduce the OFFICIAL SPECTER embeddings
shipped with SciDocs (data/scidocs/specter-embeddings/cls.jsonl)?

This is stronger than the 3-sentence M1 check: real papers, real titles+abstracts,
and a reference produced by the author's own AllenNLP model.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jittor as jt  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from src.model.bert import BertModel  # noqa: E402
from src.model.hf_loader import load_npz  # noqa: E402

jt.flags.use_cuda = 0
MDL = os.path.join(ROOT, 'models', 'specter')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def load_meta():
    p = os.path.join(ROOT, 'data', 'scidocs', 'paper_metadata_mag_mesh.json')
    with open(p) as f:
        return json.load(f)


def main():
    meta = load_meta()
    recs = []
    with open(os.path.join(ROOT, 'data', 'scidocs', 'specter-embeddings', 'cls.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            m = meta.get(r['paper_id'])
            if not m:
                continue
            text = r['title']
            if m.get('abstract'):
                text = text + ' [SEP] ' + m['abstract']
            recs.append((r['paper_id'], text, np.array(r['embedding'], dtype=np.float32)))
            if len(recs) >= N:
                break
    print(f'[data] {len(recs)} papers')

    tok = AutoTokenizer.from_pretrained(MDL, local_files_only=True)
    model = BertModel(vocab_size=31116)
    load_npz(model, os.path.join(MDL, 'specter.npz'), verbose=False)
    model.eval()

    ours = []
    for i in range(0, len(recs), 8):
        batch = recs[i:i + 8]
        enc = tok([b[1] for b in batch], padding=True, truncation=True,
                  max_length=512, return_tensors='np')
        with jt.no_grad():
            _, pooled = model(jt.array(enc['input_ids'].astype('int32')),
                              jt.array(enc['attention_mask'].astype('float32')))
            jt.sync_all()
            ours.extend(list(pooled.numpy()))

    def cos(a, b):
        a = a.reshape(-1).astype(np.float64)
        b = b.reshape(-1).astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cs = [cos(o, r[2]) for o, r in zip(ours, recs)]
    cs = np.array(cs)
    print(f'[cos vs official] mean={cs.mean():.4f} min={cs.min():.4f} p10={np.percentile(cs,10):.4f} max={cs.max():.4f}')
    print('[note] official cls.jsonl is from the AllenNLP checkpoint (scibert_scivocab_uncased);')
    print('       HF allenai/specter is scivocab_cased -> a systematic gap is expected.')


if __name__ == '__main__':
    main()
