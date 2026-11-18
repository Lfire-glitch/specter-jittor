import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data', 'training')
TRIPLETS_FILE = os.path.join(DATA, 'triplets.jsonl')
PAPERS_FILE = os.path.join(DATA, 'papers.json')
RING_FILE = os.path.join(DATA, 'triplets_knn_ring.jsonl')

def _load_citation():
    with open(TRIPLETS_FILE) as f:
        return [json.loads(line) for line in f]


def _random(papers, seed=42):
    rng = random.Random(seed)
    all_ids = list(papers)
    out = []
    for t in _load_citation():
        negs = rng.sample([x for x in all_ids if x != t['q'] and x != t['pos']], 5)
        out.append({'q': t['q'], 'pos': t['pos'], 'negs': negs})
    return out


def build_triplets(mode='citation', papers=None, embeddings=None, **kwargs):
    if mode == 'citation':
        return _load_citation()
    if mode == 'random':
        if papers is None:
            papers = json.load(open(PAPERS_FILE))
        return _random(papers, seed=kwargs.get('seed', 42))
    if mode == 'knn_ring':
        with open(RING_FILE) as f:
            return [json.loads(line) for line in f]
    raise ValueError(f'unknown mode: {mode!r}')