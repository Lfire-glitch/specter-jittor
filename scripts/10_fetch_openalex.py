"""M2b: 从 OpenAlex 抓取并构造轻量引文三元组 (官方 16GB train.pkl 的轻量替代).

三元组规则 (与 SPECTER 训练数据同配方):
  q   : CS 领域查询论文 (有摘要, >=3 条参考文献, 英文)
  pos : q 真实引用且同样有摘要的论文 (引文监督信号的来源)
  negs: 同领域、q 未引用的 5 篇 (从查询池内部采样, 保证有文本且领域一致)

产出:
  data/training/papers.json    {W_id: {title, abstract}}
  data/training/triplets.jsonl 每行 {q, pos, negs:[...5]}
"""
import json
import os
import random
import time
import urllib.parse
import urllib.request

random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'data', 'training')
API = 'https://api.openalex.org/works'
MAILTO = '13263373980@163.com'
N_QUERIES = 3000          # 轻量: 3000 个查询三元组
N_NEG = 5
MIN_REFS = 3


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': f'mailto:{MAILTO}'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            print(f'[retry {attempt}] {e}', flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'api failed: {url[:120]}')


def inv_to_text(inv):
    """abstract_inverted_index {词: [位置,...]} -> 还原成正常文本."""
    if not inv:
        return ''
    flat = [(p, w) for w, ps in inv.items() for p in ps]
    return ' '.join(w for _, w in sorted(flat))


def is_english(s):
    if not s:
        return False
    ascii_ratio = sum(c.isascii() for c in s) / len(s)
    return ascii_ratio > 0.95


def fetch_sample(seed, n=25):
    """OpenAlex 随机采样: CS 领域 2018+ 英文 article."""
    q = urllib.parse.urlencode({
        'filter': 'concepts.id:C41008148,publication_year:2018-2024,'
                  'type:article,language:en',
        'sample': n, 'seed': seed, 'mailto': MAILTO,
        'select': 'id,title,abstract_inverted_index,referenced_works',
    })
    return get(f'{API}?{q}')['results']


def fetch_by_ids(ids):
    """按 id 批量取论文 (pos 候选的文本), 每次最多 50 个 id."""
    q = urllib.parse.urlencode({
        'filter': 'ids.openalex:' + '|'.join(ids),
        'per-page': 50, 'mailto': MAILTO,
        'select': 'id,title,abstract_inverted_index',
    })
    return get(f'{API}?{q}')['results']


def paper_record(w):
    wid = w['id'].rsplit('/', 1)[-1]
    return wid, {'title': w.get('title') or '',
                 'abstract': inv_to_text(w.get('abstract_inverted_index'))}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    papers, queries = {}, []

    # ---- 1. 采查询池: 多次随机采样直到收集够合格查询 ----
    seed = 0
    while len(queries) < N_QUERIES and seed < 2000:
        seed += 1
        for w in fetch_sample(seed):
            refs = w.get('referenced_works') or []
            wid, rec = paper_record(w)
            if (len(refs) >= MIN_REFS and rec['abstract']
                    and 30 <= len(rec['abstract'].split()) <= 400
                    and is_english(rec['title'])):
                papers[wid] = rec
                queries.append({'id': wid, 'refs': [r.rsplit('/', 1)[-1]
                                                    for r in refs]})
        if seed % 20 == 0:
            print(f'[sample] seeds={seed} queries={len(queries)}', flush=True)
    queries = queries[:N_QUERIES]
    print(f'[sample] queries={len(queries)}', flush=True)

    # ---- 2. 批量抓 pos 候选文本: 每个查询取前 3 个引用, 择优录用 ----
    cand = {}
    for q in queries:
        for r in q['refs'][:3]:
            cand.setdefault(r, []).append(q['id'])
    cand_ids = list(cand)
    pos_pool = {}
    for i in range(0, len(cand_ids), 50):
        for w in fetch_by_ids(cand_ids[i:i + 50]):
            wid, rec = paper_record(w)
            if rec['abstract'] and 20 <= len(rec['abstract'].split()) <= 400:
                pos_pool[wid] = rec
        if (i // 50) % 10 == 0:
            print(f'[pos] {i}/{len(cand_ids)} fetched, usable={len(pos_pool)}',
                  flush=True)
    papers.update(pos_pool)

    # ---- 3. 组三元组: pos=首个有文本的被引文献, negs=查询池内随机未引 ----
    qids = [q['id'] for q in queries]
    triplets, skipped = [], 0
    for q in queries:
        pos = next((r for r in q['refs'] if r in pos_pool), None)
        if pos is None:
            skipped += 1
            continue
        refset = set(q['refs']) | {q['id']}
        negs = random.sample([x for x in qids if x not in refset], N_NEG)
        triplets.append({'q': q['id'], 'pos': pos, 'negs': negs})
    print(f'[triplets] built={len(triplets)} skipped_no_pos={skipped} '
          f'papers={len(papers)}', flush=True)

    with open(os.path.join(OUT_DIR, 'papers.json'), 'w') as f:
        json.dump(papers, f)
    with open(os.path.join(OUT_DIR, 'triplets.jsonl'), 'w') as f:
        for t in triplets:
            f.write(json.dumps(t) + '\n')
    print(f'[saved] {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
