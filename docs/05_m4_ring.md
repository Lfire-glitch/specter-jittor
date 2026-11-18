# M4 / SciNCL 环带采样（任务 ②）

复现粒度：**结构 + 方向对齐**，不刷分。默认全轻量级。

## 1. 定位：只换一个自变量

SPECTER/M2 用引用关系硬划正负：`pos` = 真实引用，`neg` = 随机未引用。负例太容易，
模型几乎不用学细粒度区分。SciNCL 的核心改动是：**用编码器自己的嵌入算 kNN 全排序，
按 rank 带取样**，让正/负例落在可控的相似度区间。

M4 冻结编码器、`margin loss`（L2, margin=1）、kNN 评测尺，只动「采样策略」这一个自变量。
这正是 `build_triplets(mode=...)` 接口存在的意义。

## 2. 环带口径

| 带 | rank（1-indexed，不含 q 自身） | 用途 |
|---|---|---|
| `k+` | `[20, 25]` | 近邻带：语义很近但排除最相邻的伪正例 → 正例来源 |
| 中间 | `(26 ~ 3997)` | **不采**，避免相似度模糊带污染对比信号 |
| `k-` | `[3998, 4000]` | 难负带 → 负例来源（每行 3 个） |

rank 从 q 对全池的 cosine 降序 `np.argsort(-sim[j])` 得到；`ring_indices()` 先显式剔除
q 自身再数 rank（`others[lo-1:hi]`），避免整体偏移 1 位的 off-by-one。

## 3. 轻量化口径与产物

- 检索池：`scibert-tuned.npz` 对 `papers.json` 的 8421 篇重编码，`[CLS]`，
  `max_len=128 / batch=32`，耗时 3052s（0.362 s/篇）。
  → `train_emb.npy (8421,768) float32`、`train_ids.json`，行范数 21.8~24.1（未归一化）。
- q 集合：冻结自 `triplets.jsonl` 的 2758 条，pos/negs 一律不继承。
- 采样脚本：`scripts/15_build_knn_ring.py`（用户手打）→ `triplets_knn_ring.jsonl` 2758 行、
  `logs/m4_ring_stats.json`。
- 接口接入：`src/data/triplets.py` 的 `knn_ring` 分支**读落盘表**，不重写采样过程
  （与 `citation` 对称，追求产物一致而非过程一致）。
- 回归：`scripts/13_regression.py` 增 D 闸 `knn_ring` 读表 == 落盘表，四闸全绿。

## 4. 真实结果（非合成数据）

```
[pool] 8421 papers, emb=(8421, 768)
[q] 2758 queries (frozen from citation table)
[stats] cos: pos_band=0.8857  neg_band=0.6807  citation_neg=0.6763
VERDICT: PASS (方向对齐)
```

| 指标 | 值 |
|---|---|
| rows / negs_per_row | 2758 / 3 |
| `cos_pos_band_mean` | 0.8857 |
| `cos_neg_band_mean` | 0.6807 |
| `cos_citation_neg_mean` | 0.6763 |
| `pos_same_as_citation` | 19 / 2758 |
| `neg_same_as_citation` | 7 / 2758 |

方向对齐成立：`k+` 带显著近于 `k-` 带（0.8857 > 0.6807）。
`pos_same=19`、`neg_same=7` 说明环带表与 citation 表是**真正不同的采样产物**，
接口切换有效，不是把原表抄了一遍。

## 5. 诚实警示（不可略过）

1. **口径失配（最重要）**：原文 `rank 4000` 是在几十万篇池里取的前 1%，是真难负例；
   本池只有 8421 篇，`rank 4000` ≈ 中位数。实证 `cos_neg_band=0.6807` 与
   `cos_citation_neg=0.6763` 几乎相等 —— **难负带几乎没比随机负例更难**。
   所以 VERDICT 只判 `pos_band > neg_band`（这是「按相似度排序取带」的必然结果），
   **不代表复现了 SciNCL 的难负例挖掘增益**。这是轻量化缩放的局限，不是 bug。
2. **各向异性**：SciBERT 原生 `[CLS]` 未做对齐/白化，全池 cosine 基线普遍偏高
   （连 `k-` 带都有 0.68）。绝对 cos 值不可跨数据集、跨模型比较，只看带间相对差。
3. **策略级对比，非逐行控制变量**：`KEEP_CITATION_POS=False`，pos 也从 `k+` 带采，
   pos 集合整体改变（与 citation 仅 19 行巧合相同）。冻结的是 q 集合、数据规模、
   编码器、margin loss、评测尺；若需严格逐行对照，把该开关改 `True` 即可只换 negs。
4. **检索池口径 ≠ 训练/评测口径**：`max_len=128` 仅为算 rank 的中间量，
   M2 训练是 256、SciDocs 评测是 512。它不参与 `02/03` 任何归档数字。
5. **未做下游训练对照**：本阶段只产出采样表 + 方向对齐证据，
   **没有**用 `triplets_knn_ring.jsonl` 再跑一轮 margin 训练 + 同尺 kNN lift。
   按「默认全轻量级 / 重点是技术复现而非验证已有轮子」的纪律，是否补这轮由用户拍板。

## 6. 复现命令

```bash
export cc_path=/usr/bin/g++-12
python scripts/14_encode_train_pool.py     # -> train_emb.npy / train_ids.json
python scripts/15_build_knn_ring.py        # -> triplets_knn_ring.jsonl / m4_ring_stats.json
python scripts/13_regression.py            # -> A / A2 / B / C /D 四闸 PASS
```

## 7. 收口判断

② 的**采样机制**（环带构造、接口接入、方向对齐证据、回归闸）已完整。
剩下唯一开放项是第 5.5 条：要不要再跑一轮「knn_ring 训练 → 1200 篇重编码 → 同尺 lift 对照」。
按既定纪律，这一步属于「再验证一遍轮子」，可不做；若要把它作为 M4 的完整对照实验，
需用户明确开工。
