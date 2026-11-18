# 04 · 三个可替换接口（2026-11-18，里程碑 ①）

依据 `00_复现对接_Jittor.md` 第七节。目标不是"能跑"，而是让 M4/M5 的对照实验
**只改一个参数**。本里程碑是纯重构，不改任何数字。

## 接口契约

| 接口 | 位置 | 签名 | 换的变量 | 冻结的变量 |
|---|---|---|---|---|
| 1 嵌入导出 | `src/model/encoder.py` | `encode(texts, fmt=None, normalize=False) -> np.ndarray[n,768]` | 权重（构造参数） | 编码器/分词/取 [CLS] |
| 2 采样策略 | `src/data/triplets.py` | `build_triplets(mode='citation'/'random'/'knn_ring') -> list[dict]` | 采样策略 | q 集合、数据规模 |
| 3 表示形态 | `src/model/encoder.py` | `encode(..., fmt=None/'clf'/'rgn')` | 表示形态 | 编码器、训练数据 |

## 四条边界决策（回归纪律）

1. **默认不 L2 归一化**（`normalize=False`）。交接文档写"输出 L2 归一化"，但 M2 的
   margin loss 用 L2 距离、评测尺内部自做 cosine 归一化——若 encode 强行归一化，
   `02/03` 归档数字全变。留 `normalize=True` 开关，默认关闭。
2. **`fmt` 今日只立形状**：`fmt is not None` 抛 `NotImplementedError`，注释标明 M5
   两个扩展点（control code 拼接位 / adapter 路由位）。不假装实现。
3. **训练侧 encode 不合并**：`11_train_margin.py` 需梯度 + `train()` 模式（dropout 开），
   `PaperEncoder` 恒 `eval()` + `no_grad()`。同一模型、不同执行模式，硬合并会破坏 M2 数字。
4. **`build_triplets('citation')` 读落盘表**，不重写 `10` 的采样逻辑。
   追求"产物一致"而非"过程一致"：生成逻辑挂着 `random` 全局状态、OpenAlex 返回顺序、
   242 篇过滤等隐含随机性，重跑一遍得到的会是另一张表，M2 即不可复现。
   读表 = 引用同一个数据对象，保证单一事实来源 + M4 控制变量。

   注：② 环带采样实装时 `KEEP_CITATION_POS=False`，pos 也从 `k+` 带采，pos 集合随之
   改变 —— 实际是「策略级对比」而非逐行「只换 negs」。冻结的是 q 集合 / 数据规模 /
   编码器 / margin loss / 评测尺。改 `True` 即退化为严格逐行对照。
   详见 `docs/05_m4_ring.md` 第 5 节，差异不藏。

## 回归校验（`scripts/13_regression.py`，PASS）

| 闸 | 判据 | 实测 |
|---|---|---|
| A | `encode(fmt=None)` 对 light 前 48 篇 vs `07b` 产物 | cos=1.0000000000, **max_abs=0.00e+00** |
| A2 | `normalize=True` 行范数 | 1.0000 |
| B | `build_triplets('citation')` == `triplets.jsonl` | True, 2758 rows |
| C | `fmt='clf'` 抛 NotImplementedError | True |
| D | `build_triplets('knn_ring')` == `triplets_knn_ring.jsonl` | True, 2758 rows |

A 的逐位相同 = 接口化后 `02/03` 数字一个不差，无需重算；D = ② 接口接入后读表与
落盘产物逐行相等，且 `A/A2/B/C` 四闸未受污染。

## ② M4 已实现（2026-11-18，原「待办」已销项）

`knn_ring` 分支不再抛异常：`scripts/15_build_knn_ring.py` 用 `scibert-tuned.npz`
对 8421 篇训练语料自举 kNN 邻域表，按 `k⁺∈[20,25]` / `k⁻∈[3998,4000]`、
两带之间不采，产出 `triplets_knn_ring.jsonl`（2758 行）；
`build_triplets('knn_ring')` 读该落盘表（D 闸验证）。真实数据方向对齐 PASS：
`cos_pos_band 0.8857 > cos_neg_band 0.6807`。详见 `docs/05_m4_ring.md`。

诚实边界：8421 池里 `rank 4000` ≈ 中位数，`cos_neg_band 0.6807 ≈ cos_citation_neg 0.6763`，
难负带未体现原文「几十万池取前 1%」的难度增益；且本阶段**未**跑下游训练 +
同尺 lift 对照，该开放项按「默认全轻量级」纪律由用户拍板。
