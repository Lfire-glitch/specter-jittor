# 04 · 三个可替换接口（2026-11-18，里程碑 ①）

依据 `00_复现对接_Jittor.md` 第七节。目标不是"能跑"，而是让 M4/M5 的对照实验
**只改一个参数**。本里程碑是纯重构，不改任何数字。

## 接口契约

| 接口 | 位置 | 签名 | 换的变量 | 冻结的变量 |
|---|---|---|---|---|
| 1 嵌入导出 | `src/model/encoder.py` | `encode(texts, fmt=None, normalize=False) -> np.ndarray[n,768]` | 权重（构造参数） | 编码器/分词/取 [CLS] |
| 2 采样策略 | `src/data/triplets.py` | `build_triplets(mode='citation'/'random'/'knn_ring') -> list[dict]` | 采样策略 | q/pos 集合、数据规模 |
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
   读表 = 引用同一个数据对象，保证单一事实来源 + M4 控制变量（q/pos 冻结，只换 negs）。

## 回归校验（`scripts/13_regression.py`，PASS）

| 闸 | 判据 | 实测 |
|---|---|---|
| A | `encode(fmt=None)` 对 light 前 48 篇 vs `07b` 产物 | cos=1.0000000000, **max_abs=0.00e+00** |
| A2 | `normalize=True` 行范数 | 1.0000 |
| B | `build_triplets('citation')` == `triplets.jsonl` | True, 2758 rows |
| C | `fmt='clf'` 抛 NotImplementedError | True |

A 的逐位相同 = 接口化后 `02/03` 数字一个不差，无需重算。

## 待办（② M4）

`knn_ring` 分支当前抛 NotImplementedError，任务 ② 实现：用训好的嵌入对
8421 篇训练语料自举 kNN 邻域表，正例带 `k⁺∈[20,25]`、难负例带 `k⁻∈[3998,4000]`、
两带之间不采，同尺对比 citation vs knn_ring 的 lift。
