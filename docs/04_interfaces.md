# 04 · 三个可替换接口（2026-11-18 里程碑 ①；2026-11-19 更新 fmt；2026-11-20 补 F 闸）

依据 `00_复现对接_Jittor.md` 第七节。目标不是"能跑"，而是让 M4/M5 的对照实验
**只改一个参数**。本里程碑是纯重构，不改任何数字。

## 接口契约

| 接口 | 位置 | 签名 | 换的变量 | 冻结的变量 |
|---|---|---|---|---|
| 1 嵌入导出 | `src/model/encoder.py` | `encode(texts, fmt=None, normalize=False) -> np.ndarray[n,768]` | 权重（构造参数） | 编码器/分词/取 [CLS] |
| 2 采样策略 | `src/data/triplets.py` | `build_triplets(mode='citation'/'random'/'knn_ring') -> list[dict]` | 采样策略 | q 集合、数据规模 |
| 3 表示形态 | `src/model/encoder.py` | `encode(..., fmt=None/'clf'/'rgn')` | 表示形态 | 编码器、训练数据 |

接口 3 在 M5（`docs/06_m5_formats.md`）已从占位变成真通路：`fmt` 生效时在 id 层插控制码、
取控制码位 hidden。三个接口至此全部落地。M5 第二课（`docs/07_m5_train.md`）进一步用
训后权重证明两格式可分；`PaperEncoder(weights=...)` 本就是接口 1 的"换权重"能力，
无需新增接口。

## 四条边界决策（回归纪律）

1. **默认不 L2 归一化**（`normalize=False`）。交接文档写"输出 L2 归一化"，但 M2 的
   margin loss 用 L2 距离、评测尺内部自做 cosine 归一化——若 encode 强行归一化，
   `02/03` 归档数字全变。留 `normalize=True` 开关，默认关闭。
2. **`fmt=None` 恒逐位不变**：M5 加 `fmt` 分支时，`fmt=None` 路径一行未动，
   A 闸 `cos=1.0 / max_abs=0.00e+00` 护住 `02/03/05` 归档数字。
   `fmt='clf'/'rgn'` 只改输入侧（插控制码）与取位（0→1），模型权重完全相同。
   非法 fmt 抛 `ValueError`。实现细节与诚实边界见 `docs/06`。
3. **训练侧 encode 不合并**：`11_train_margin.py` 需梯度 + `train()` 模式（dropout 开），
   `PaperEncoder` 恒 `eval()` + `no_grad()`。同一模型、不同执行模式，硬合并会破坏 M2 数字。
   ⚠️ **Jittor 契约补充（2026-11-19 实测）**：Jittor 的 `eval()` 会 `stop_grad` 全部参数
   （不同于 PyTorch 只切 dropout/BN）。故**训练态一律不调 `eval()`**，主体冻结靠
   "参数不进 optimizer"实现；`PaperEncoder` 只在推理时用。详见 `docs/07` 第 2 节。
4. **`build_triplets('citation')` 读落盘表**，不重写 `10` 的采样逻辑。
   追求"产物一致"而非"过程一致"：生成逻辑挂着 `random` 全局状态、OpenAlex 返回顺序、
   242 篇过滤等隐含随机性，重跑一遍得到的会是另一张表，M2 即不可复现。
   读表 = 引用同一个数据对象，保证单一事实来源 + M4 控制变量。

   注：② 环带采样实装时 `KEEP_CITATION_POS=False`，pos 也从 `k+` 带采，pos 集合随之
   改变 —— 实际是「策略级对比」而非逐行「只换 negs」。冻结的是 q 集合 / 数据规模 /
   编码器 / margin loss / 评测尺。改 `True` 即退化为严格逐行对照。
   详见 `docs/05_m4_ring.md` 第 5 节，差异不藏。

## 回归校验（`scripts/13_regression.py`，七闸 PASS）

| 闸 | 判据 | 实测 / 语义 |
|---|---|---|
| A | `encode(fmt=None)` 对 light 前 48 篇 vs `07b` 产物 | cos=1.0000000000, **max_abs=0.00e+00** |
| A2 | `normalize=True` 行范数 | 1.0000 |
| B | `build_triplets('citation')` == `triplets.jsonl` | True, 2758 rows |
| C | 非法 `fmt='xxx'` 抛 `ValueError` | True（**M5 后改判据**；旧版判 `fmt='clf'` 抛 `NotImplementedError`） |
| D | `build_triplets('knn_ring')` == `triplets_knn_ring.jsonl` | True, 2758 rows |
| E | `fmt` 输出流可区分 + 有限 + 确定性 | cos(None,clf)=0.4473（8 篇）, True, True |
| E\* | `cos(clf,rgn)`（基线权重） | 0.9993 —— **记录项，不作判据**：控制码未训练，格式尚不可分 |
| **F1** | `scibert-fmt.npz` 仅 `row2/3` 变、198 张量逐位相等 + finite | True（梯度 mask 生效，主体冻结） |
| **F2** | 训后权重 `cos(clf,rgn) < 0.9` | 0.6319（基线 0.9992，方向建立） |

（冒烟 16 篇口径为 0.4805 / 0.9992，差异来自 batch padding；详见 `docs/06` 第 4 节。）

A 的逐位相同 = 接口化后 `02/03` 数字一个不差，无需重算。
E 只验「接口形状成立」；F2 用训后权重补上「方向」，`E*` 与 `F2` 描述不同权重，不矛盾。

## ② M4 已实现（2026-11-18）

`knn_ring` 分支不再抛异常：`scripts/15_build_knn_ring.py` 用 `scibert-tuned.npz`
对 8421 篇训练语料自举 kNN 邻域表，按 `k⁺∈[20,25]` / `k⁻∈[3998,4000]`、
两带之间不采，产出 `triplets_knn_ring.jsonl`（2758 行）；
`build_triplets('knn_ring')` 读该落盘表（D 闸验证）。真实数据方向对齐 PASS：
`cos_pos_band 0.8857 > cos_neg_band 0.6807`。详见 `docs/05_m4_ring.md`。

诚实边界：8421 池里 `rank 4000` ≈ 中位数，`cos_neg_band 0.6807 ≈ cos_citation_neg 0.6763`，
难负带未体现原文「几十万池取前 1%」的难度增益；且本阶段**未**跑下游训练 +
同尺 lift 对照（2026-11-18 用户拍板收口，按轻量纪律不补）。

## ③ M5 已实现（2026-11-19 形状 / 2026-11-20 方向）

`fmt` 分支 = SPECTER2 control codes 路线的**形状复现**：复用 `[unused]` 空槽不扩词表、
id 层插码、取控制码位 hidden，`fmt=None` 零改动。第一课六闸 PASS，但
`cos(clf,rgn)=0.9992` **方向未立**（诊断证明 0.48 差异来自换位置而非控制码身份）。

第二课 `scripts/16_train_fmt.py` 冻结 BERT 主体、只训 row2/3 + 两个小头
（CLF 交叉熵 / RGN 引用数 MSE），把 `cos(clf,rgn)` 拉到 **0.6319**，
方向建立；F1/F2 闸固化。两课均为机制复现，非论文效果复现。
详见 `docs/06_m5_formats.md`、`docs/07_m5_train.md`。
