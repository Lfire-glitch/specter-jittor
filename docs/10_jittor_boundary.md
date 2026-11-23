# 10 · Jittor 化边界与逐文件判读（2026-11-23，最终收口）

本项目从 M0 起就在 WSL2 CPU + Jittor 1.3.11.0 上做「结构 / 权重迁移 / 接口 /
训练机制 / 评测口径」的复现。收口时必须回答一个容易被外部误读的问题：
**「全项目用 Jittor 实现」到底覆盖到哪一层？** 本文给出实测能力清单、边界
论断、逐文件判读表和最终执行的分级，作为技术汇报的附录。

## 1. 一句话结论

Jittor 对标的是 **PyTorch / TensorFlow 那一层「可微分张量 + 模型 + 训练」框架**；
它自带一批 NumPy-like 算子，但**不是** NumPy / SciPy / sklearn / igraph /
leidenalg / pandas / json / torch / transformers 的全生态替代。

判据只有一句：**是否进入可训练计算图、是否需要梯度、是否属于模型定义**。
是 → Jittor；否 → 它是「测量仪器 / 工程工具」，用通用库不算违规。PyTorch
项目也照样用 `sklearn.metrics` / `np.savez`，没人因此说它没用 PyTorch 实现模型。

因此本项目的「全项目用 Jittor」= **模型侧、训练侧、可学习参数侧全部 Jittor**；
而协议 / 评测 / 图 / 权重归档 / 文件 IO 是合规的仪器层。`jt.array(np.load(...))`
这种桥接是标准做法，不算「用 NumPy 实现模型」。

## 2. 本机 Jittor 1.3.11.0 实测能力

| 层 | 有能力 | 本机缺口 |
|---|---|---|
| 张量 | `Var / array / matmul / sum / mean / std / max / min / argmax / argsort / concat / stack / reshape / permute / where`；`.numpy()` ↔ `jt.array(np)` 桥接 | `jt.array(float64)` **强制降为 float32**；`matmul` 仅支持 float32，混 float64 常量编译失败（`support float32 only now`） |
| 自动微分 | `opt.backward` / `W.opt_grad(opt)` / `g.assign(...)` 梯度掩码 / `jt.no_grad()` | — |
| 模型 | `Module / nn.Linear / Embedding / LayerNorm / Dropout / softmax / relu / cross_entropy_loss` | `jt.models` 本机无 |
| 训练 | `optim.Adam / AdamW / SGD` / `lr_scheduler` / `clip_grad_norm` / `step()` | — |
| 数据 | `dataset / transform / contrib / misc / linalg / random` / `save/load` / `set_seed` | 无 `jittor.numpy` 模块；无 `random.default_rng`；`jt.linalg.norm` 该命名空间下无；无 `jt.savez` 多键契约；无 `np.percentile / bincount / unique` 同语义契约；`jt.argsort(x)` 返回 `(indices, sorted_values)` 元组而非纯索引数组 |
| 生态 | — | 无 `igraph / leidenalg / sklearn.ARI / urllib / csv / torch.load / transformers` 对应能力 |

**float32 是硬约束**：L1 探针第一次跑挂，就是因为造了 float64 的 `eps` 常量
与 float32 张量混用。统一 float32 入图、出图回 numpy 重归一后才能稳定。

## 3. 分级需求与最终选择

| 级别 | 做法 | 可行性 | 回归要求 | 是否执行 |
|---|---|---|---|---|
| **L0** | 不改代码，只写边界文档 | 最高 | 无 | ✅ 本文 |
| **L1** | 只把新增 `src/data/concept.py` 的三个纯算子（`l2norm` / `subtree_scores` 的 GEMM / `coverage_pi`）改成 `jt.Var` | 高、零回归面 | 19 全量 + 13 A~H + 归档对账 | ✅ **已执行**（`b0d0b3a`） |
| **L2** | 再改存量 1 处纯 norm/GEMM：`encoder.py:68` 或 `15_build_knn_ring.py` | 中 | 必须复跑 A2 / D / G / 18 全量 | ❌ 收益低、风险中，不做 |
| **L3** | 全面替换 numpy/sklearn/igraph/leidenalg/npz/default_rng | 不可行 | 会击穿全部归档数字 | ❌ 不做 |

### L1 实测（`_probe20_l1.py` 探针 + 落盘实跑）

- 三算子漂移：`l2norm 3.9e-8` / `subtree 1.1e-7` / `coverage_pi 7.5e-9`；
- 19 注入 A/B/C 三档（只换 `coverage_pi` / 加 `l2norm` / 三个全换）全部 `PASS`，
  与原 `logs/concept_demo.json` **全字段 diffs=0**（tol 1e-9）；
- 落盘版：19 全量 `PASS`、L1 前后日志逐字段 `diffs=0`、13 九闸回归 `PASS`
  且 A~G 归档数字逐位不变；
- 关键设计：**显式 float32 入图 + numpy 收尾重归一**。若直接把 float32 结果
  当概率用，`pi.sum()` 的 ~1e-7 偏差会击穿 19 里 `<1e-12` 的归一化断言。
  `coverage_pi` 末尾 `pi / pi.sum()` 保证口径精确，使「Jittor 化」不改变任何数字。

L1 只动 `src/data/concept.py`；`19_concept_demo.py` / `13_regression.py` /
模型主链 / A~H 闸 / 其它仪器层一律不动。**任何归档数字漂移都会回退并记为
「口径不可迁移」，本次未发生。**

## 4. 逐文件判读（21 个 scripts + 13 个 src，收口版）

### A. 必须保持 Jittor，绝对不动（模型 / 训练 / 编码主链）

| 文件 | 依据 |
|---|---|
| `src/model/bert.py` | 纯 Jittor BERT 前向；M1 13 层 `cos=1.0` / `transposed=0` 命脉 |
| `src/model/hf_loader.py` | `np.load` 只做权重 IO，落点 `tgt.assign(jt.array(...))` |
| `src/model/encoder.py` | `jt.no_grad → jt.array → model → .numpy()`；A/A2/C/E/F2 全覆盖 |
| `03_m1_align.py` `06_scibert_align.py` | HF 参照 + Jittor 对齐判据 |
| `04_encode_check.py` `07/07b/07c/14_encode*.py` | 编码主链与 `.npy` 产物契约 |
| `11_train_margin.py` `16_train_fmt.py` | loss / `opt_grad` / 梯度掩码 / `Adam.step()`；训练态禁用 `eval()` |

### B. 已做 L1 的新增文件（纯算子 jt.Var，其余留 numpy）

| 文件 | 范围 |
|---|---|
| `src/data/concept.py` | `l2norm` / `subtree_scores` 的 `d@centers.T` / `coverage_pi` 用 `jt.Var`；`argsort / default_rng / JSON / npz` 仍 numpy |
| `19_concept_demo.py` | 纯 numpy 编排 + 四段方向断言，import 上面的 L1 模块，本身不改 |

### C. 不建议改（仪器 / 判据 / 随机 / IO 口径）

| 文件 | 关键不可迁语义 |
|---|---|
| `src/data/audit.py` `18_audit_protocol.py` | `bincount / percentile / argsort / permutation / integers`；MC 20 万 + 置换 1000 + 排除自身基线；G 闸与 18 判据 |
| `13_regression.py` | `.npz` 198 张量逐位比较、`array_equal`、`default_rng(2026/2027)`、A~H 闸 |
| `09_eval_light.py` `12_eval_tuned.py` `src/eval/classification.py` | majority / random / kNN macro-F1，`np.argsort(-sim)` 并列顺序影响投票 |
| `15_build_knn_ring.py` `src/data/triplets.py` | `triplets*.jsonl` 逐行相等；`random.Random(seed).sample`；B/D 闸 |
| `17_build_communities.py` `17b_augment_citation.py` | `default_rng`、Leiden CPM、ARI、`np.savez`；图/ARI 子部分实质 D |
| `08_build_light_ids.py` | `csv.DictReader` / `random.sample(seed=42)` / `light_ids.json` 契约 |

### D. 不能用 Jittor 替代（生态缺口）

| 文件 | 缺的能力 |
|---|---|
| `01_convert_weights.py` `05_convert_scibert.py` | `torch.load`、HF `state_dict`、多键 `.npz` 归档 |
| `10_fetch_openalex.py` | `urllib / HTTP / jsonl`，纯数据准备 |
| `17* 图分区` | `igraph.Graph` / `leidenalg.CPMVertexPartition` / `sklearn.adjusted_rand_score` |
| 全部 `json / os / csv` IO | 标准库文件与随机清单语义 |

空 `__init__.py`（`src` / `data` / `eval` / `model` / `train`）不适用判读；
`src/train/` 无实现，训练在 11 / 16。

## 5. 诚实边界（随结论一起读）

- 「全项目用 Jittor」指 **模型 / 训练 / 可学习参数侧**；评测 / 图 / IO 是合规仪器层，
  不是「用 NumPy 实现了模型」。
- L1 是**边界性 / 象征性** Jittor 化：把三个无第三方语义依赖的纯算子接进
  `jt.Var`，证明框架边界自洽；它**不提升任何复现效果**，也不改变任何归档数字。
- L3 全量替换不做：float32 漂移会击穿 A~H 闸、置换 CI、F1 逐位比较、
  `npz` 权重契约和图算法；那会把「验协议」变成「验框架语义差异」。
- 绝对不动：`bert.py` / `hf_loader.py` / `encoder.py` 主链、11 / 16 训练与梯度掩码、
  `13_regression.py`、`audit.py` 的 MC/置换、18、09 / 12、17 / 17b 图与 ARI、
  01 / 05（`torch.load + npz`）、10（`urllib`）、08（csv / 随机清单）。
