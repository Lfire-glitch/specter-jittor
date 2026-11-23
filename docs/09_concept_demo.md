# 09 · CCQGen/CCExpand 技术演示：只复现调度机制（2026-11-23）

19 是独立只读 demo，零侵入主链。它不新增 retriever、不引测试集、不跑
`R@K / NDCG / MAP`、不微调、不报任何检索增益数字；只把 2601.00567 里
「概念索引如何调度生成与匹配」的**纯确定性算子**端到端跑一遍并做方向断言。

定级：**接口/机制级技术演示**，低于 M4 的「部分完成」。论文 Table 1 / Table 5
的绝对数字不与本 demo 比较、不外推。

## 1. 为什么只做 demo

这篇论文是检索(IR)方向，效果全部体现在下游 `NDCG@10 / Recall@K / MAP@K`，
而那条链恰好是本机最缺的：CSFCube / DORIS-MAE 标注 query 本地没有，S2ORC
80 万候选违反轻量纪律，Contriever / SPECTER2 fine-tune + 16 万 query 全库检索
是 GPU 活。仓库主线只有 `encode / fmt / build_triplets / margin loss / kNN /
社区`，没有 retriever + 排序评测链。动测试端等于另起一套 IR 系统，偏离
「结构/接口/机制复现」定位。

README 第一天就写死：`不做主线，仅 top-k 概念路由 demo 接口`。本 doc 是把
这条边界执行到底，不是临时降级。

## 2. LLM 在哪里、stub 在哪里

| 环节 | 论文 | 19 的处理 |
|---|---|---|
| §3.1 core topic / phrase 选择（2 处） | LLM 从候选集挑最多 10/15 个 | 受控 fixture 的 `groups / df` 直接给定，**不调用 LLM** |
| §4.2 `q = LLM([P;C])` 迭代 M 轮 | LLM 生成 query | `stub_query()` 纯模板字符串 |
| §5.1 `s = LLM(inst,d,q)` 每文档 M=5 段 | LLM（gpt-4o-mini）生成 snippet | `stub_snippet()` 固定 4 句模板 |
| §4.3 fine-tune retriever + §6 检索评测 | 对比学习训练 + CSFCube/DORIS-MAE | **完全不做** |

概念索引是 8 篇文档 / 120 个人造 phrase / 30 个概念节点的受控矩阵，
**不是** MAG 431,416 节点 taxonomy + 18,440 phrases；concept extractor 不训练，
`y_Q` 用注入矩阵模拟。4 处 LLM 环节全部跳过，等于跳过论文最核心的生成器。

## 3. 纯确定性可演示的四段机制

### 3.1 §3.1 索引构造

`subtree_scores()`：概念节点取子树 phrase 向量均值 `s(d,c)`，再与文档算
cosine，并把节点分数广播回 phrase。`distinctiveness = log(N/df)` 压低 96 个
通用高频 phrase；`top_fraction_mask(0.20)` 保留 24 个，`k_p'=20` 截断后
重归一，稀疏且 `pi_sum=1`。

### 3.2 §4.1 `pi` 覆盖闭环（核心）

```
pi = normalize(max(y_d - y_Q, eps))          # Eq.1
每轮采样 floor(k_p'/M) = 20/4 = 5 个         # Eq.2
```

论文 Eq.2 是 multinomial（允许重复）；本 demo 为把「覆盖并集单调上升」钉死，
采用**无放回 + blocked 屏蔽**的受控口径，不声称复现原采样分布。这个口径
差异写进 `src/data/concept.py` 的 `sample_round()` docstring。

### 3.3 §4.3 z-score 融合过滤

`fuse = z(s_text) + z(s_concept)`。构造「词汇不匹配但概念对齐」的 query：
单看 `s_text` 会选 q0，融合后选 q1。复现论文设这个算子的动机，而不是
round-trip 过滤的完整实现。

### 3.4 §5.2 argmax vs mean

`s* = argmax_s sim(q,s)`，再 `rel = (1-alpha)*a + alpha*b`。固定反例：

```
mean(snippet) = 0.3268  <  base a = 0.3313  <  max(snippet) = 0.3412
```

max 保住概念峰值、mean 稀释；alpha 扫描 `[0, 0.25, 0.5, 0.75, 1]` 记录
形状。这是表 5 反例的**机制**，不是表 5 的效果复现。

## 4. 实跑结果（seed=42）

```
[index]    nodes=30 phrases=120 top20=24 kp=20 kept=20 pi_sum=1.000000
[coverage] unions=[5, 10, 15, 20]
           covered_pi   0.05000 -> 0.00007
           uncovered_pi 0.05000 -> 0.06664
[filter]   text_top1=0 -> fused_top1=1  fused=[0.3142, 1.2617, -1.5759]
[relevance] mean=0.3268  base=0.3313  max=0.3412  s*=1
VERDICT: PASS
```

`logs/concept_demo.json` 四条判据全 OK，`note` 写入硬边界；产物被
`.gitignore` 的 `/logs/` 排除，不入库。

三条读法：

1. `top20=24 -> kp=20` 证明 distinctiveness 与截断口径自洽，高频通用 phrase
   质量全为 0。
2. `unions` 严格单调、已覆盖项降权 / 未覆盖项升权、每轮 `pi_sum=1`，证明
   §4.1 的去冗余方向可证；这是**设计出的确定性**，不是生成质量。
3. `0.3268 < 0.3313 < 0.3412` 证明 max 与 mean 在概念峰值上的方向差异，
   正是论文选 argmax 而非平均的动机。

## 5. H 闸：共用模块的规范化写法

按拍板，H 闸照 G 闸规范写在 `scripts/13_regression.py` 里，import
`src/data/concept.py` 的共用函数，用内联确定性小算例（6 个概念 / 3 轮 /
标签与打分独立）自检，不依赖任何被 gitignore 排除的产物：

```
[H ] coverage=True  pi_shift=True  fuse=True  argmax=True
[H ] union=[2, 4, 6]  mean=0.3268  base=0.3313  max=0.3412
```

13 全量九闸回归 `VERDICT: PASS`，A~G 归档数字逐位不变：

```
[A ] cos=1.0000000000  max_abs=0.00e+00
[F1] changed=['embeddings.word_embeddings.weight']  rows=[2, 3]
[F2] cos(clf,rgn) trained=0.6319  < 0.9
[G ] base_excl=0.2462  perm_mean=1.0034  CI=[0.869,1.143]
[H ] union=[2, 4, 6]
```

## 6. 诚实边界（必须随结论一起读）

- 复现的是「**索引如何调度生成与匹配**」，不是「生成质量 / 检索增益」；
  一个 `NDCG / Recall / MAP` 数字都不报、不外推、不与论文表 1 / 表 5 比较。
- 4 处 LLM 调用全部 fixture/stub；概念索引是受控人造向量；concept extractor
  不训练；`y_Q` 注入矩阵模拟。这把尺子**测不到论文声称的增益**，只测调度
  机制自洽。
- 19 用 NumPy 写纯确定性协议，是合规的「测量仪器层」；Jittor 负责全项目
  的模型/训练侧。逐文件 Jittor 化判读与可选 L1 修改留到最终统一收口（#20），
  不在本里程碑混入，避免改动归档数字。
- 源码逐行关键注释、13 的 H 闸 docstring 按约定走完整条流程后统一补。
