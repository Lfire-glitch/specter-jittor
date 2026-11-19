# M5 / SPECTER2 表示形态（控制码路线，2026-11-19）

复现粒度：**接口形状 + 方向对齐**，不刷分。默认全轻量级。
原文：`specter2_2211.13308_text.txt`（任务目录），机制引用均出自该 txt。

## 1. 论文机制（4.1 Model，行 488-514）

SPECTER2 论点：一个嵌入装不下四类下游任务（CLF 分类 / RGN 回归 / PRX 邻近 / QRY 检索），
故为每个格式学一套表示。两条实现路线：

- **Control Codes**（本文采用）：给输入 prepend 一个格式专属 special token，
  **取该 token 在最终层的 embedding** 作为文档表示，接任务头（行 499-504）。
  注意是"控制码自己的位置"，不是 `[CLS]`。
- **Adapters**（仅立概念）：Adapter Fusion / PALs，任务专属小模块 + fusion（行 505-514）。

训练目标按格式分（行 105-107）：CLF 交叉熵、PRX/QRY triplet margin、RGN MSE。
原文结论（行 644-648）：adapter 变体略优，control codes 更简单高效。

## 2. 材料核对（第 0 步，实测）

| 项 | 结果 |
|---|---|
| SPECTER2 原文 txt/pdf | ✅ 有 |
| scivocab `vocab.txt` | ✅ 31090 行，`[unused0]`(id 2) / `[unused1]`(id 3) 空槽 |
| SPECTER2 adapter 权重 | ❌ 无（`models/` 仅 scibert / specter），全盘扫描确认 |

## 3. 三条实现决策（`src/model/encoder.py`，用户手打）

1. **复用 `[unused]` 空槽，不扩词表** → `vocab_size=31090` 不变，
   M1 的 npz 与全部归档权重完全兼容。控制码 embedding 是**未训练的随机值**
   （实测范数 id2=1.059 / id3=1.058，全体行均值 1.080）。
2. **在 id 层插码，不走 tokenizer 文本解析**。`BertTokenizer` 不认识 `[CLF]`，
   会被 wordpiece 拆碎；`_insert_ctrl()` 在 `input_ids / attention_mask / token_type_ids`
   三个数组的 `[CLS]` 之后各插一列（控制码 id / mask=1 / tt=0），截回 `max_len`
   防 position embedding 越界。
3. **`fmt=None` 路径零改动**，A 闸逐位保护 `02/03/05` 归档数字。
   `fmt='clf'/'rgn'` 时取 `h[:, 1]`（控制码位），`fmt=None` 取 `h[:, 0]`（`[CLS]`）。

接口：`encode(texts, fmt=None/'clf'/'rgn', normalize=False)`；非法值抛 `ValueError`。

## 4. 实测结果（真实论文，非合成）

冒烟脚本 `_m5_smoke.py`（16 篇，batch=16）：

```
[A ] fmt=None vs 07b  cos=1.0000000000  max_abs=0.00e+00     ← 逐位不变
[E1] cos(None,clf)=0.4805  cos(None,rgn)=0.4815  cos(clf,rgn)=0.9992
[E2] unknown fmt raises ValueError  True
[E3] clf deterministic  True       [E4] finite  True
VERDICT: PASS (接口形状)
```

归档回归 `scripts/13_regression.py`（E 闸用 8 篇、batch=16）：
`cos(None,clf)=0.4473`、`cos(clf,rgn)=0.9993`。两处 `0.48` vs `0.45`
的差异来自**同一批论文在不同 batch 里的 padding 长度不同**（padding 影响 position
部分的 attention），不是实现不一致；`cos(clf,rgn)≈0.999` 的结论在两种口径下完全一致。

## 5. 诊断：0.48 到底来自哪里（本课最该掌握的部分）

构造 A=无控制码、B=插 id2、C=插 id3 三组输入，测各位置 hidden：

```
cos(A.cls,  B.cls ) = 0.9871   控制码对 [CLS] 本身影响很小
cos(A.cls,  B.ctrl) = 0.4805   ← encode 里 fmt=None vs fmt=clf 的总差异
cos(B.cls,  B.ctrl) = 0.4940   ← 同一输入内 0/1 两位置的天然差异
cos(B.ctrl, C.ctrl) = 0.9992   ← 换控制码 id(2→3) 几乎不变
```

`0.4805 ≈ 0.4940`：**那 0.48 的差异几乎全部来自"取的位置从 0 换成 1"**，
不是控制码身份。换 id 只带来 `0.9992` —— 因为 `[unused]` 的 embedding 是未训练随机值，
经 12 层自注意力后差异被淹没，模型没学过"看见 `[CLF]` 就切换表示"。

**机制点**：control code 不是魔法 token，它的区分能力来自**多任务训练时梯度把那几个
embedding 拉开**。没有训练，它只是一个位置标记。

## 6. 诚实边界（不可略过）

1. **只完成了"接口形状"，没完成"格式专属表示"这个方向**。`cos(clf,rgn)=0.9992`
   —— clf 与 rgn 尚不可分。比前瞻预判"只能做形状"还退一步：连"两格式可区分"都没达标。
   根因是无 SPECTER2 预训练权重 + 未做多任务训练，符合预期，**不是代码 bug**。
   `13_regression.py` 把它列为 **E\* 记录项、不作判据**，不藏也不假装 PASS。
2. 接口切换确实改变了输出流（`cos(None,clf)=0.48`），但这来自换位置，不是格式语义。
   不能据此宣称"复现了 SPECTER2 的多格式表示"。
3. 控制码插在 `[CLS]` 之后、title 之前；截回 `max_len` 时代价是末尾少 1 个 token（短文本不受影响）。
4. `fmt` 分支与 `fmt=None` 权重完全相同，`02/03/05` 归档数字一个未动。
5. 未做**第二课**（格式专属训练头：CLF 用 MAG 标签 / RGN 用引用数）。按「默认全轻量级 /
   不验证已有轮子」纪律留给用户拍板；只有做了训练，`clf/rgn` 才可能真正分开。

## 7. 回归（`scripts/13_regression.py`，六闸全绿）

| 闸 | 判据 | 语义变化 |
|---|---|---|
| A / A2 | 逐位 / 行范数 | 不变 |
| B / D | citation / knn_ring 读表 | 不变 |
| C | `fmt='xxx'` 抛 `ValueError` | **M5 后更新**（旧版判 `fmt='clf'` 抛 `NotImplementedError`） |
| E | `fmt` 输出流可区分 + 有限 + 确定性 | **新增** |
| E\* | `cos(clf,rgn)` 记录 | 记录项，不作判据 |

## 8. 复现命令

```bash
export cc_path=/usr/bin/g++-12
python scripts/13_regression.py                 # 六闸 PASS
# 诊断脚本为 /tmp/_m5_diag.py（临时产物, 不入库）
```

## 9. 收口判断

接口 3 从"抛 NotImplementedError 的占位"变成"真通路"，三个可替换接口全部落地。
M5 定性：与 ①/② 一致的**机制/形状复现**，不是论文效果复现。
开放项：格式专属训练头（第二课）、adapter 路线（需权重）均未开工。
