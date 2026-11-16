# M1 · 权重迁移数值对齐（通过）

日期：2026-11-16　环境：WSL2 CPU，jittor 1.3.11.0，g++-12

## 结果

输入：3 条文本（SciBERT 标题 / SPECTER 摘要句 / Attention 标题），padding 到 32，`max_length=64`。

| 层 | cosine | max_abs |
|---|---|---|
| 0（embedding 输出） | 1.00000000 | 1.91e-06 |
| 1–12 | 1.00000000 | 3.8e-06 ~ 2.3e-05 |
| [CLS] pooled | 1.00000000 | — |

判据：每层 cos > 0.9999、[CLS] > 0.999 → **PASS**。
残差量级 1e-5 属 fp32 累加顺序差异，不是实现错误。

## 关键事实（与交接文档不一致，以本文件为准）

1. **Jittor `nn.Linear` 权重是 `[out, in]`，与 PyTorch 相同，不需要转置。**
   实测 `nn.Linear(4, 3).weight.shape == (3, 4)`；加载 200 个张量，`transposed=0`，直接赋值即对齐。
   交接文档 `00_复现对接_Jittor.md` 第四节第 1 条“必须转置”是错误判断，应删除。
2. 权重文件来自 `allenai/specter`（HF），key **没有 `bert.` 前缀**，共 200 个张量 / 109.94M 参数；
   唯一被跳过的是 buffer `embeddings.position_ids`。
3. `cc_path` 必须在脚本内显式 `export`（非交互 shell 不读 `.bashrc`），否则回落 g++15 直接编译失败。

## 复现命令

```bash
export cc_path=/usr/bin/g++-12
python scripts/01_convert_weights.py      # bin -> npz（仅此步用 torch）
python scripts/03_m1_align.py             # torch 参考 vs jittor，输出逐层 cos
```

`logs/m1_align.json` 存有逐层数值。

## 顺带证实的能力

- 编码器 `BertModel(vocab_size=31116)` 可前向，参数量与 HF 完全一致（109.94M）。
- 首次 JIT 编译 33 个算子约 6.3 s，之后走缓存。
