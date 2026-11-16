# specter-jittor

用 Jittor 复现科学论文文档编码器（CPU 环境，结构+方向对齐，不追分数）。

## 范围

| 论文 | 目标 | 判据 |
|---|---|---|
| SPECTER 2004.07180 | 编码器 + triplet margin loss 训练闭环 | loss 下降；同域近邻率 >> 随机 |
| SciNCL 2202.06671 | 只换采样（kNN 环带，k+=25 / k-=4000），复用同一编码器 | 同域近邻率/检索方向一致 |
| SPECTER2 2211.13308 | 只做 CLF / RGN 两格式 | 格式内 > 共享 [CLS]（方向） |
| 审计 TopicIsNotAgenda | 只复现测量协议：10 万 CS 子图两级社区 + top-10 同社区率 | L1 高、L2 崩的趋势 |
| CCQGen/CCExpand 2601.00567 | 不做主线，仅 top-k 概念路由 demo 接口 | — |

## 环境（实测 2026-11-16）

- WSL2 Ubuntu，16 核 / 15.38 GB RAM / 951 GB 可用
- conda env `jittor`：Python 3.12.14 + jittor 1.3.11.0 + numpy 1.26.4
- torch 2.15.0+cpu（只做权重转换与数值对照）、transformers 5.23.0、sklearn、scipy、igraph、leidenalg
- **必须 `export cc_path=/usr/bin/g++-12`**（g++15 报 `uint8_t does not name a type`；非交互 shell 不读 `.bashrc`）

## 里程碑

| 里程碑 | 状态 | 证据 |
|---|---|---|
| M0 环境 | ✅ | `docs/00_env.md` |
| M1 权重迁移对齐 | ✅ 2026-11-16 | `docs/01_m1_alignment.md`：13 层 cos=1.00000000，`transposed=0` |
| M2 训练闭环 | 待做 | 万级查询，margin loss 下降 + 近邻可读 |
| M3 采样旋钮 | 待做 | SciNCL 环带采样对照 |
| M4 表示形态 | 待做 | CLF / RGN 两格式 |
| M5 审计协议 | 待做 | 10 万 CS 子图 Leiden 两级 |

## 重要更正

Jittor `nn.Linear` 权重是 `[out, in]`，**与 PyTorch 相同，不需要转置**（实测 + `transposed=0`）。

## 纪律

数字只来自论文原文 txt 或代码输出；每个里程碑先贴命令与输出，再下结论。
