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
| M2 SciBERT 基线 + 训练闭环 | ✅ 2026-11-17 | `docs/02_scibert_baseline.md`、`docs/03_m2_closed_loop.md`：margin loss，同尺 MAG +0.193 / MeSH +0.424 |
| M3 三接口收敛 ① | ✅ 2026-11-18 | `docs/04_interfaces.md`：encode/fmt/build_triplets，四闸回归 PASS |
| M4 SciNCL 环带采样 ② | ✅ 2026-11-18 | `docs/05_m4_ring.md`：k+/k- 环带，方向对齐 PASS（cos 0.8857 > 0.6807） |
| M5 表示形态（控制码） | ✅ 2026-11-19 | `docs/06_m5_formats.md`：`fmt='clf'/'rgn'` 真通路，六闸回归 PASS；仅形状成立，格式尚不可分（无训练权重） |
| M6 审计协议 | 待做 | 10 万 CS 子图 Leiden 两级 |

注：表内 M3/M4 为本表早期顺序编号；`docs/04`、`docs/05` 按任务 ①/② 叙述，
对应的 ① = 三接口收敛、② = SciNCL 环带采样。M6 排队未开工；
M5 为机制/形状复现，非论文效果复现，诚实边界见 `docs/06` 第 6 节。

## 重要更正

Jittor `nn.Linear` 权重是 `[out, in]`，**与 PyTorch 相同，不需要转置**（实测 + `transposed=0`）。

## 纪律

- 默认全轻量级：不下 16 GB `train.pkl`、不跑 48k 全量重编码；报告重点是技术复现与机制成立，
  不是验证已有轮子或刷绝对分数。
- 数字只来自论文原文 txt 或代码输出；每个里程碑先贴命令与输出，再下结论。
- 本地提交持续累积，**最后一并推送**（WSL NAT 无法直连 Windows 代理）。
