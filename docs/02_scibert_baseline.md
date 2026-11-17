# 02 · SciBERT 轻量基线（2026-11-17）

## 目标与定位

SciBERT 是 SPECTER 的底座（架构与 BERT 完全相同，差异只在预训练语料与 scivocab 词表）。
本里程碑产出两件事：

1. **底座分数**：未做引文训练的嵌入在轻量评测协议下的表现；
2. **验证过的评测器**：用官方 SPECTER 嵌入（已知更优）校准同一把尺。

约定：CPU 环境只走轻量子集（结构+方向对齐），不与论文绝对分数对比。

## 轻量协议（`scripts/08_build_light_ids.py`，seed=42）

- MAG：pool=1500（train 抽样）/ query=1000（test 抽样），19 类
- MeSH：pool=1000 / query=1000，11 类
- 共 4,500 篇唯一论文；`scripts/07b_encode_light.py` 编码（Jittor SciBERT，最后一层 [CLS]，不过 pooler）
- 编码耗时 12.6h（CPU 长时满载降频，0.72→0.10 篇/s）

## 评测器（`src/eval/classification.py`）

kNN（k=5，余弦相似度，rank 加权投票）+ macro-F1。合成数据自检：
噪声嵌入 0.194 ≈ 随机，聚类嵌入 1.000 → 判别力合格。

## 结果（`logs/scibert_baseline.json`）

| 任务 | majority | random | SciBERT 底座 | 官方 SPECTER | 训练差距 |
|---|---|---|---|---|---|
| MAG | 0.005 | 0.047 | 0.631 | 0.711 | **+0.080** |
| MeSH | 0.034 | 0.089 | 0.304 | 0.697 | **+0.393** |

## 解读

1. 评测器可信：官方向量全面更高，且 lift 与基线差距一致拉大。
2. 差距结构符合论文叙事：细粒度 MeSH 收益（+0.393）远大于粗粒度 MAG（+0.080）
   ——引文训练的主要收益在细粒度语义聚合。
3. 底座 mesh<mag 倒挂（0.304<0.631），官方嵌入修正为 0.697≈0.711；
   M2 训练后应观察倒挂收窄。
4. 绝对分数不可与论文（全量协议：kNN 池 16k+、不同实现）直接对比，仅同尺内比较。

## M2 的量化目标

在**同一协议、同一子集**上训练我们自己的 margin-loss 编码器：
- MAG 从 0.631 向 0.711 靠近；MeSH 从 0.304 向 0.697 靠近；
- 至少要求：训练后 lift 显著为正且 mesh 倒挂收窄。

## 附：过程教训

- WSL 虚拟机曾卡死（E_UNEXPECTED），`wsl --shutdown` 恢复；编码脚本需分段落盘续跑（M2 前补）。
- `paper_metadata_view_cite_read.json` 未下载 → cite/cocite/coread/coview 任务暂缺，待 VPN 补齐后可加链接预测任务。
