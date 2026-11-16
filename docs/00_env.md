# 环境事实（2026-11-16 实测，非推测）

## 机器

| 项 | 值 |
|---|---|
| WSL | Ubuntu，内核 `6.18.33.2-microsoft-standard-WSL2` |
| 资源 | 16 核 / 15.38 GB RAM / 根分区 1007G，可用 951G |
| 用户 | guo76，家目录 `/home/guo76` |
| 网络 | `pypi.org` 0.15s ✅，`api.openalex.org` 0.29s ✅，`raw.githubusercontent.com` ✅，`api.semanticscholar.org` ✅ |
| 网络 | `huggingface.co` **不通**；`hf-mirror.com` 通但 26 KB/s；AWS S3（`ai2-s2-research*`、`openalex.s3`）**DNS 失败**；WSL 到 Windows 代理 `172.20.96.1:7877/7890` 不通 |

## Python 环境（conda env `jittor`）

```
解释器  /home/guo76/miniconda3/envs/jittor/bin/python
Python  3.12.14
numpy   1.26.4        # 不能升 2.x，jittor 的 C++ 按 numpy 1.x 的 _ARRAY_API 下标取值
jittor  1.3.11.0      # has_cuda=False
torch   2.15.0+cpu    # 只用于权重转换与数值对照，不参与训练
```

## M0 验收记录（2026-11-16 10:29）

```
jittor 1.3.11.0 | has_cuda False
matmul sum 16777216.0        # 256x256 全 1 矩阵乘再求和，期望 256*256*256 = 16777216
grad 6.0                     # y=x^2, x=3
Compiling Operators(5/5) used: 2.31s
```

## 两个必须记住的坑

1. **`cc_path` 必须在每个脚本里显式 export**：`export cc_path=/usr/bin/g++-12`。
   `.bashrc` 第 134 行虽有这行，但非交互 shell 会在开头 `return`，不执行到它；此时 jittor 回落 g++15，直接报 `types.h:22:9: error: 'uint8_t' does not name a type`（GCC 15 取消了 `<cstdint>` 的传递包含）。
2. **编译缓存**：`~/.cache/jittor/jt1.3.11/g++12.5.0/...` 必须保留；`~/.cache/jittor/jt1.3.11/g++15.2.0/` 是失败残留，重装/清理时只删后者。

## 算力实测（决定实验规模）

| 测项 | 结果 |
|---|---|
| FFN 前向+反向（512 token × 768 → 3072） | 0.021 s/步 |
| 推算 BERT-base（12 层，batch 4，seq 128） | 0.4–1.2 s/步 |
| 1 万查询（≈3 万三元组，7500 步） | 1–2.5 h/epoch |
| 官方全量 146K 查询（≈17 万步） | 20–55 h/epoch → 不做 |

## 数据可得性

| 数据 | 状态 |
|---|---|
| OpenAlex API（引文、字段、引用数） | 可用；无 key 时共享每日 $0.10 预算，建议申请免费 key（$1/天） |
| SciDocs 官方包 | 在 AWS S3，当前不可达；需 VPN 后重试 |
| SciRepEval | 官方仓库路径 404，需重新确认；数据同样依赖 S3 |
| CSFCube / DORIS-MAE | 未核实 |
