"""M6 / 18 —— 层次社区 top-K 同社区率测量协议(复现「尺子」而非分数)。

在同一批冻结 SciBERT 文本嵌入上量 top-10 近邻同社区率相对随机基线的
lift, 用阴性/阳性对照证明这把尺子既不造假信号、也能检出真信号。
基线统一为「随机抽两篇不同论文」的排除自身口径(见 src/data/audit.py),
每组标签各自跑 1000 次置换检验得到自己的 95% 区间。
判据: 阴性四组 lift 落各自 CI 内 + MAG 阳性 lift 显著 > 1。
MAG 在 2500 篇子集内自洽计算 base 与 rate, 不补齐到 4500。
诚实边界(见落盘 note): 受控社区非论文 3.58M 引文社区, 绝对数字不外推。
"""

import json
import os
import sys
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
from src.data.audit import audit_labels
D = os.path.join(ROOT, "data", "scidocs")
A = os.path.join(ROOT, "data", "audit")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
SEED=42
K=10
Block=256
MC_trials=200000
Perm_trials=1000
Mag_files=["train.csv","val.csv","test.csv"]
# MAG 19 类学科标签; 返回子集下标与重编码标签, 与嵌入按同一子集对齐
def load_mag(ids):
    lab={}
    for f in Mag_files:
        for line in open(os.path.join(D,"mag",f)).readlines()[1:]:
            pid,c=line.strip().split(",")
            lab[pid]=int(c)
    sel=[i for i,p in enumerate(ids) if p in lab]
    raw=np.array([lab[ids[i]] for i in sel],dtype=np.int64)
    _,lm=np.unique(raw,return_inverse=True)
    return sel,lm.astype(np.int64)
def main():
    z=np.load(os.path.join(A,"communities.npz"),allow_pickle=True)
    a=np.load(os.path.join(A,"augmented.npz"),allow_pickle=True)
    ids=[str(x) for x in z["ids"]]
    E=np.load(os.path.join(D,"scibert-embeddings-light.npy"))
    assert E.shape[0] == len(ids) and bool(np.isfinite(E).all())
    print("[input] n=%d emb=%s finite=True" % (len(ids), E.shape), flush=True)
    rng=np.random.default_rng(SEED)
    cfg=dict(top_k=K,block_size=Block,mc_trials=MC_trials,perm_trials=Perm_trials)
    # 阴性主对照: 17 的 Leiden 分区 + 真值标签(仪器自检), 均进判据
    main_groups = [
        ("L1", z["l1"]), ("L2", z["l2"]),
        ("L1_true", z["l1_true"]), ("L2_true", z["l2_true"]),
    ]
    # 17b 增强图分区: 仅稳健性记录, 不进主判据
    robust_groups = [("L1_aug", a["l1_aug"]), ("L2_aug", a["l2_aug"])]
    main_tags = {t for t, _ in main_groups}
    report = {
        "n": len(ids), "seed": SEED,
        "config": {"K": K, "block": Block, "mc_trials": MC_trials,
                   "perm_trials": Perm_trials},
        "negative": {}, "robustness": {}, "positive": {}, "checks": [],
    }
    for tag, lab in main_groups + robust_groups:
        res = audit_labels(E, lab, rng, **cfg)
        bucket = "negative" if tag in main_tags else "robustness"
        report[bucket][tag] = res
        print("[%s] rate=%.5f base_excl=%.5f mc=%.5f lift=%.4f CI=[%.4f,%.4f] in=%s" %
              (tag, res["rate"], res["base_excl"], res["mc_base"], res["lift"],
               res["perm"]["p2_5"], res["perm"]["p97_5"], res["in_ci_95"]), flush=True)
        # 只有阴性组写入 checks; aug 两组只进 robustness
        if bucket == "negative":
            report["checks"].append({"name": "negative " + tag + " in 95% CI",
                                     "ok": res["in_ci_95"]})
    sel, lm = load_mag(ids)
    # 阳性对照: 子集嵌入 + 子集标签, base/rate 都在 2500 内算
    mag = audit_labels(E[sel], lm, rng, **cfg)
    report["positive"]["MAG"] = mag
    print("[MAG] n=%d classes=%d rate=%.4f base_excl=%.5f lift=%.2f CI=[%.3f,%.3f]" %
          (mag["n"], mag["classes"], mag["rate"], mag["base_excl"], mag["lift"],
           mag["perm"]["p2_5"], mag["perm"]["p97_5"]), flush=True)
    # 阳性判据: lift 超过自己的置换 p97.5
    mag_ok = bool(mag["lift"] > mag["perm"]["p97_5"])
    report["checks"].append({"name": "positive MAG lift > perm p97.5", "ok": mag_ok})
    ok = all(c["ok"] for c in report["checks"])
    report["verdict"] = "PASS" if ok else "FAIL"
    # 诚实边界随日志落盘
    report["note"] = ("planted/augmented labels are independent of frozen SciBERT "
                      "text embeddings, so lift~=1 is the correct negative-control "
                      "output; MAG classes correlate with text, hence high lift. "
                      "Controlled communities, not the paper's 3.58M citation graph; "
                      "absolute numbers are NOT comparable.")
    json.dump(report, open(os.path.join(LOG_DIR, "m6_audit.json"), "w"), indent=2)
    print("VERDICT:", report["verdict"], flush=True)
    print("[ok] logs/m6_audit.json", flush=True)
    sys.exit(0 if ok else 1)
if __name__=="__main__":
    main()
