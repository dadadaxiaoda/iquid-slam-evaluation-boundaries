#!/usr/bin/env python3
"""
KITTI 判据对比实验: 运动学特征 vs +ORB-SLAM3 跟踪质量特征

对比配置(同一轨迹、同一协议, 唯一变量是特征集合)
  B0 : motion(7)                         —— 基线, 复现旧结论
  B1 : motion + track(11)                —— 加跟踪质量水平量
  B2 : motion + track + dtrack(11)       —— 再加一阶差分(检验"逐帧驱动"假设)
  B3 : motion + 打乱的 track/dtrack       —— 容量对照(同维度但信息被破坏)

判据(与 2026-09-16 定稿一致)
  阶段① 可学性: R2 > 0.3 且 ATE 降幅 > 20%  -> 方向6.2 成立
  阶段② 结构排名: CfC vs LSTM/GRU/MLP 只作主张强弱

严谨性
  - 分块5折CV(contiguous block), 与基线脚本一致
  - 标准化统计量只用训练折(避免标准化泄漏)
  - B3 容量对照: 排除"维度变多导致的能力提升"这一混淆
  - R2 与 ATE 全部在米制空间计算

用法: python kitti_train_multi.py <前缀如K00i> [--configs B0,B1,B2,B3] [--models ...]
      [--seeds 0,1,2] [--epochs 150] [--tag run1]
"""
import sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.torch import CfC
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rrot

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--configs", default="B0,B1,B2,B3")
ap.add_argument("--models", default="CfC,LSTM,GRU,MLP")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--batch", type=int, default=128,
                help="批大小。CfC(ncps) 是逐步 Python 展开, 批越大每 epoch 的展开次数越少、越快")
ap.add_argument("--patience", type=int, default=30, help="早停耐心")
ap.add_argument("--probe", action="store_true", help="只跑 1 配置1模型1种子, 用于计时")
ap.add_argument("--tag", default="run1")
a = ap.parse_args()

PREFIX = a.prefix
DATA = Path("/opt/slam-study/legacy/data")
FIGS = Path("/opt/slam-study/legacy/figs")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIGS = a.configs.split(",")
MODELS = a.models.split(",")
SEEDS = [int(s) for s in a.seeds.split(",")]
K = a.folds
if a.probe:
    CONFIGS, MODELS, SEEDS = CONFIGS[:1], MODELS[:1], SEEDS[:1]
    print("[probe] 只跑 1 配置 1 模型 1 种子, 用于计时", flush=True)

# ---------------- 数据 ----------------
d = np.load(DATA / f"{PREFIX}_dataset_instr.npz")
X_ALL, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)
Y_raw = d["Y_raw"].astype(np.float32)
NAMES = list(d["feat_names"])
GROUPS = list(d["groups"])
W = int(d["W"])
S_G = float(d["S_GLOBAL"])
nwin = len(X_ALL)

lab = np.load(DATA / f"{PREFIX}_scale_label.npz")
p_s_raw, p_g, q_s = lab["p_slam"], lab["p_gt"], lab["q_slam"]
N_total = len(p_s_raw)
rot_s = Rrot.from_quat(q_s)
Rm = rot_s.as_matrix()

def umeyama(A, B):
    mu_A, mu_B = A.mean(0), B.mean(0)
    Ac, Bc = A - mu_A, B - mu_B
    cov = Bc.T @ Ac / len(A)
    U, S, Vt = np.linalg.svd(cov)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(D @ np.diag(S)) / ((Ac ** 2).sum() / len(A))
    return s, R, mu_B - s * (R @ mu_A)

S_G2, R_G, T_G = umeyama(p_s_raw, p_g)
assert abs(S_G2 - S_G) < 1e-6
p_s = S_G2 * (R_G @ p_s_raw.T).T + T_G
GIDX = np.arange(W, N_total)
assert len(GIDX) == nwin, (len(GIDX), nwin)

# ---------------- 特征列选择 ----------------
def cols_for(cfg):
    if cfg == "B0":
        return [i for i, g in enumerate(GROUPS) if g == "motion"]
    if cfg == "B1":
        return [i for i, g in enumerate(GROUPS) if g in ("motion", "track")]
    if cfg in ("B2", "B3", "O1"):
        return list(range(len(GROUPS)))
    raise ValueError(cfg)

# ---- Oracle 上界对照 ----
# O1 = motion(7) + "上一个关键帧的真实残差"(3) 放在窗口最后一帧位置。
# 该特征需要 GT 才能算, 在线不可获得, 仅用作**可学性上界**:
# 若 O1 在分块CV下能拿到高 R², 说明评测协议本身没问题, 那么 B0~B2 的失败
# 就只能归因于"在线特征里没有那个信息", 而不是模型/协议的问题。
Yprev = np.zeros((nwin, 3), dtype=np.float32)
Yprev[1:] = Y_raw[:-1]

# B3 容量对照: 把 track/dtrack 列在所有样本上随机打乱(含窗口内), 破坏信息但保留维度
rng = np.random.RandomState(1234)
X_SHUF = X_ALL.copy()
shuf_cols = [i for i, g in enumerate(GROUPS) if g in ("track", "dtrack")]
for c in shuf_cols:
    perm = rng.permutation(nwin)
    X_SHUF[:, :, c] = X_ALL[perm][:, :, c]

_motion_cols = [i for i, g in enumerate(GROUPS) if g == "motion"]
X_O1 = np.concatenate([X_ALL[:, :, _motion_cols],
                       np.zeros((nwin, W, 3), dtype=np.float32)], axis=2)
X_O1[:, -1, -3:] = Yprev

def X_for(cfg):
    if cfg == "B3":
        return X_SHUF
    if cfg == "O1":
        return X_O1
    return X_ALL

def ncols_for(cfg):
    return X_for(cfg).shape[2] if cfg == "O1" else len(cols_for(cfg))

def Xc_for(cfg):
    if cfg == "O1":
        return X_O1
    return X_for(cfg)[:, :, cols_for(cfg)]

# ---------------- 模型 ----------------
class MLP(nn.Module):
    def __init__(self, d_in, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 3))
    def forward(self, x): return self.net(x[:, -1])

class SeqModel(nn.Module):
    def __init__(self, kind, d_in, hid=64):
        super().__init__()
        self.rnn = (nn.LSTM if kind == "LSTM" else nn.GRU)(d_in, hid, batch_first=True)
        self.head = nn.Linear(hid, 3)
    def forward(self, x):
        out, _ = self.rnn(x); return self.head(out[:, -1])

class CfCNet(nn.Module):
    def __init__(self, d_in, hid=64, out=3):
        super().__init__()
        self.cell = CfC(d_in, AutoNCP(hid, out), batch_first=True)
    def forward(self, x):
        out, _ = self.cell(x); return out[:, -1]

def build(name, d_in):
    return {"CfC": lambda: CfCNet(d_in), "LSTM": lambda: SeqModel("LSTM", d_in),
            "GRU": lambda: SeqModel("GRU", d_in), "MLP": lambda: MLP(d_in)}[name]()

# ---------------- 评估 ----------------
def metrics(pred_std, va, ym, ys, naive_m=None):
    """pred_std: (n,3) 标准化空间预测; 返回 (R2, ATE)"""
    gidx = GIDX[va]
    Yva_m = Y_raw[va]
    pred_m = np.tile(naive_m, (len(va), 1)) if pred_std is None else pred_std * ys + ym
    r2 = 1 - ((pred_m - Yva_m) ** 2).sum() / ((Yva_m - Yva_m.mean(0)) ** 2).sum()
    p_corr = p_s[gidx] + np.einsum("nij,nj->ni", Rm[gidx], pred_m)
    ate = float(np.sqrt(((np.linalg.norm(p_g[gidx] - p_corr, axis=1)) ** 2).mean()))
    return float(r2), ate

loss_fn = nn.MSELoss()
bounds = np.linspace(0, nwin, K + 1).astype(int)
results = {}
t_start = time.time()

# 全局基线 ATE(未修正)
ate0_all = float(np.sqrt(((np.linalg.norm(p_g - p_s, axis=1)) ** 2).mean()))
print("=" * 88)
print(f"[{PREFIX}] 判据对比实验  设备={DEV}  窗口={nwin}  K折={K}  种子={SEEDS}  epochs={a.epochs}")
print(f"[{PREFIX}] 全序列 SE3 对齐 ATE = {ate0_all:.4f} m | 全局尺度 = {S_G:.4f}")
print(f"[{PREFIX}] 配置: " + ", ".join(f"{c}({ncols_for(c)}维)" for c in CONFIGS))
print("=" * 88)

for cfg in CONFIGS:
    cols = cols_for(cfg)
    d_in = ncols_for(cfg)
    Xc = Xc_for(cfg)
    row = {}
    print(f"\n########## {cfg}  特征 {d_in} 维 ##########")
    for name in ["Naive"] + MODELS:
        row[name] = []
    for k in range(K):
        va = np.arange(bounds[k], bounds[k + 1])
        tr = np.ones(nwin, dtype=bool); tr[va] = False
        # 标准化统计量只用训练折
        xm = Xc[tr].mean((0, 1)); xs = Xc[tr].std((0, 1)) + 1e-8
        Xn = torch.from_numpy(((Xc - xm) / xs).astype(np.float32))
        ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-8
        Yn = torch.from_numpy(((Y - ym) / ys).astype(np.float32))
        Xtr, Ytr = Xn[tr].to(DEV), Yn[tr].to(DEV)
        Xva, Yva = Xn[va].to(DEV), Yn[va].to(DEV)

        gidx = GIDX[va]
        ate_b = float(np.sqrt(((np.linalg.norm(p_g[gidx] - p_s[gidx], axis=1)) ** 2).mean()))
        r2n, ate_n = metrics(None, va, ym, ys, naive_m=Y_raw[tr].mean(0))
        row["Naive"].append(dict(r2=r2n, ate_b=ate_b, ate_a=ate_n))

        for name in MODELS:
            t_m = time.time()
            r2s, ates = [], []
            for sd in SEEDS:
                torch.manual_seed(sd); np.random.seed(sd)
                model = build(name, d_in).to(DEV)
                opt = torch.optim.Adam(model.parameters(), lr=a.lr)
                best, best_st, wait = float("inf"), None, 0
                for ep in range(a.epochs):
                    model.train()
                    perm = torch.randperm(len(Xtr), device=DEV)
                    for s0 in range(0, len(Xtr), a.batch):
                        idx = perm[s0:s0 + a.batch]
                        loss = loss_fn(model(Xtr[idx]), Ytr[idx])
                        opt.zero_grad(); loss.backward(); opt.step()
                    model.eval()
                    with torch.no_grad():
                        vl = loss_fn(model(Xva), Yva).item()
                    if vl < best:
                        best, wait = vl, 0
                        best_st = {k_: v_.clone() for k_, v_ in model.state_dict().items()}
                    else:
                        wait += 1
                        if wait >= a.patience: break
                model.load_state_dict(best_st); model.eval()
                with torch.no_grad():
                    pv = model(Xva).cpu().numpy()
                r2, ate = metrics(pv, va, ym, ys)
                r2s.append(r2); ates.append(ate)
            row[name].append(dict(r2=float(np.mean(r2s)), r2_sd=float(np.std(r2s)),
                                  ate_b=ate_b, ate_a=float(np.mean(ates))))
            el = time.time() - t_m
            print(f"    fold{k+1} {name:<5} {el:6.1f}s  R2={np.mean(r2s):+.3f}", flush=True)
        print(f"  fold{k+1}/{K} done  (t={time.time()-t_start:.0f}s)")
    results[cfg] = row
    print(f"  --- {cfg} 汇总 ---")
    for name in ["Naive"] + MODELS:
        r2 = np.array([r["r2"] for r in row[name]])
        imp = np.array([(1 - r["ate_a"] / r["ate_b"]) * 100 for r in row[name]])
        print(f"    {name:<6} R2={r2.mean():+.3f}±{r2.std():.3f}   ATE降幅={imp.mean():+.1f}%±{imp.std():.1f}%")

# ---------------- 汇总表 ----------------
print("\n" + "=" * 88)
print(f"[{PREFIX}] 判据总表 (米制空间, 分块{K}折)")
hdr = f"{'配置':<6}{'模型':<6}{'R²':>18}{'ATE降幅':>18}"
print(hdr); print("-" * 88)
for cfg in CONFIGS:
    for name in ["Naive"] + MODELS:
        r2 = np.array([r["r2"] for r in results[cfg][name]])
        imp = np.array([(1 - r["ate_a"] / r["ate_b"]) * 100 for r in results[cfg][name]])
        print(f"{cfg:<6}{name:<6}{r2.mean():>+11.3f} ± {r2.std():.3f}"
              f"{imp.mean():>+12.1f}% ± {imp.std():.1f}%")
print("=" * 88)
# "最佳"只在在线可获得的配置(B*)里评, O1 是 oracle 上界, 不参与判据
B_cfgs = [c for c in CONFIGS if c.startswith("B") and c != "B3"]
best = max(((cfg, m, np.mean([r["r2"] for r in results[cfg][m]]),
             np.mean([(1 - r["ate_a"] / r["ate_b"]) * 100 for r in results[cfg][m]]))
            for cfg in B_cfgs for m in MODELS), key=lambda z: z[2])
print(f"最佳(在线特征): {best[0]}/{best[1]}  R2={best[2]:+.3f}  ATE降幅={best[3]:+.1f}%")
ok_stage1 = best[2] > 0.3 and best[3] > 20
print(f"阶段① 可学性判据 (R2>0.3 且 ATE降幅>20%): {'通过 ✔' if ok_stage1 else '未通过 ✘'}")

# Oracle 上界对照
if "O1" in CONFIGS:
    print("\n[Oracle 上界] O1 = motion + 上一关键帧真实残差(离线GT, 在线不可得):")
    for m in MODELS:
        r2o = np.mean([r["r2"] for r in results["O1"][m]])
        io = np.mean([(1 - r["ate_a"] / r["ate_b"]) * 100 for r in results["O1"][m]])
        r2b0 = np.mean([r["r2"] for r in results["B0"][m]]) if "B0" in CONFIGS else float("nan")
        print(f"  {m:<6} O1 R2={r2o:+.3f} ATE降幅={io:+.1f}%   (对比 B0 R2={r2b0:+.3f})")
    print("  判读: O1 高而 B0~B2 低 => 评测协议本身可学, 瓶颈是'在线特征无该信息'")

# B2 vs B3 信息性检验
if "B2" in CONFIGS and "B3" in CONFIGS:
    print("\n[信息性对照] B2(真跟踪特征) vs B3(打乱同维度):")
    for m in MODELS:
        r2b2 = np.mean([r["r2"] for r in results["B2"][m]])
        r2b3 = np.mean([r["r2"] for r in results["B3"][m]])
        i2 = np.mean([(1 - r["ate_a"] / r["ate_b"]) * 100 for r in results["B2"][m]])
        i3 = np.mean([(1 - r["ate_a"] / r["ate_b"]) * 100 for r in results["B3"][m]])
        print(f"  {m:<6} R2 {r2b2:+.3f} vs {r2b3:+.3f} (Δ={r2b2-r2b3:+.3f})   "
              f"ATE降幅 {i2:+.1f}% vs {i3:+.1f}%")

# ---------------- 保存 ----------------
out = {"prefix": PREFIX, "device": str(DEV), "nwin": nwin, "folds": K,
       "seeds": SEEDS, "epochs": a.epochs, "lr": a.lr,
       "n_features": {c: ncols_for(c) for c in CONFIGS},
       "feat_names": NAMES, "groups": GROUPS,
       "ate0_all": ate0_all, "S_global": S_G,
       "results": {c: {m: results[c][m] for m in results[c]} for c in results},
       "best": {"cfg": best[0], "model": best[1], "r2": best[2], "ate_imp": best[3]},
       "stage1_pass": bool(ok_stage1)}
js = DATA / f"{PREFIX}_judge_{a.tag}.json"
js.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\n结果已存: {js}   用时 {time.time()-t_start:.0f}s")

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
xs = np.arange(len(MODELS)); w = 0.8 / max(len(CONFIGS), 1)
for ci, cfg in enumerate(CONFIGS):
    r2v = [np.mean([r["r2"] for r in results[cfg][m]]) for m in MODELS]
    iv = [np.mean([(1 - r["ate_a"] / r["ate_b"]) * 100 for r in results[cfg][m]]) for m in MODELS]
    axes[0].bar(xs + ci * w - 0.4, r2v, w, label=cfg)
    axes[1].bar(xs + ci * w - 0.4, iv, w, label=cfg)
axes[0].axhline(0.3, color="g", ls="--", lw=1, label="criterion R2=0.3")
axes[0].set_xticks(xs); axes[0].set_xticklabels(MODELS)
axes[0].set_ylabel("R² (meter space)"); axes[0].set_title(f"[{PREFIX}] R² by feature set")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")
axes[1].axhline(20, color="g", ls="--", lw=1, label="criterion 20%")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xticks(xs); axes[1].set_xticklabels(MODELS)
axes[1].set_ylabel("ATE reduction (%)"); axes[1].set_title(f"[{PREFIX}] ATE reduction by feature set")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGS / f"kitti_{PREFIX}_judge.png", dpi=110, bbox_inches="tight")
print(f"图已存: figs/kitti_{PREFIX}_judge.png")
