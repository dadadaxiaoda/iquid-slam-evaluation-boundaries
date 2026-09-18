#!/usr/bin/env python3
"""
A1 序列分析管线（泛化版）: 对任意 EuRoC 序列做 尺度漂移分析
用法: python a1_seq_analyze.py <序列名> <SLAM轨迹TUM> <数据集目录> [输出前缀]
输出: data/<seq>_scale_label.npz + figs/a1_<seq>_*.png + 终端汇总表
"""
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rrot, Slerp
from scipy.signal import medfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seq = sys.argv[1]                 # e.g. V201
slam_tum = sys.argv[2]            # SLAM 轨迹 (TUM, 秒)
ds_dir = Path(sys.argv[3])        # e.g. ~/datasets/V201 (含 mav0)
DATA = Path("/opt/slam-study/legacy/data")
FIGS = Path("/opt/slam-study/legacy/figs")

# ---------- 1. 读真值 CSV -> (t, p, q) ----------
gt_csv = ds_dir / "mav0/state_groundtruth_estimate0/data.csv"
rows = []
for line in open(gt_csv):
    if line.startswith("#"):
        continue
    v = line.strip().split(",")
    if len(v) < 8:
        continue
    rows.append([float(x) for x in v[:8]])
gt = np.array(rows)                      # t_ns, px,py,pz, qw,qx,qy,qz
t_g = gt[:, 0] / 1e9
p_g = gt[:, 1:4]
q_g = np.concatenate([gt[:, 5:8], gt[:, 4:5]], axis=1)   # -> xyzw

# ---------- 2. 读 SLAM TUM ----------
slam = []
for line in open(slam_tum):
    if line.startswith("#"):
        continue
    v = line.split()
    if len(v) < 8:
        continue
    slam.append([float(x) for x in v[:8]])
slam = np.array(slam)
assert slam.shape[0] >= 50, f"SLAM 轨迹太短: {slam.shape}"
t0_abs = slam[0, 0]
t_s = slam[:, 0]                    # 绝对时间(秒)
p_s = slam[:, 1:4]
q_s = slam[:, 4:8]

# ---------- 3. 配对: 真值插值到 SLAM 时刻 (统一绝对时间轴!) ----------
valid = (t_s >= t_g[0]) & (t_s <= t_g[-1])
t_sv = t_s[valid]
n_drop = int((~valid).sum())
p_g_i = np.stack([np.interp(t_sv, t_g, p_g[:, k]) for k in range(3)], axis=1)
slerp = Slerp(t_g, Rrot.from_quat(q_g))
q_g_i = slerp(t_sv).as_quat()
p_sv = p_s[valid]
q_sv = q_s[valid]
t_sv_rel = t_sv - t_sv[0]   # 仅用于画图
N = len(t_sv)
print(f"[{seq}] SLAM帧 {len(slam)} -> 配对 {N} (剔 {n_drop} 超范围)")

# ---------- 4. 全局 Sim3 对齐 (为得到对齐轨迹与全局尺度) ----------
def umeyama(A, B, with_scale=True):
    mu_A, mu_B = A.mean(0), B.mean(0)
    Ac, Bc = A - mu_A, B - mu_B
    cov = Bc.T @ Ac / len(A)
    U, S, Vt = np.linalg.svd(cov)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    var_A = (Ac ** 2).sum() / len(A)
    s = np.trace(D @ np.diag(S)) / var_A if with_scale else 1.0
    return s, R, mu_B - s * (R @ mu_A)

S_G, R_G, T_G = umeyama(p_sv, p_g_i)
p_align = (S_G * (R_G @ p_sv.T).T + T_G)
ate_sim3 = float(np.sqrt(((np.linalg.norm(p_g_i - p_align, axis=1)) ** 2).mean()))

def ate_se3(A, B):
    s, R, t = umeyama(A, B, with_scale=False)
    A_al = (R @ A.T).T + t
    return float(np.sqrt(((np.linalg.norm(B - A_al, axis=1)) ** 2).mean()))

ate_base = ate_se3(p_sv, p_g_i)

# ---------- 5. 滑窗局部尺度 ----------
L = 20
s_local = np.ones(N)
quality = np.zeros(N)
for i in range(N):
    lo, hi = max(0, i - L), min(N, i + L + 1)
    A, B = p_align[lo:hi], p_g_i[lo:hi]
    quality[i] = np.linalg.norm(np.ptp(A, axis=0))
    if quality[i] > 0.5:
        s_local[i] = umeyama(A, B)[0]
s_smooth = medfilt(s_local, 5)
obs_frac = (quality > 0.5).mean()

# ---------- 6. 常数尺度 vs 逐帧 Oracle ----------
def apply_scale(P, s):
    Pc = np.zeros_like(P)
    Pc[0] = P[0]
    for k in range(1, len(P)):
        Pc[k] = Pc[k-1] + s[k] * (P[k] - P[k-1])
    return Pc

ate_const = ate_se3(apply_scale(p_sv, np.full(N, S_G)), p_g_i)
ate_oracle = ate_se3(apply_scale(p_sv, s_smooth * S_G), p_g_i)

# 噪声对照: 局部尺度 std 与理论估计噪声(残差/窗口位移)
noise_pred = ate_sim3 / max(np.median(quality), 1e-6)
print(f"\n[{seq}] ================= 结果汇总 =================")
print(f"全局尺度 s = {S_G:.4f}   (路径长 SLAM {np.linalg.norm(np.diff(p_sv,axis=0),axis=1).sum():.1f} vs GT {np.linalg.norm(np.diff(p_g_i,axis=0),axis=1).sum():.1f} m)")
print(f"SE(3) 基线 ATE        : {ate_base:.4f} m")
print(f"常数尺度修正 ATE      : {ate_const:.4f} m  (= Sim3 水平 {ate_sim3:.4f})")
print(f"逐帧Oracle ATE        : {ate_oracle:.4f} m")
print(f"局部尺度: std={s_local.std():.4f}  理论噪声估计≈{noise_pred:.4f}  可观测帧 {obs_frac*100:.0f}%")
drift_signal = s_local.std() - noise_pred
print(f"=> 尺度漂移信号量(std - 理论噪声) = {drift_signal:+.4f} "
      f"({'存在真实漂移' if drift_signal > 0.005 and ate_oracle < ate_const * 0.95 else '基本是估计噪声'})")
print("=" * 52)

# ---------- 7. 保存 ----------
np.savez(DATA / f"{seq}_scale_label.npz",
         t=t_sv_rel, p_slam=p_sv, p_gt=p_g_i, q_slam=q_sv, q_gt=q_g_i,
         s_local=s_local, s_smooth=s_smooth, s_abs=s_smooth * S_G,
         quality=quality, S_GLOBAL=S_G,
         ate_base=ate_base, ate_const=ate_const, ate_oracle=ate_oracle,
         ate_sim3=ate_sim3, drift_signal=drift_signal)

# ---------- 8. 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(t_sv_rel, s_local, ".", ms=2, alpha=0.4, label="raw $s_i$")
axes[0].plot(t_sv_rel, s_smooth, "-", lw=1.5, color="#E63946", label="smoothed")
axes[0].axhline(1.0, color="k", ls="--", lw=0.8)
axes[0].set_title(f"[{seq}] local scale (std={s_local.std():.4f}, noise≈{noise_pred:.4f})")
axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("$s_i$"); axes[0].legend(); axes[0].grid(alpha=0.3)
Pc = apply_scale(p_sv, s_smooth * S_G)
axes[1].plot(p_sv[:, 0], p_sv[:, 1], "-", lw=1, alpha=0.5, label=f"raw SE3={ate_base:.3f}m")
axes[1].plot(Pc[:, 0], Pc[:, 1], "-", lw=1, alpha=0.6, color="#E63946", label=f"oracle={ate_oracle:.3f}m")
axes[1].plot(p_g_i[:, 0], p_g_i[:, 1], "k-", lw=1, alpha=0.6, label="GT")
axes[1].set_aspect("equal"); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_title(f"[{seq}] trajectories")
plt.tight_layout()
plt.savefig(FIGS / f"a1_{seq}_analysis.png", dpi=110, bbox_inches="tight")
print(f"已保存: data/{seq}_scale_label.npz, figs/a1_{seq}_analysis.png")
