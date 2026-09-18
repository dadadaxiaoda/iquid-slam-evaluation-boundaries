#!/usr/bin/env python3
"""
单目惯性融合数据集构建: 7维运动 + 跟踪质量 + **跨模态不一致(视觉-惯性)**

与 kitti_dataset_instr.py 同思路, 但特征来源换成 ORB-SLAM3 的 LocalInertialBA:
  EdgeInertial 残差 [er(rad), ev(m/s), ep(m)] —— IMU预积分 vs 视觉估计 的不一致,
  这个量**在线可算**(不需要 GT), 是"视觉惯性不一致能否预测漂移"的核心特征。

特征分组
--------
motion (7) : dt, Δp_b(3), Δq_b(3)
track  (9) : log1p(n_inliers), inlier_ratio, lost_ratio, reproj_mean, reproj_std,
             log1p(n_map_pts), obs_per_mp, log1p(n_local_mps), kf_gap
vi    (14) : log1p(n_inertial_edges), log1p(n_vis_edges),
             err_R_mean, err_R_std, err_R_max,
             err_V_mean, err_V_std, err_P_mean, err_P_std, err_P_max,
             chi2_inertial_mean, rel_robust_chi2, |bg|, |ba|
dvi   (14) : vi 的一阶差分(首帧置0)

用法: python vi_dataset_instr.py <前缀如V103vi> <埋点目录名如vi_V103>
"""
import sys
import csv
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rrot

prefix = sys.argv[1]
instr_name = sys.argv[2] if len(sys.argv) > 2 else f"vi_{prefix[:4]}"

DATA = Path("/opt/slam-study/legacy/data")
INSTR = Path("/opt/slam-study/legacy/instr_out") / instr_name
W = 20

lab = np.load(DATA / f"{prefix}_scale_label.npz")
t_kf = lab["t_abs"].astype(np.float64)
p_s_raw = lab["p_slam"].astype(np.float64)
p_g = lab["p_gt"].astype(np.float64)
q_s = lab["q_slam"].astype(np.float64)
M = len(t_kf)
frame_dt = float(np.median(np.diff(t_kf)))
TOL = 0.5 * frame_dt
print(f"[{prefix}] 关键帧 {M}, 平均间隔 {frame_dt*1000:.1f} ms, 匹配容差 {TOL*1000:.1f} ms")

# ---------------- 帧级跟踪信号 ----------------
frames = list(csv.DictReader(open(INSTR / "frame_signals.csv")))
ft_ts = np.array([float(r["ts"]) for r in frames])
f_fid = np.array([int(r["frame_id"]) for r in frames])
order = np.argsort(ft_ts)
ts_sorted = ft_ts[order]
pos = np.clip(np.searchsorted(ts_sorted, t_kf), 0, len(ts_sorted) - 1)
left = np.clip(pos - 1, 0, len(ts_sorted) - 1)
better = np.abs(ts_sorted[left] - t_kf) < np.abs(ts_sorted[pos] - t_kf)
pos[better] = left[better]
match = order[pos]
dtf = np.abs(ft_ts[match] - t_kf)
print(f"[{prefix}] 帧级信号 {len(frames)} 行, 匹配 {int((dtf<TOL).sum())}/{M} "
      f"(偏差中位 {np.median(dtf)*1e3:.3f} ms)")
if (dtf < TOL).sum() < M * 0.9:
    raise SystemExit("[FAIL] 帧级匹配率过低")

def fcol(name):
    return np.array([float(frames[j][name]) for j in match], dtype=np.float64)

n_kps = fcol("n_kps"); n_map_pts = fcol("n_map_pts")
n_inliers = fcol("n_inliers"); n_lost = fcol("n_lost_kps")
reproj_mean = fcol("reproj_mean_px"); reproj_std = fcol("reproj_std_px")
obs_per_mp = fcol("obs_per_mp"); n_local_mps = fcol("n_local_mps")
fid = f_fid[match]
kf_gap = np.zeros(M); kf_gap[1:] = np.diff(fid)

track = np.stack([
    np.log1p(n_inliers),
    n_inliers / np.maximum(n_map_pts, 1),
    n_lost / np.maximum(n_kps, 1),
    reproj_mean, reproj_std,
    np.log1p(n_map_pts), obs_per_mp, np.log1p(n_local_mps),
    kf_gap,
], axis=1).astype(np.float32)

# ---------------- 跨模态不一致 (LocalInertialBA) ----------------
vi_rows = list(csv.DictReader(open(INSTR / "vi_signals.csv")))
vt = np.array([float(r["ts"]) for r in vi_rows])
print(f"[{prefix}] vi_signals {len(vi_rows)} 行, 时间范围 [{vt.min():.2f},{vt.max():.2f}] "
      f"vs KF [{t_kf[0]:.2f},{t_kf[-1]:.2f}]")

def vcol(name, rows_idx):
    return np.array([float(vi_rows[j][name]) for j in rows_idx], dtype=np.float64)

# 每个关键帧: 取 ts 最近的 LocalInertialBA 记录(容差内), 否则沿用上一条
VCOLS = ["n_inertial_edges", "n_vis_edges", "err_R_mean", "err_R_std", "err_R_max",
         "err_V_mean", "err_V_std", "err_P_mean", "err_P_std", "err_P_max",
         "chi2_inertial_mean", "rel_robust_chi2", "bg_x", "bg_y", "bg_z",
         "ba_x", "ba_y", "ba_z"]
vidx = np.full(M, -1, dtype=int)
hit = 0
for i, ts in enumerate(t_kf):
    j = int(np.argmin(np.abs(vt - ts)))
    if np.abs(vt[j] - ts) < TOL:
        vidx[i] = j
        hit += 1
    else:
        vidx[i] = -1
# 未命中的用上一条(前向填充)
last = 0
for i in range(M):
    if vidx[i] < 0:
        vidx[i] = last
    else:
        last = vidx[i]
print(f"[{prefix}] 跨模态信号直接命中 {hit}/{M} ({hit/M*100:.0f}%), 其余前向填充")

g = lambda n: vcol(n, vidx)
bg = np.stack([g("bg_x"), g("bg_y"), g("bg_z")], axis=1)
ba = np.stack([g("ba_x"), g("ba_y"), g("ba_z")], axis=1)
vi = np.stack([
    np.log1p(g("n_inertial_edges")),
    np.log1p(g("n_vis_edges")),
    g("err_R_mean"), g("err_R_std"), g("err_R_max"),
    g("err_V_mean"), g("err_V_std"),
    g("err_P_mean"), g("err_P_std"), g("err_P_max"),
    g("chi2_inertial_mean"), g("rel_robust_chi2"),
    np.linalg.norm(bg, axis=1), np.linalg.norm(ba, axis=1),
], axis=1).astype(np.float32)

# ---------------- 运动特征 ----------------
def umeyama(A, B):
    mu_A, mu_B = A.mean(0), B.mean(0)
    Ac, Bc = A - mu_A, B - mu_B
    cov = Bc.T @ Ac / len(A)
    U, S, Vt = np.linalg.svd(cov)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(D @ np.diag(S)) / ((Ac ** 2).sum() / len(A))
    return s, R, mu_B - s * (R @ mu_A)

S_G, R_G, T_G = umeyama(p_s_raw, p_g)
p_s = S_G * (R_G @ p_s_raw.T).T + T_G
rot_s = Rrot.from_quat(q_s)
motion = np.zeros((M, 7), dtype=np.float32)
motion[1:, 0] = np.diff(t_kf).astype(np.float32)
dps = p_s[1:] - p_s[:-1]
motion[1:, 1:4] = np.stack([rot_s[i].inv().apply(dps[i - 1]) for i in range(1, M)])
motion[1:, 4:7] = (rot_s[1:] * rot_s[:-1].inv()).as_rotvec().astype(np.float32)
motion[0] = motion[1]

dvi = np.zeros_like(vi)
dvi[1:] = vi[1:] - vi[:-1]

X_full = np.concatenate([motion, track, vi, dvi], axis=1)
NAMES = (["dt", "dp_bx", "dp_by", "dp_bz", "dq_bx", "dq_by", "dq_bz"]
         + [f"t_{n}" for n in ["log_inliers", "inlier_ratio", "lost_ratio",
                               "reproj_mean", "reproj_std", "log_map_pts",
                               "obs_per_mp", "log_local_mps", "kf_gap"]]
         + [f"v_{n}" for n in ["log_n_inertial", "log_n_vis",
                               "errR_mean", "errR_std", "errR_max",
                               "errV_mean", "errV_std",
                               "errP_mean", "errP_std", "errP_max",
                               "chi2_inertial", "rel_robust_chi2",
                               "nbg", "nba"]]
         + [f"d_{n}" for n in ["log_n_inertial", "log_n_vis",
                               "errR_mean", "errR_std", "errR_max",
                               "errV_mean", "errV_std",
                               "errP_mean", "errP_std", "errP_max",
                               "chi2_inertial", "rel_robust_chi2",
                               "nbg", "nba"]])
GROUPS = ["motion"] * 7 + ["track"] * 9 + ["vi"] * 14 + ["dvi"] * 14
assert X_full.shape[1] == len(NAMES) == len(GROUPS), (X_full.shape, len(NAMES), len(GROUPS))

# ---------------- Y: 机体系残差 (vi_analyze 已算好) ----------------
Y_full = lab["res_b"].astype(np.float32)

n_win = M - W
X = np.zeros((n_win, W, X_full.shape[1]), dtype=np.float32)
Y = np.zeros((n_win, 3), dtype=np.float32)
for i in range(n_win):
    X[i] = X_full[i:i + W]
    Y[i] = Y_full[i + W]

out = DATA / f"{prefix}_dataset_instr.npz"
np.savez(out, X=X, Y=Y, X_raw=X, Y_raw=Y, W=W, S_GLOBAL=S_G,
         X_full=X_full, Y_full=Y_full,
         feat_names=np.array(NAMES), groups=np.array(GROUPS),
         n_kf=M, n_vi_hit=int(hit))
print(f"[{prefix}] X={X.shape} Y={Y.shape} | 存至 {out}")
print(f"[{prefix}] |res| mean={np.linalg.norm(Y_full,axis=1).mean()*100:.2f} cm, 尺度={S_G:.4f}")
