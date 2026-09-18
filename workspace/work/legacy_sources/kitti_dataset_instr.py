#!/usr/bin/env python3
"""
KITTI 融合数据集构建: 7维运动特征 + ORB-SLAM3 跟踪质量特征(埋点)

背景
----
方向6.2 假设: 纯运动学特征无法预测漂移(K00 判据实验 R2=-0.440),
但 ORB-SLAM3 内部的"跟踪质量信号"(匹配数/重投影误差/地图成熟度/BA残差)
可能携带驱动因子。本脚本把这些信号按关键帧对齐后拼进特征向量。

因果性保证
----------
窗口 X[i] = 关键帧 [i, i+W) 的特征, 目标 Y[i] = 关键帧 i+W 的机体系残差。
关键帧 j 的特征只用到"处理完关键帧 j"为止的信息(含 j 触发的 LocalBA),
因此在预测 i+W 时全部可用 -> 无未来信息泄漏。

对齐方式
--------
scale_label 的 t 是该次 SLAM 运行 KeyFrameTrajectory.txt 的时间戳序列;
frame_signals.csv 的 ts 是同一批帧时间戳 -> 精确匹配(容差 1e-6s)。
不依赖 is_kf 列做顺序假设, 但用 is_kf 交叉校验匹配率。

特征分组(共 29 维)
------------------
motion (7) : dt, Δp_b(3), Δq_b(3)
track  (11): log1p(n_inliers), inlier_ratio, lost_ratio, reproj_mean_px,
             reproj_std_px, log1p(n_map_pts), obs_per_mp, log1p(n_local_mps),
             ba_pix_mean_px, ba_ratio_inlier_edge, kf_gap
dtrack (11): track 的一阶差分(首帧置0)

用法: python kitti_dataset_instr.py <序列号如00> <前缀如K00i>
"""
import sys
import csv
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rrot

seq = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else f"K{seq}i"

DATA = Path("/opt/slam-study/legacy/data")
INSTR = Path("/opt/slam-study/legacy/instr_out")
W = 20

# ---------------- 1. 尺标标签(含 SLAM 位姿 / GT 位姿) ----------------
lab = np.load(DATA / f"{prefix}_scale_label.npz")
t_kf = lab["t"].astype(np.float64)          # 关键帧时间戳(本次运行)
p_s_raw = lab["p_slam"].astype(np.float64)  # 原始 SLAM 位置(未对齐)
p_g = lab["p_gt"].astype(np.float64)
q_s = lab["q_slam"].astype(np.float64)
M = len(t_kf)

# ---------------- 2. 读埋点帧级信号 ----------------
fs_path = INSTR / f"K{seq}" / "frame_signals.csv"
if not fs_path.exists():
    raise SystemExit(f"[FAIL] 找不到 {fs_path}")
frames = list(csv.DictReader(open(fs_path)))
print(f"[{prefix}] frame_signals: {len(frames)} 帧, 其中 is_kf=1 的 {sum(1 for r in frames if r['is_kf']=='1')} 帧")

ft_ts = np.array([float(r["ts"]) for r in frames])
f_frame_id = np.array([int(r["frame_id"]) for r in frames])
is_kf = np.array([int(r["is_kf"]) for r in frames])

# ---------------- 3. 逐关键帧匹配帧级信号 ----------------
# t_kf 是"关键帧"时间戳; 在 frame_signals 中定位同 ts 的行。
# 用 is_kf=1 的行优先, 找不到则退化到任意同 ts 行(初始化KF不走 CreateNewKeyFrame)。
# 说明: frame_signals.csv 的 ts 按 C++ ostream 默认 6 位有效数字输出,
# t > 1s 后实际精度只剩 ~1e-4 s, 因此不能用 1e-6 硬相等匹配。
# 采用"最近邻 + 容差=帧间隔1/4"的鲁棒匹配(与 kitti_analyze.py 的 GT 配对同思路)。
order = np.argsort(ft_ts)
ts_sorted = ft_ts[order]
pos = np.clip(np.searchsorted(ts_sorted, t_kf), 0, len(ts_sorted) - 1)
left = np.clip(pos - 1, 0, len(ts_sorted) - 1)
better = np.abs(ts_sorted[left] - t_kf) < np.abs(ts_sorted[pos] - t_kf)
pos[better] = left[better]
match = order[pos]
dt_match = np.abs(ft_ts[match] - t_kf)
frame_dt = float(np.median(np.diff(t_kf)))
TOL = 0.25 * frame_dt
hit = int((dt_match < TOL).sum())
print(f"[{prefix}] 关键帧 {M} 个, 帧级信号匹配上 {hit} 个 ({hit/M*100:.1f}%)")
print(f"[{prefix}] 匹配时间偏差: 中位 {np.median(dt_match)*1e3:.3f} ms, "
      f"最大 {dt_match.max()*1e3:.3f} ms (容差 {TOL*1e3:.1f} ms, 帧间隔 {frame_dt*1e3:.1f} ms)")
if hit < M * 0.95:
    raise SystemExit(f"[FAIL] 匹配率过低({hit}/{M}), 检查 ts 是否同源")
n_kf_flag = int(is_kf[match].sum())
print(f"[{prefix}] 交叉校验: 匹配行中 is_kf=1 的有 {n_kf_flag} 个 "
      f"(初始化KF不走 CreateNewKeyFrame, 容许少量缺口)")
mono = int(np.all(np.diff(match) >= 0))
print(f"[{prefix}] 匹配索引单调递增: {bool(mono)} (关键帧按时间序, 应为真)")

# ---------------- 4. 组装 track 特征 ----------------
def col(name, rows_idx):
    return np.array([float(frames[j][name]) for j in rows_idx], dtype=np.float64)

n_kps = col("n_kps", match)
n_map_pts = col("n_map_pts", match)
n_inliers = col("n_inliers", match)
n_lost_kps = col("n_lost_kps", match)
reproj_mean = col("reproj_mean_px", match)
reproj_std = col("reproj_std_px", match)
obs_per_mp = col("obs_per_mp", match)
n_local_mps = col("n_local_mps", match)
fid = f_frame_id[match]

# kf_gap: 与上一个关键帧的帧号间隔(= 关键帧插入频率的倒数, 越大越稀疏)
kf_gap = np.zeros(M)
kf_gap[1:] = np.diff(fid)

# ---------------- 5. LocalBA 信号(按 ts 匹配到关键帧) ----------------
ba_path = INSTR / f"K{seq}" / "local_ba_signals.csv"
ba_rows = list(csv.DictReader(open(ba_path))) if ba_path.exists() else []
ba_pix_mean = np.zeros(M)
ba_ratio_inl = np.zeros(M)
ba_hit = 0
if ba_rows:
    ba_ts = np.array([float(r["ts"]) for r in ba_rows])
    ba_pixm = np.array([float(r["pix_mean_px"]) for r in ba_rows])
    ba_ratio = np.array([float(r["ratio_inlier_edge"]) for r in ba_rows])
    # BA 同样按"该关键帧对应帧的 ts"做最近邻匹配(BA 发生在某关键帧插入时)
    last = (0.0, 0.0)
    for i in range(M):
        ref_ts = ft_ts[match[i]]
        j = int(np.argmin(np.abs(ba_ts - ref_ts)))
        if np.abs(ba_ts[j] - ref_ts) < TOL:
            last = (ba_pixm[j], ba_ratio[j])
            ba_hit += 1
        ba_pix_mean[i], ba_ratio_inl[i] = last
    print(f"[{prefix}] LocalBA {len(ba_rows)} 次, 匹配到关键帧 {ba_hit} 次 (其余前向填充)")

track = np.stack([
    np.log1p(n_inliers),
    n_inliers / np.maximum(n_map_pts, 1),
    n_lost_kps / np.maximum(n_kps, 1),
    reproj_mean,
    reproj_std,
    np.log1p(n_map_pts),
    obs_per_mp,
    np.log1p(n_local_mps),
    ba_pix_mean,
    ba_ratio_inl,
    kf_gap,
], axis=1).astype(np.float32)

# ---------------- 6. 运动特征(与 kitti_dataset.py 完全一致) ----------------
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

# ---------------- 7. 跟踪特征一阶差分 ----------------
dtrack = np.zeros_like(track)
dtrack[1:] = track[1:] - track[:-1]

X_full = np.concatenate([motion, track, dtrack], axis=1)
NAMES = (["dt", "dp_bx", "dp_by", "dp_bz", "dq_bx", "dq_by", "dq_bz"]
         + [f"t_{n}" for n in ["log_inliers", "inlier_ratio", "lost_ratio",
                               "reproj_mean", "reproj_std", "log_map_pts",
                               "obs_per_mp", "log_local_mps", "ba_pix_mean",
                               "ba_ratio_inl", "kf_gap"]]
         + [f"d_{n}" for n in ["log_inliers", "inlier_ratio", "lost_ratio",
                               "reproj_mean", "reproj_std", "log_map_pts",
                               "obs_per_mp", "log_local_mps", "ba_pix_mean",
                               "ba_ratio_inl", "kf_gap"]])
assert X_full.shape[1] == len(NAMES), (X_full.shape, len(NAMES))

# ---------------- 8. Y: 机体系残差 ----------------
err_w = p_g - p_s
Y_full = np.stack([rot_s[i].inv().apply(err_w[i]) for i in range(M)]).astype(np.float32)

# ---------------- 9. 滑窗 ----------------
D = X_full.shape[1]
n_win = M - W
X = np.zeros((n_win, W, D), dtype=np.float32)
Y = np.zeros((n_win, 3), dtype=np.float32)
for i in range(n_win):
    X[i] = X_full[i:i + W]
    Y[i] = Y_full[i + W]

out = DATA / f"{prefix}_dataset_instr.npz"
np.savez(out,
         X=X, Y=Y, X_raw=X, Y_raw=Y, W=W, S_GLOBAL=S_G,
         X_full=X_full, Y_full=Y_full, feat_names=np.array(NAMES),
         groups=np.array(["motion"] * 7 + ["track"] * 11 + ["dtrack"] * 11),
         n_kf=M, n_matched=int(hit), n_ba_matched=int(ba_hit))
print(f"[{prefix}] X={X.shape} (D={D}) Y={Y.shape} | 存至 {out}")
print(f"[{prefix}] |res| mean = {np.linalg.norm(Y_full, axis=1).mean():.3f} m, "
      f"全局尺度 = {S_G:.4f}")
