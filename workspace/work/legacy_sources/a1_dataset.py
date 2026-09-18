#!/usr/bin/env python3
"""
A1 数据集泛化版: 从 {seq}_scale_label.npz + IMU csv 构建滑动窗口数据集
特征 19 维 = 7 维运动增量(机体系) + 12 维 IMU(积分+均值)
输出 Y = 机体系残差 R_i^T (p_gt - p_slam_sim3对齐)  3 维
用法: python a1_dataset.py <seq>
"""
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rrot

seq = sys.argv[1]
DATA = Path("/opt/slam-study/legacy/data")
W = 20

d = np.load(DATA / f"{seq}_scale_label.npz")
t = d["t"].astype(np.float64)
p_s_raw = d["p_slam"].astype(np.float64)
p_g = d["p_gt"].astype(np.float64)
q_s = d["q_slam"].astype(np.float64)
N = len(t)

# ---------- Sim3 对齐(得到对齐后轨迹) ----------
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
p_s = (S_G * (R_G @ p_s_raw.T).T + T_G)      # 对齐后
rot_s = Rrot.from_quat(q_s)

# ---------- IMU ----------
imu_raw = np.loadtxt(f"/opt/slam-study/datasets/{seq}/mav0/imu0/data.csv",
                     delimiter=",", comments="#", usecols=range(7))
t_imu_abs = imu_raw[:, 0] / 1e9
imu = imu_raw[:, 1:7].astype(np.float32)      # wx wy wz ax ay az
# 绝对参考: TUM 文件首行时间戳 = SLAM 首关键帧绝对时刻
t0_abs = None
for line in open(DATA / f"{seq}_slam_tum.txt"):
    v = line.split()
    if len(v) >= 8:
        t0_abs = float(v[0])
        break
assert t0_abs is not None
t_imu = t_imu_abs - t0_abs                     # 与 t 同轴(相对 SLAM 首帧)
print(f"IMU 时间轴: [{t_imu[0]:.2f}, {t_imu[-1]:.2f}] 覆盖关键帧 [{t[0]:.2f}, {t[-1]:.2f}]")

imu_feats = np.zeros((N, 12), dtype=np.float32)
for i in range(1, N):
    t0, t1 = t[i-1], t[i]
    mask = (t_imu >= t0) & (t_imu <= t1)
    if mask.sum() < 2:
        for k in range(6):
            v = np.interp(t1, t_imu, imu[:, k])
            imu_feats[i, k] = v * (t1 - t0)
            imu_feats[i, k+6] = v
        continue
    seg = imu[mask]; ts = t_imu[mask]
    dt = np.diff(ts, prepend=ts[0] - 1e-9).astype(np.float32)
    imu_feats[i] = np.concatenate([(seg * dt[:, None]).sum(0), seg.mean(0)])
imu_feats[0] = imu_feats[1]

# ---------- 19 维特征 ----------
motion = np.zeros((N, 7), dtype=np.float32)
motion[1:, 0] = np.diff(t).astype(np.float32)
dps = p_s[1:] - p_s[:-1]
motion[1:, 1:4] = np.stack([rot_s[i].inv().apply(dps[i-1]) for i in range(1, N)])
dq = (rot_s[1:] * rot_s[:-1].inv()).as_rotvec()
motion[1:, 4:7] = dq.astype(np.float32)
motion[0] = motion[1]
X_full = np.concatenate([motion, imu_feats], axis=1)

# ---------- Y: 机体系残差 ----------
err_w = p_g - p_s
Y_full = np.stack([rot_s[i].inv().apply(err_w[i]) for i in range(N)]).astype(np.float32)

# ---------- 滑窗 ----------
X = np.zeros((N - W, W, 19), dtype=np.float32)
Y = np.zeros((N - W, 3), dtype=np.float32)
for i in range(N - W):
    X[i] = X_full[i:i+W]
    Y[i] = Y_full[i+W]

n_train = int(0.8 * len(X))
xm, xs = float(X[:n_train].mean()), float(X[:n_train].std())
ym, ys = Y[:n_train].mean(0), Y[:n_train].std(0)
np.savez(DATA / f"{seq}_dataset_imu.npz",
         X=(X - xm) / xs, Y=(Y - ym) / ys,
         x_mean=xm, x_std=xs, y_mean=ym, y_std=ys,
         X_raw=X, Y_raw=Y, W=W, S_GLOBAL=S_G)
print(f"[{seq}] X={X.shape} Y={Y.shape} |Y|mean={np.linalg.norm(Y_full,axis=1).mean():.3f}m "
      f"IMU重叠检查: t_imu[{t_imu[0]:.1f},{t_imu[-1]:.1f}] vs t[{t[0]:.1f},{t[-1]:.1f}]")
