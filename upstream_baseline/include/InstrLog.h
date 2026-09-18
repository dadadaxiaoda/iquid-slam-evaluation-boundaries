/**
 * ORB-SLAM3 跟踪信号埋点模块（LNN 漂移补偿项目专用）
 *
 * 设计原则：非侵入式
 *   - 仅新增本头文件，不改动任何既有类结构（除 Tracking 增加 2 个成员/方法）
 *   - 通过环境变量 ORB_SLAM3_INSTR_DIR 启用；未设置时调用点立即 return，零开销
 *   - 输出两个 CSV：
 *       <INSTR_DIR>/frame_signals.csv       每帧一行（帧级跟踪信号）
 *       <INSTR_DIR>/local_ba_signals.csv    每次 Local BA 一行（BA 级信号）
 *
 * 覆盖用户要求的 4 类信号：
 *   1. 当前匹配数            -> frame_signals.n_inliers / n_map_pts
 *   2. LocalBA 重投影误差     -> local_ba_signals.pix_mean_px / pix_std_px
 *   3. 地图点观测数          -> frame_signals.n_local_mps / n_obs_sum
 *   4. 关键帧插入频率        -> frame_signals.is_kf（每帧是否插KF）
 */

#ifndef ORB_SLAM3_INSTR_LOG_H
#define ORB_SLAM3_INSTR_LOG_H

#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

namespace ORB_SLAM3 {
namespace instr {

// ------------------------------------------------------------------
// 文件句柄（单例，惰性初始化）
// ------------------------------------------------------------------
struct InstrFiles {
    bool active;
    std::ofstream frame;
    std::ofstream ba;
    std::ofstream vi;
    std::mutex mtx;
    InstrFiles() : active(false) {}
};

inline InstrFiles& files() {
    static InstrFiles s;
    static std::once_flag once;
    std::call_once(once, []() {
        const char* dir = std::getenv("ORB_SLAM3_INSTR_DIR");
        if (!dir || !dir[0]) return;
        std::string d(dir);
        s.frame.open((d + "/frame_signals.csv").c_str(), std::ios::out);
        s.ba.open((d + "/local_ba_signals.csv").c_str(), std::ios::out);
        s.vi.open((d + "/vi_signals.csv").c_str(), std::ios::out);
        if (!s.frame.is_open() || !s.ba.is_open() || !s.vi.is_open()) {
            s.active = false;
            return;
        }
        // 帧级表头
        s.frame << "frame_id,ts,state,n_kps,n_map_pts,n_inliers,n_lost_kps,"
                   "reproj_mean_px,reproj_std_px,reproj_median_px,n_reproj,"
                   "obs_per_mp,n_local_mps,n_local_kfs,is_kf,"
                   "n_global_kfs,n_global_mps,image_name\n";
        // BA 级表头
        s.ba << "kf_id,ts,n_fixed_kf,n_opt_kf,n_mps,n_edges,n_inlier_edges,"
                "ratio_inlier_edge,chi2_mean,chi2_std,"
                "pix_mean_px,pix_std_px,pix_median_px,pix_p90_px,opt_ms\n";
        // 单目惯性级表头(跨模态不一致: IMU预积分 vs 视觉估计)
        s.vi << "kf_id,ts,n_inertial_edges,n_vis_edges,"
                "err_R_mean,err_R_std,err_R_max,"
                "err_V_mean,err_V_std,err_V_max,"
                "err_P_mean,err_P_std,err_P_max,"
                "chi2_inertial_mean,chi2_inertial_max,rel_robust_chi2,"
                "n_gyro_rw,n_acc_rw,"
                "bg_x,bg_y,bg_z,ba_x,ba_y,ba_z,"
                "vel_x,vel_y,vel_z\n";
        // 精度：默认 6 位有效数字会让 t>1s 的时间戳只剩 ~1e-4 s 精度，
        // 下游按时间戳做关键帧对齐时会失败。提高到 12 位有效数字。
        s.frame << std::setprecision(12);
        s.ba << std::setprecision(12);
        s.vi << std::setprecision(12);
        s.active = true;
    });
    return s;
}

// ------------------------------------------------------------------
// 分布统计
// ------------------------------------------------------------------
struct DistStats {
    int n;
    double mean, stdv, median, p90;
    DistStats() : n(0), mean(0), stdv(0), median(0), p90(0) {}
};

inline DistStats distStats(std::vector<float>& v) {
    DistStats s;
    s.n = static_cast<int>(v.size());
    if (s.n == 0) return s;
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) sum += v[i];
    s.mean = sum / s.n;
    double var = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        double dd = v[i] - s.mean;
        var += dd * dd;
    }
    s.stdv = (s.n > 1) ? std::sqrt(var / (s.n - 1)) : 0.0;
    std::sort(v.begin(), v.end());
    s.median = v[v.size() / 2];
    size_t idx = static_cast<size_t>(0.9 * (v.size() - 1));
    s.p90 = v[idx];
    return s;
}

// ------------------------------------------------------------------
// 帧级信号写入
// ------------------------------------------------------------------
struct FrameSignal {
    int frameId;
    double ts;
    int state;
    int nKps;          // ORB 特征点数
    int nMapPts;       // 成功关联到地图点的特征数
    int nInliers;      // 跟踪内点数 (mnMatchesInliers)
    int nLostKps;      // 未关联上地图点的特征数
    double reprojMean; // 当前帧重投影误差 (像素)
    double reprojStd;
    double reprojMedian;
    int nReproj;       // 参与误差统计的点数
    double obsPerMp;   // 平均地图点观测次数
    int nLocalMps;     // 局部地图点数
    int nLocalKfs;     // 局部关键帧数
    int isKf;          // 本帧是否插入关键帧
    int nGlobalKfs;    // 全局关键帧数
    int nGlobalMps;    // 全局地图点数
    std::string imgName;
    FrameSignal() : frameId(-1), ts(0), state(-1), nKps(0), nMapPts(0),
        nInliers(0), nLostKps(0), reprojMean(0), reprojStd(0), reprojMedian(0),
        nReproj(0), obsPerMp(0), nLocalMps(0), nLocalKfs(0), isKf(0),
        nGlobalKfs(0), nGlobalMps(0) {}
};

inline void logFrame(const FrameSignal& f) {
    InstrFiles& s = files();
    if (!s.active) return;
    std::lock_guard<std::mutex> lk(s.mtx);
    s.frame << f.frameId << "," << f.ts << "," << f.state << ","
            << f.nKps << "," << f.nMapPts << "," << f.nInliers << ","
            << f.nLostKps << ","
            << f.reprojMean << "," << f.reprojStd << "," << f.reprojMedian << ","
            << f.nReproj << "," << f.obsPerMp << ","
            << f.nLocalMps << "," << f.nLocalKfs << "," << f.isKf << ","
            << f.nGlobalKfs << "," << f.nGlobalMps << ","
            << f.imgName << "\n";
    // 每帧立即落盘：避免进程异常退出(如ORB-SLAM3退出时的segfault)导致缓冲数据丢失
    s.frame.flush();
}

// ------------------------------------------------------------------
// Local BA 信号写入
// ------------------------------------------------------------------
struct BASignal {
    int kfId;
    double ts;
    int nFixedKf, nOptKf, nMps, nEdges, nInlierEdges;
    double chi2Mean, chi2Std;
    double pixMean, pixStd, pixMedian, pixP90;
    double optMs;
    BASignal() : kfId(-1), ts(0), nFixedKf(0), nOptKf(0), nMps(0), nEdges(0),
        nInlierEdges(0), chi2Mean(0), chi2Std(0), pixMean(0), pixStd(0),
        pixMedian(0), pixP90(0), optMs(0) {}
};

inline void logBA(const BASignal& b) {
    InstrFiles& s = files();
    if (!s.active) return;
    double ratio = (b.nEdges > 0) ? (double)b.nInlierEdges / b.nEdges : 0.0;
    std::lock_guard<std::mutex> lk(s.mtx);
    s.ba << b.kfId << "," << b.ts << ","
         << b.nFixedKf << "," << b.nOptKf << "," << b.nMps << ","
         << b.nEdges << "," << b.nInlierEdges << "," << ratio << ","
         << b.chi2Mean << "," << b.chi2Std << ","
         << b.pixMean << "," << b.pixStd << ","
         << b.pixMedian << "," << b.pixP90 << ","
         << b.optMs << "\n";
    s.ba.flush();
}

// ------------------------------------------------------------------
// 单目惯性: 跨模态不一致信号写入
//
// 数据来源: Optimizer::LocalInertialBA 里 g2o::EdgeInertial 的残差
//   _error = [ er(3, rad) , ev(3, m/s) , ep(3, m) ]
// 含义: IMU 预积分外推出的相对运动 与 视觉优化出的相对运动 之差。
// 这个量**在线可算**(不需要 GT), 是"视觉-惯性互相不信任程度"的直接度量,
// 因此是"能否在线预测漂移"最有力的候选特征。
// 同时记录当前关键帧的陀螺/加计零偏与速度(偏置漂移是漂移的经典驱动因子)。
// ------------------------------------------------------------------
struct VISignal {
    int kfId;
    double ts;
    int nInertialEdges, nVisEdges;
    double errRMean, errRStd, errRMax;
    double errVMean, errVStd, errVMax;
    double errPMean, errPStd, errPMax;
    double chi2InertialMean, chi2InertialMax;
    double relRobustChi2;
    int nGyroRW, nAccRW;
    double bgx, bgy, bgz;
    double bax, bay, baz;
    double vx, vy, vz;
    VISignal() : kfId(-1), ts(0), nInertialEdges(0), nVisEdges(0),
        errRMean(0), errRStd(0), errRMax(0),
        errVMean(0), errVStd(0), errVMax(0),
        errPMean(0), errPStd(0), errPMax(0),
        chi2InertialMean(0), chi2InertialMax(0), relRobustChi2(0),
        nGyroRW(0), nAccRW(0),
        bgx(0), bgy(0), bgz(0), bax(0), bay(0), baz(0),
        vx(0), vy(0), vz(0) {}
};

inline void logVI(const VISignal& v) {
    InstrFiles& s = files();
    if (!s.active) return;
    std::lock_guard<std::mutex> lk(s.mtx);
    s.vi << v.kfId << "," << v.ts << ","
         << v.nInertialEdges << "," << v.nVisEdges << ","
         << v.errRMean << "," << v.errRStd << "," << v.errRMax << ","
         << v.errVMean << "," << v.errVStd << "," << v.errVMax << ","
         << v.errPMean << "," << v.errPStd << "," << v.errPMax << ","
         << v.chi2InertialMean << "," << v.chi2InertialMax << ","
         << v.relRobustChi2 << ","
         << v.nGyroRW << "," << v.nAccRW << ","
         << v.bgx << "," << v.bgy << "," << v.bgz << ","
         << v.bax << "," << v.bay << "," << v.baz << ","
         << v.vx << "," << v.vy << "," << v.vz << "\n";
    s.vi.flush();
}

}  // namespace instr
}  // namespace ORB_SLAM3

#endif  // ORB_SLAM3_INSTR_LOG_H
