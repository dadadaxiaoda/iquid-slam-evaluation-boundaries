void Tracking::LogFrameSignals()
{
    const char* directory = std::getenv("SLAM_QUALITY_DIR");
    if(!directory || !*directory || mSensor != System::IMU_STEREO) return;
    auto began = std::chrono::steady_clock::now();
    QualityContext& c = qualityContext();
    static thread_local std::ofstream csv;
    if(!csv.is_open()) {
        csv.open(std::string(directory)+"/quality.csv");
        if(!csv) throw std::runtime_error("Cannot open quality.csv");
        csv << "input_id,frame_id,ts,frame_ts,state,pose_valid,imu_initialized,map_id,reset_epoch,map_create_epoch,map_change_index,big_change_index,context_changed,last_reloc_id,last_init_id,n_kps,n_associated,n_inlier_observed,n_outlier_associated,n_depth_valid,depth_valid_fraction,n_reproj,reproj_median_px,reproj_p90_px,reproj_mean_px,mean_observations,inlier_grid_coverage,is_kf,n_local_mps,n_local_kfs,imu_rotation_residual_rad,imu_dt,imu_interval_valid,track_core_ms,logger_ms,px,py,pz,qx,qy,qz,qw\n";
        csv << std::fixed << std::setprecision(12);
    }
    const double missing=std::numeric_limits<double>::quiet_NaN();
    Map* map=mpAtlas ? mpAtlas->GetCurrentMap() : nullptr;
    std::unique_lock<std::mutex> mapLock;
    if(map) mapLock=std::unique_lock<std::mutex>(map->mMutexMapUpdate);
    const long mapId=map ? long(map->GetId()) : -1;
    const int change=map ? map->GetMapChangeIndex() : -1;
    const int big=map ? map->GetLastBigChangeIdx() : -1;
    const bool initialized=map && map->isImuInitialized();
    const bool sameContext=c.havePrevious && c.previousMap==mapId && c.previousReset==c.resetEpoch && c.previousCreate==c.createEpoch;
    const bool contextChanged=c.havePrevious && (!sameContext || big!=c.previousBig);
    const bool poseValid=mCurrentFrame.HasPose();
    Sophus::SE3f Twc=poseValid ? mCurrentFrame.GetPose().inverse() : Sophus::SE3f();
    Eigen::Vector3f p=Twc.translation(); Eigen::Quaternionf quaternion=Twc.unit_quaternion();
    int associated=0,inliers=0,outliers=0,depths=0,obsCount=0;
    double obsSum=0; bool grid[48]={false}; std::vector<double> errors;
    const int n=mCurrentFrame.N;
    const bool compatible=mCurrentFrame.Nleft==-1;
    if(compatible && poseValid) {
        errors.reserve(n);
        for(int i=0;i<n;++i) {
            if(i<int(mCurrentFrame.mvDepth.size()) && i<int(mCurrentFrame.mvuRight.size()) && mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvuRight[i]>=0) ++depths;
            if(i>=int(mCurrentFrame.mvpMapPoints.size())) continue;
            MapPoint* point=mCurrentFrame.mvpMapPoints[i];
            if(!point || point->isBad()) continue;
            ++associated;
            if(i>=int(mCurrentFrame.mvbOutlier.size()) || mCurrentFrame.mvbOutlier[i]) {++outliers;continue;}
            const int observations=point->Observations();
            if(observations<=0) continue;
            ++inliers; obsSum+=observations; ++obsCount;
            const cv::Point2f key=mCurrentFrame.mvKeysUn[i].pt;
            if(key.x>=0 && key.x<mImGray.cols && key.y>=0 && key.y<mImGray.rows) {
                int gx=int(key.x*8/mImGray.cols),gy=int(key.y*6/mImGray.rows); grid[gy*8+gx]=true;
            }
            Eigen::Vector3f pc=mCurrentFrame.GetPose()*point->GetWorldPos();
            if(pc(2)<=0) continue;
            Eigen::Vector2f uv=mCurrentFrame.mpCamera->project(pc);
            const double error=std::hypot(double(uv(0)-key.x),double(uv(1)-key.y));
            if(std::isfinite(error)) errors.push_back(error);
        }
    }
    std::sort(errors.begin(),errors.end());
    auto quantile=[&](double fraction)->double {
        if(errors.empty()) return missing;
        double location=fraction*(errors.size()-1); size_t low=size_t(location),high=std::min(low+1,errors.size()-1);
        return errors[low]+(location-low)*(errors[high]-errors[low]);
    };
    double median=quantile(.5),p90=quantile(.9),mean=missing;
    if(!errors.empty()) mean=std::accumulate(errors.begin(),errors.end(),0.)/errors.size();
    int covered=0; for(bool occupied:grid) covered+=occupied;
    double residual=missing,imuDt=missing;
    IMU::Preintegrated* pre=mCurrentFrame.mpImuPreintegratedFrame;
    bool intervalValid=poseValid && mState==OK && initialized && c.previousValid && sameContext && change==c.previousChange && big==c.previousBig && pre;
    Eigen::Matrix3f bodyRotation=poseValid ? mCurrentFrame.GetImuRotation() : Eigen::Matrix3f::Identity();
    if(intervalValid) {
        imuDt=pre->dT;
        intervalValid=imuDt>0 && std::abs(imuDt-(c.inputTs-c.previousTs))<.002;
        if(intervalValid) {
            Eigen::Matrix3f visual=c.previousBody.transpose()*bodyRotation;
            Eigen::Matrix3f inertial=pre->GetDeltaRotation(mCurrentFrame.mImuBias);
            residual=Sophus::SO3f(inertial.transpose()*visual).log().norm();
        }
    }
    const double loggerMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
    csv<<c.inputId<<","<<mCurrentFrame.mnId<<","<<c.inputTs<<","<<mCurrentFrame.mTimeStamp<<","<<int(mState)<<","<<poseValid<<","<<initialized
       <<","<<mapId<<","<<c.resetEpoch<<","<<c.createEpoch<<","<<change<<","<<big<<","<<contextChanged<<","<<mnLastRelocFrameId<<","<<((c.resetEpoch||c.createEpoch)?long(mnLastInitFrameId):-1)
       <<","<<n<<","<<associated<<","<<inliers<<","<<outliers<<","<<depths<<","<<(n>0&&poseValid&&compatible?double(depths)/n:missing)
       <<","<<errors.size()<<","<<median<<","<<p90<<","<<mean<<","<<(obsCount?obsSum/obsCount:missing)<<","<<(inliers?double(covered)/48:missing)
       <<","<<(mnInstrLastKFId==int(mCurrentFrame.mnId))<<","<<mvpLocalMapPoints.size()<<","<<mvpLocalKeyFrames.size()
       <<","<<residual<<","<<imuDt<<","<<intervalValid<<","<<c.trackMs<<","<<loggerMs
       <<","<<(poseValid?p.x():missing)<<","<<(poseValid?p.y():missing)<<","<<(poseValid?p.z():missing)
       <<","<<(poseValid?quaternion.x():missing)<<","<<(poseValid?quaternion.y():missing)<<","<<(poseValid?quaternion.z():missing)<<","<<(poseValid?quaternion.w():missing)<<"\n";
    if(c.inputId%100==0) csv.flush();
    c.havePrevious=true; c.previousMap=mapId; c.previousReset=c.resetEpoch; c.previousCreate=c.createEpoch;
    c.previousBig=big; c.previousChange=change; c.previousTs=c.inputTs;
    c.previousValid=poseValid && mState==OK && initialized;
    c.previousBody=bodyRotation;
}
