    // Both poses are reconstructed using reference keyframes as they exist NOW.
    // No final-export/future optimization is used by this online callback.
    static thread_local std::ofstream pairCsv;
    if(!pairCsv.is_open()) {
        pairCsv.open(std::string(directory)+"/relative_pairs.csv");
        if(!pairCsv) throw std::runtime_error("Cannot open relative_pairs.csv");
        pairCsv<<"input_id,start_ts,end_ts,pair_valid,imu_initialized,imu_ba1,imu_ba2,map_id,reset_epoch,map_create_epoch,dx,dy,dz,qx,qy,qz,qw\n";
        pairCsv<<std::fixed<<std::setprecision(12);
    }
    bool pairValid=poseValid && mState==OK && initialized && c.recentStates.size()==21;
    for(const auto& entry:c.recentStates) if(entry.second!=int(OK)) pairValid=false;
    if(pairValid && std::abs(c.inputTs-c.recentStates.front().first-1.)>.002) pairValid=false;
    double startTs=missing;
    Sophus::SE3f relativeMotion;
    if(pairValid && mlRelativeFramePoses.size()>=21 && mlpReferences.size()==mlRelativeFramePoses.size() && mlFrameTimes.size()==mlRelativeFramePoses.size()) {
        auto firstRelative=mlRelativeFramePoses.end();std::advance(firstRelative,-21);
        auto firstReference=mlpReferences.end();std::advance(firstReference,-21);
        auto firstTime=mlFrameTimes.end();std::advance(firstTime,-21);
        startTs=*firstTime;
        pairValid=std::abs(startTs-c.recentStates.front().first)<.002 && std::abs(mlFrameTimes.back()-c.inputTs)<.002;
        auto reconstruct=[&](const Sophus::SE3f& relative,KeyFrame* reference,Sophus::SE3f& pose)->bool {
            Sophus::SE3f correction;
            int traversed=0;
            while(reference && reference->isBad()) {
                if(++traversed>1000) return false;
                correction=correction*reference->mTcp;
                reference=reference->GetParent();
            }
            if(!reference || reference->GetMap()!=map) return false;
            pose=relative*correction*reference->GetPose();
            return true;
        };
        Sophus::SE3f firstPose,currentPose;
        if(pairValid) pairValid=reconstruct(*firstRelative,*firstReference,firstPose) && reconstruct(mlRelativeFramePoses.back(),mlpReferences.back(),currentPose);
        if(pairValid) relativeMotion=firstPose*currentPose.inverse();
    } else pairValid=false;
    Eigen::Vector3f relativeP=relativeMotion.translation();Eigen::Quaternionf relativeQ=relativeMotion.unit_quaternion();
    pairCsv<<c.inputId<<","<<startTs<<","<<c.inputTs<<","<<pairValid<<","<<initialized<<","<<(map&&map->GetIniertialBA1())<<","<<(map&&map->GetIniertialBA2())
           <<","<<mapId<<","<<c.resetEpoch<<","<<c.createEpoch
           <<","<<(pairValid?relativeP.x():missing)<<","<<(pairValid?relativeP.y():missing)<<","<<(pairValid?relativeP.z():missing)
           <<","<<(pairValid?relativeQ.x():missing)<<","<<(pairValid?relativeQ.y():missing)<<","<<(pairValid?relativeQ.z():missing)<<","<<(pairValid?relativeQ.w():missing)<<"\n";
    if(c.inputId%100==0) pairCsv.flush();
