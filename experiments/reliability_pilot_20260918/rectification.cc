#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){
 cv::FileStorage f(argv[1],cv::FileStorage::READ);
 auto k=[&](std::string c)->cv::Mat{return (cv::Mat_<double>(3,3)<<double(float(f[c+".fx"])),0,double(float(f[c+".cx"])),0,double(float(f[c+".fy"])),double(float(f[c+".cy"])),0,0,1);};
 auto d=[&](std::string c)->cv::Mat{return (cv::Mat_<float>(4,1)<<float(f[c+".k1"]),float(f[c+".k2"]),float(f[c+".p1"]),float(f[c+".p2"]));};
 cv::Mat tlr;f["Stereo.T_c1_c2"]>>tlr;
 cv::Mat rtl=tlr.inv(),r1,r2,p1,p2,q; rtl.convertTo(rtl,CV_64F);
 cv::stereoRectify(k("Camera1"),d("Camera1"),k("Camera2"),d("Camera2"),cv::Size(752,480),rtl(cv::Rect(0,0,3,3)),rtl(cv::Rect(3,0,1,3)),r1,r2,p1,p2,q,cv::CALIB_ZERO_DISPARITY,-1,cv::Size(752,480));
 std::cout<<std::setprecision(17)<<"[";
 for(int i=0;i<3;i++){if(i)std::cout<<",";std::cout<<"[";for(int j=0;j<3;j++){if(j)std::cout<<",";std::cout<<r1.at<double>(i,j);}std::cout<<"]";}
 std::cout<<"]\n";
}
