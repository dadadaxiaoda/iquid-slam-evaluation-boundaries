from pathlib import Path
import subprocess,shlex,hashlib,json,os,time,resource,difflib
ROOT=Path(__file__).resolve().parent
repo=Path('/opt/slam-study/ORB_SLAM3'); build=repo/'build'
source=repo/'src/Tracking.cc'; example=repo/'Examples/Stereo-Inertial/stereo_inertial_euroc.cc'
config=repo/'Examples/Stereo-Inertial/EuRoC.yaml'
protected=list((repo/'src').rglob('*'))+list((repo/'include').rglob('*'))+[example,config,repo/'lib/libORB_SLAM3.so']
protected=[p for p in protected if p.is_file()]
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
(ROOT/'original_manifest.json').write_text(json.dumps(manifest,indent=2))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
text=source.read_text()
helper='''
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <cstdlib>
namespace {
struct QualityContext {
 long inputId=-1, previousMap=-1;
 double inputTs=0,previousTs=0,trackMs=0;
 unsigned resetEpoch=0,createEpoch=0,previousReset=0,previousCreate=0;
 int previousBig=-1,previousChange=-1;
 bool havePrevious=false,previousValid=false;
 std::deque<std::pair<double,int>> recentStates;
 Eigen::Matrix3f previousBody=Eigen::Matrix3f::Identity();
};
QualityContext& qualityContext(){static thread_local QualityContext context;return context;}
}
'''
text=text.replace('using namespace std;',helper+'\nusing namespace std;',1)
start=text.index('Sophus::SE3f Tracking::GrabImageStereo('); end=text.index('Sophus::SE3f Tracking::GrabImageRGBD(',start)
stereo=text[start:end]
assert stereo.count('    Track();')==1
stereo=stereo.replace('    Track();','''    QualityContext& quality=qualityContext();
    ++quality.inputId; quality.inputTs=timestamp;
    auto trackingBegan=std::chrono::steady_clock::now();
    Track();
    quality.trackMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-trackingBegan).count();
    LogFrameSignals();''')
text=text[:start]+stereo+text[end:]
start=text.index('void Tracking::LogFrameSignals()'); end=text.index('void Tracking::GrabImuData(',start)
text=text[:start]+(ROOT/'quality_body.inc').read_text()+'\n\n'+text[end:]
for signature,statement in [('void Tracking::Reset(bool bLocMap)','++qualityContext().resetEpoch;'),('void Tracking::ResetActiveMap(bool bLocMap)','++qualityContext().resetEpoch;'),('void Tracking::CreateMapInAtlas()','++qualityContext().createEpoch;')]:
 needle=signature+'\n{'; assert text.count(needle)==1; text=text.replace(needle,needle+'\n    '+statement)
target=ROOT/'Tracking_quality.cc';target.write_text(text)
(ROOT/'tracking_changes.diff').write_text(''.join(difflib.unified_diff(source.read_text().splitlines(True),text.splitlines(True),fromfile=str(source),tofile=str(target))))
def flags(name):
 result=[]
 for line in (build/f'CMakeFiles/{name}.dir/flags.make').read_text().splitlines():
  if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')): result+=shlex.split(line.split('=',1)[1])
 return result
def run(command,label,cwd=build):
 print(label,flush=True)
 (ROOT/f'{label}_args.json').write_text(json.dumps(command,indent=2))
 with (ROOT/f'{label}.log').open('w') as log: subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
run(['/usr/bin/c++']+flags('ORB_SLAM3')+['-c',str(target),'-o',str(ROOT/'Tracking_quality.o')],'compile_tracking')
libdir=ROOT/'lib';libdir.mkdir(exist_ok=True)
args=shlex.split((build/'CMakeFiles/ORB_SLAM3.dir/link.txt').read_text())
for i,arg in enumerate(args):
 if arg.endswith('/Tracking.cc.o'):args[i]=str(ROOT/'Tracking_quality.o')
 elif arg.startswith('CMakeFiles/') or (arg.startswith('../') and args[i-1]!='-o'):args[i]=str((build/arg).resolve())
args[args.index('-o')+1]=str(libdir/'libORB_SLAM3.so')
run(args,'link_isolated_library')
ex=example.read_text(); assert 'IMU_STEREO, false' in ex
ex=ex.replace('    cv::Mat imLeft, imRight;','''    ofstream online("online.csv");
    online << "input_id,ts,state,px,py,pz,qx,qy,qz,qw,api_ms\\n" << fixed << setprecision(12);
    cv::Mat imLeft, imRight;''')
needle='SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas);';assert ex.count(needle)==1
ex=ex.replace(needle,'''Sophus::SE3f Tcw=SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas);
            double apiMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t1).count();
            Sophus::SE3f Twc=Tcw.inverse(); Eigen::Vector3f p=Twc.translation(); Eigen::Quaternionf q=Twc.unit_quaternion();
            online << ni << "," << tframe << "," << SLAM.GetTrackingState() << "," << p.x() << "," << p.y() << "," << p.z()
                   << "," << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "," << apiMs << "\\n";''')
extarget=ROOT/'stereo_inertial_quality.cc';extarget.write_text(ex)
args=shlex.split((build/'CMakeFiles/stereo_inertial_euroc.dir/link.txt').read_text());args[1:1]=[f for f in flags('stereo_inertial_euroc') if f.startswith('-I') or f.startswith('-D')]
# Preserve isystem pairs separately from the original flags file.
for line in (build/'CMakeFiles/stereo_inertial_euroc.dir/flags.make').read_text().splitlines():
 if line.startswith('CXX_INCLUDES ='):
  inc=shlex.split(line.split('=',1)[1]); args[1:1]=inc
for i,arg in enumerate(args):
 if arg.endswith('stereo_inertial_euroc.cc.o'):args[i]=str(extarget)
 elif arg=='../lib/libORB_SLAM3.so':args[i]=str(libdir/'libORB_SLAM3.so')
 elif arg.startswith('../') and args[i-1]!='-o':args[i]=str((build/arg).resolve())
 elif arg.startswith('-Wl,-rpath,'):args[i]=arg.replace(str(repo/'lib'),str(libdir))
args[args.index('-o')+1]=str(ROOT/'stereo_inertial_quality')
run(args,'compile_example')
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(libdir)+':'+env.get('LD_LIBRARY_PATH','');env['DISPLAY']='';env['WAYLAND_DISPLAY']='';env['OMP_NUM_THREADS']='2'
linked=subprocess.check_output(['ldd',str(ROOT/'stereo_inertial_quality')],env=env,text=True)
(ROOT/'linked_libraries.txt').write_text(linked); assert str(libdir/'libORB_SLAM3.so') in linked
name='V202';dst=ROOT/name;dst.mkdir(exist_ok=True);env['SLAM_QUALITY_DIR']=str(dst);env['ORB_SLAM3_INSTR_DIR']=str(dst)
dataset=Path('/opt/slam-study/datasets')/name;stamps=dst/'timestamps.txt'
stamps.write_text('\n'.join(l.split(',')[0] for l in (dataset/'mav0/cam0/data.csv').read_text().splitlines() if l and not l.startswith('#'))+'\n')
cmd=[str(ROOT/'stereo_inertial_quality'),str(repo/'Vocabulary/ORBvoc.txt'),str(config),str(dataset),str(stamps),name]
started=time.time();print('REPLAY_START',name,flush=True)
with (dst/'run.log').open('w') as log:result=subprocess.run(cmd,cwd=dst,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
(ROOT/'replay_status.json').write_text(json.dumps({'sequence':name,'exit_code':result.returncode,'seconds':time.time()-started,'viewer':False,'training_runs':0},indent=2))
checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items()}
(ROOT/'original_integrity.json').write_text(json.dumps(checks,indent=2));assert all(checks.values())
print('REPLAY_DONE',result.returncode,flush=True);assert result.returncode==0
