"""Compile isolated headless example and collect matching mono instrumentation, no install."""
from pathlib import Path
import subprocess,shlex,resource,json,time,hashlib,os
root=Path(__file__).resolve().parent/'matched'; root.mkdir(exist_ok=True)
repo=Path('/opt/slam-study/ORB_SLAM3'); build=repo/'build'
orig=repo/'Examples/Monocular/mono_euroc.cc'; target=root/'mono_euroc_headless.cc'
text=orig.read_text(); old='ORB_SLAM3::System::MONOCULAR, true'
assert text.count(old)==1
text=text.replace(old,'ORB_SLAM3::System::MONOCULAR, false')
initial='ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, false);'
assert text.count(initial)==1
text=text.replace(initial,initial+'\n    ofstream onlinePose("online_camera_tum.txt");\n    onlinePose << fixed << setprecision(12);\n    ofstream onlineState("online_state.csv");\n    onlineState << "timestamp,state\\n" << fixed << setprecision(12);')
track='SLAM.TrackMonocular(im,tframe); // TODO change to monocular_inertial'
assert text.count(track)==1
text=text.replace(track,'''Sophus::SE3f onlineTcw = SLAM.TrackMonocular(im,tframe);
            int onlineTrackingState = SLAM.GetTrackingState();
            onlineState << tframe << "," << onlineTrackingState << "\\n";
            if (onlineTrackingState == 2) {
                Sophus::SE3f onlineTwc = onlineTcw.inverse();
                Eigen::Vector3f p = onlineTwc.translation();
                Eigen::Quaternionf q = onlineTwc.unit_quaternion();
                onlinePose << tframe << " " << p(0) << " " << p(1) << " " << p(2) << " "
                           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
            }''')
target.write_text(text)
flags=(build/'CMakeFiles/mono_euroc.dir/flags.make').read_text().splitlines()
include=[]
for line in flags:
 if line.startswith('CXX_INCLUDES =') or line.startswith('CXX_DEFINES ='): include+=shlex.split(line.split('=',1)[1])
args=shlex.split((build/'CMakeFiles/mono_euroc.dir/link.txt').read_text()); args[1:1]=include
for i,arg in enumerate(args):
 if arg.endswith('mono_euroc.cc.o'): args[i]=str(target)
 if arg.startswith('../') and i and args[i-1]!='-o': args[i]=str((build/arg).resolve())
args[args.index('-o')+1]=str(root/'mono_euroc_headless')
manifest=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [orig,repo/'lib/libORB_SLAM3.so',repo/'Examples/Monocular/EuRoC.yaml']]
(root/'compile_args.json').write_text(json.dumps(args,indent=2))
with (root/'compile.log').open('w') as log: subprocess.run(args,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
resource.setrlimit(resource.RLIMIT_CORE,(0,0)); statuses=[]
for name in ['V101','V102','V201','V103','V202']:
 dst=root/name; dst.mkdir(exist_ok=True); env=os.environ.copy(); env['ORB_SLAM3_INSTR_DIR']=str(dst)
 stamp=repo/'Examples/Monocular/EuRoC_TimeStamps'/f'{name}.txt'
 if not stamp.exists():
  raw=Path('/opt/slam-study/datasets')/name/'mav0/cam0/data.csv'
  stamp=dst/'timestamps.txt'; stamp.write_text('\n'.join(line.split(',')[0] for line in raw.read_text().splitlines() if line and not line.startswith('#'))+'\n')
 command=[str(root/'mono_euroc_headless'),str(repo/'Vocabulary/ORBvoc.txt'),str(repo/'Examples/Monocular/EuRoC.yaml'),str(Path('/opt/slam-study/datasets')/name),str(stamp),name]
 print('REPLAY_START',name,flush=True); start=time.time()
 with (dst/'run.log').open('w') as log: result=subprocess.run(command,cwd=dst,env=env,stdout=log,stderr=subprocess.STDOUT)
 f=dst/'online_camera_tum.txt'; quality=dst/'frame_signals.csv'
 rec={'sequence':name,'exit_code':result.returncode,'seconds':time.time()-start,'frame_trajectory_exists':f.exists(),'quality_exists':quality.exists()}
 if f.exists():
  import numpy as np
  a=np.loadtxt(f); a=np.atleast_2d(a)
  if np.median(a[:,0])>1e12: a[:,0]/=1e9
  np.savetxt(dst/f'{name}_slam_tum.txt',a,fmt='%.12f'); rec['trajectory_rows']=len(a)
 statuses.append(rec); (root/'replay_status.json').write_text(json.dumps(statuses,indent=2)); print('REPLAY_DONE',json.dumps(rec),flush=True)
 if result.returncode!=0: print('NONZERO_REPLAY_REQUIRES_REVIEW',name,flush=True)
for entry in manifest: entry['unchanged']=hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
assert all(e['unchanged'] for e in manifest)
(root/'original_build_integrity.json').write_text(json.dumps(manifest,indent=2))
print('COLLECTION_COMPLETE',flush=True)
