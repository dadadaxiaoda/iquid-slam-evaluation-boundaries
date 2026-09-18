from pathlib import Path
import subprocess, shlex, hashlib, json, os, resource, time
ROOT=Path(__file__).resolve().parent
repo=Path('/opt/slam-study/ORB_SLAM3'); build=repo/'build'
orig=repo/'Examples/Stereo-Inertial/stereo_inertial_euroc.cc'
config=repo/'Examples/Stereo-Inertial/EuRoC.yaml'
protected=[orig,config,repo/'lib/libORB_SLAM3.so']
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
(ROOT/'protected_before.json').write_text(json.dumps(manifest,indent=2))
src=orig.read_text()
assert 'IMU_STEREO, false' in src
needle='    cv::Mat imLeft, imRight;'
src=src.replace(needle,'''    ofstream online("online.csv");
    online << "ts,state,px,py,pz,qx,qy,qz,qw\\n" << fixed << setprecision(12);
'''+needle)
needle='SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas);'
assert src.count(needle)==1
src=src.replace(needle,'''Sophus::SE3f Tcw = SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas);
            Sophus::SE3f Twc = Tcw.inverse();
            Eigen::Vector3f p = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            online << tframe << "," << SLAM.GetTrackingState() << "," << p.x() << "," << p.y() << "," << p.z()
                   << "," << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << endl;''')
target=ROOT/'stereo_inertial_headless.cc'; target.write_text(src)
flags=(build/'CMakeFiles/stereo_inertial_euroc.dir/flags.make').read_text().splitlines()
includes=[]
for line in flags:
 if line.startswith(('CXX_INCLUDES =','CXX_DEFINES =')): includes+=shlex.split(line.split('=',1)[1])
args=shlex.split((build/'CMakeFiles/stereo_inertial_euroc.dir/link.txt').read_text()); args[1:1]=includes
for i,a in enumerate(args):
 if a.endswith('stereo_inertial_euroc.cc.o'): args[i]=str(target)
 if a.startswith('../') and i and args[i-1]!='-o': args[i]=str((build/a).resolve())
args[args.index('-o')+1]=str(ROOT/'stereo_inertial_headless')
with (ROOT/'compile.log').open('w') as log: subprocess.run(args,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
resource.setrlimit(resource.RLIMIT_CORE,(0,0)); records=[]
for name in ['V101','V201','V202']:
 dst=ROOT/name; dst.mkdir(exist_ok=True)
 dataset=Path('/opt/slam-study/datasets')/name
 stamps=dst/'timestamps.txt'
 stamps.write_text('\n'.join(l.split(',')[0] for l in (dataset/'mav0/cam0/data.csv').read_text().splitlines() if l and not l.startswith('#'))+'\n')
 env=os.environ.copy(); env['ORB_SLAM3_INSTR_DIR']=str(dst)
 env['DISPLAY']=''; env['WAYLAND_DISPLAY']=''; env['OMP_NUM_THREADS']='2'
 cmd=[str(ROOT/'stereo_inertial_headless'),str(repo/'Vocabulary/ORBvoc.txt'),str(config),str(dataset),str(stamps),name]
 print('START',name,flush=True); started=time.time()
 with (dst/'run.log').open('w') as log:
  result=subprocess.run(cmd,cwd=dst,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 records.append({'sequence':name,'exit_code':result.returncode,'seconds':time.time()-started})
 (ROOT/'replay_status.json').write_text(json.dumps(records,indent=2))
 print('DONE',records[-1],flush=True)
 if result.returncode: break
checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items()}
(ROOT/'protected_after.json').write_text(json.dumps(checks,indent=2)); assert all(checks.values())
