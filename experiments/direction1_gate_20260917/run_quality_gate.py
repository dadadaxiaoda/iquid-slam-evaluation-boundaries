"""Run gate experiment on fresh real-time mono pose and matching frame quality logs."""
from pathlib import Path
root=Path(__file__).resolve().parent; out=root/'quality_gate'; out.mkdir(exist_ok=True)
reference=(root/'recipe2_reference.py').read_text()
reference=reference.replace("ROOT=Path(__file__).resolve().parent; W=40", "ROOT=Path(__file__).resolve().parent/'quality_gate'; W=40")
reference=reference.replace("Path('/opt/slam-study/legacy/data')/f'{name}_slam_tum.txt'", "Path('/opt/slam-study/experiments/direction1_gate_20260917/matched')/name/f'{name}_slam_tum.txt'")
reference=reference.replace("for p in paths: manifest.append", "paths.extend([Path('/opt/slam-study/experiments/direction1_gate_20260917/matched')/name/'frame_signals.csv',Path('/opt/slam-study/experiments/direction1_gate_20260917/matched')/name/'online_state.csv'])\n for p in paths: manifest.append")
reference=reference.replace(' scenarios={}\n', ''' quality=np.genfromtxt(paths[4],delimiter=',',names=True,usecols=range(17))
 qi=np.argsort(quality['ts']); quality=quality[qi]; qt=quality['ts']
 nk=np.maximum(quality['n_kps'],1)
 qvec=np.c_[quality['state'],np.log1p(nk),quality['n_map_pts']/nk,
  quality['n_inliers']/nk,quality['n_lost_kps']/nk,quality['reproj_mean_px'],
  quality['reproj_std_px'],quality['reproj_median_px'],quality['n_reproj']/nk,
  quality['obs_per_mp'],np.log1p(quality['n_local_mps']),np.log1p(quality['n_local_kfs']),quality['is_kf']]
 qvec=np.nan_to_num(qvec)
 states=np.genfromtxt(paths[5],delimiter=',',names=True)
 scenarios={}
''')
reference=reference.replace("   rows.append({'x':seq", '''   # Entire input window must belong to uninterrupted tracking state OK.
   sl=np.searchsorted(states['timestamp'],t[kept[k-W]],side='left')
   sh=np.searchsorted(states['timestamp'],t[e],side='right')
   if sh<=sl or np.any(states['state'][sl:sh]!=2): continue
   if np.max(np.diff(t[kept[k-W]:e+1]))>.11: continue
   target_times=t[kept[k-W+1:k+1]]
   right=np.clip(np.searchsorted(qt,target_times),0,len(qt)-1); left=np.maximum(right-1,0)
   matched=np.where(abs(qt[left]-target_times)<abs(qt[right]-target_times),left,right)
   # Existing library logs timestamps to 12 significant digits (~10ms resolution).
   if np.max(abs(qt[matched]-target_times))>.011: continue
   seq=np.c_[seq,qvec[matched]]
   rows.append({'x':seq''')
reference=reference.replace('Offline saved monocular trajectories; GT initial20% scale calibration; no live SLAM integration.',
 'Fresh per-frame real-time monocular camera poses; matched tracking logs; GT initial20% scale calibration; compensation not integrated.')
(out/'recipe2_reference.py').write_text(reference)
code=(root/'run_gate.py').read_text()
code=code.replace('root=Path(__file__).resolve().parent','root=Path(__file__).resolve().parent/\'quality_gate\'')
code=code.replace('CfC(25,48','CfC(gx.shape[2],48').replace('nn.GRU(25,48','nn.GRU(gx.shape[2],48')
code=code.replace('else 25)+gs.shape[1]','else gx.shape[2])+gs.shape[1]')
code=code.replace('40 observations motion/IMU consistency plus observed-only summary; no mismatched VI quality logs',
 '40 observations, motion/IMU consistency and 13 matched frame-quality signals; uninterrupted tracking windows only')
code=code.replace('Offline saved mono trajectories, initial20% GT scale calibration.',
 'Fresh real-time camera poses, matched quality signals, initial20% GT scale calibration; compensation offline.')
code=code.replace('No full reprojection/feature geometry signals collected in this probe.',
 'Matched reprojection and inlier statistics included; no spatial feature-distribution signal; timestamp nearest-match tolerance11ms.')
code=code.replace('probability_cutoff','gate_cutoff')
(out/'generated_run_gate.py').write_text(code)
compile(reference,str(out/'recipe2_reference.py'),'exec')
compile(code,str(out/'generated_run_gate.py'),'exec')
import sys
if '--prepare-only' in sys.argv:
 print('QUALITY_SCRIPTS_PREPARED'); sys.exit(0)
exec(compile(code,str(out/'generated_run_gate.py'),'exec'),globals())
