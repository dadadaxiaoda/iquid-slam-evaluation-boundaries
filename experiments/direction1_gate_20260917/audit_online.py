from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parent; rows=[]
for name in ['V101','V102','V201','V103','V202']:
 folder=root/'matched'/name
 if not (folder/'online_state.csv').exists() or not (folder/f'{name}_slam_tum.txt').exists(): continue
 state=np.genfromtxt(folder/'online_state.csv',delimiter=',',names=True)
 poses=np.atleast_2d(np.loadtxt(folder/f'{name}_slam_tum.txt'))
 q=np.genfromtxt(folder/'frame_signals.csv',delimiter=',',names=True,usecols=range(17)); qt=np.sort(q['ts'])
 t=poses[:,0]; right=np.clip(np.searchsorted(qt,t),0,len(qt)-1); left=np.maximum(right-1,0)
 nearest=np.where(abs(qt[left]-t)<abs(qt[right]-t),left,right); err=abs(qt[nearest]-t)
 row={'sequence':name,'total_frames':len(state),'online_pose_rows':len(poses),
  'tracking_OK_frames':int(np.sum(state['state']==2)),'tracking_OK_pct':float(100*np.mean(state['state']==2)),
  'quality_rows':len(q),'max_timestamp_match_error_s':float(np.max(err)),
  'unmatched_poses':int(np.sum(err>.011)),'pose_gap_count_over110ms':int(np.sum(np.diff(t)>.11))}
 assert np.isfinite(poses).all() and row['unmatched_poses']==0
 rows.append(row)
(root/'online_data_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
