"""Read-only, single-replay exploratory quality-versus-error analysis. No fits."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent; folder=ROOT/'V202'
paths=[folder/'quality.csv',folder/'online.csv',Path('/opt/slam-study/datasets/V202/mav0/state_groundtruth_estimate0/data.csv'),
 Path('/opt/slam-study/experiments/reliability_pilot_20260918/rectification.json'),Path('/opt/slam-study/experiments/reliability_pilot_20260918/protocol.json'),folder/'f_V202.txt']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
q=np.genfromtxt(paths[0],delimiter=',',names=True)
a=np.genfromtxt(paths[1],delimiter=',',names=True)
gt=np.genfromtxt(paths[2],delimiter=',',skip_header=1);gt[:,0]/=1e9
threshold=json.loads(paths[4].read_text())['risk_threshold_m']
t=a['ts']; n=len(t); valid=q['pose_valid']==1
assert len(q)==n and np.array_equal(a['input_id'],q['input_id']) and np.max(abs(q['ts']-t))<1e-9
clipped=np.clip(t,gt[0,0],gt[-1,0]); gv=(t>=gt[0,0])&(t<=gt[-1,0])
gr=Slerp(gt[:,0],Rotation.from_quat(gt[:,[5,6,7,4]]))(clipped).as_matrix()
gp=np.stack([np.interp(clipped,gt[:,0],gt[:,j]) for j in [1,2,3]],axis=1)
rb=np.array([[.0148655429818,-.999880929698,.00414029679422],[.999557249008,.0149672133247,.025715529948],[-.0257744366974,.00375618835797,.999660727178]])
pb=np.array([-.0216401454975,-.064676986768,.00981073058949])
gp+=np.einsum('nij,j->ni',gr,pb);gr=gr@rb@np.array(json.loads(paths[3].read_text())).T
sr=Rotation.from_quat(np.c_[a['qx'],a['qy'],a['qz'],a['qw']]).as_matrix()
sp=np.c_[a['px'],a['py'],a['pz']]
saved=np.loadtxt(paths[5]);saved[:,0]/=1e9
saved_body_rotation=Rotation.from_quat(saved[:,4:8]).as_matrix()
saved_camera_rotation=saved_body_rotation@rb@np.array(json.loads(paths[3].read_text())).T
saved_camera_position=saved[:,1:4]+np.einsum('nij,j->ni',saved_body_rotation,pb)
right=np.clip(np.searchsorted(saved[:,0],t),0,len(saved)-1);left=np.maximum(right-1,0)
matches=np.where(abs(saved[left,0]-t)<abs(saved[right,0]-t),left,right)
matched=abs(saved[matches,0]-t)<.001
def saved_rpe(i,j):
 if not matched[i] or not matched[j] or not gv[i] or not gv[j]:return np.nan
 u,v=matches[i],matches[j]
 ds=saved_camera_rotation[u].T@(saved_camera_position[v]-saved_camera_position[u])
 dg=gr[i].T@(gp[j]-gp[i])
 return float(np.linalg.norm(ds-dg))
def rpe(i,j):
 return float(np.linalg.norm(sr[i].T@(sp[j]-sp[i])-gr[i].T@(gp[j]-gp[i]))) if valid[i] and valid[j] and gv[i] and gv[j] else np.nan
def crossed(i,j,fields): return any(np.any(np.diff(q[k][i:j+1])!=0) for k in fields)
context_fields=['map_id','reset_epoch','map_create_epoch','big_change_index']
future=np.full(n,np.nan);past=np.full(n,np.nan);step=np.full(n,np.nan)
for i in range(n):
 if i+20<n and np.all(q['state'][i:i+21]==2):future[i]=rpe(i,i+20)
 if i>=20 and np.all(q['state'][i-20:i+1]==2):past[i]=rpe(i-20,i)
 if i>0:step[i]=rpe(i-1,i)
indices=[]
for i in range(39,n-20,5):
 if t[i-39]-t[0]<15 or not np.all(q['state'][i-39:i+1]==2) or not np.all(valid[i-39:i+1]):continue
 if not np.all(gv[[i,i+20]]) or np.max(np.diff(t[i-39:i+21]))>.075:continue
 if np.isfinite(future[i]):indices.append(i)
indices=np.array(indices);y=future[indices]>threshold
context=np.array([crossed(i,i+20,context_fields) for i in indices])
updates=np.array([crossed(i,i+20,['map_change_index']) for i in indices])
pastcontext=np.array([crossed(i-39,i,context_fields) for i in indices])
already=past[indices]>threshold
clean=~context & ~pastcontext & ~already
positive_indices=indices[y]
groups=np.split(positive_indices,np.flatnonzero(np.diff(positive_indices)>5)+1) if len(positive_indices) else []

def stats(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return {'count':len(x),'median_cm':float(np.median(x)*100),'p90_cm':float(np.quantile(x,.9)*100),
 'rmse_cm':float(np.sqrt(np.mean(x*x))*100),'max_cm':float(np.max(x)*100)} if len(x) else None
def auc_ap(labels,scores):
 finite=np.isfinite(scores);labels=np.asarray(labels,dtype=int)[finite];scores=scores[finite]
 pos=int(labels.sum());neg=len(labels)-pos
 if not pos or not neg:return {'available_windows':len(labels),'positive_windows':pos,'auroc':None,'average_precision':None}
 auc=(rankdata(scores)[labels==1].sum()-pos*(pos+1)/2)/(pos*neg)
 order=np.argsort(-scores,kind='stable');sy=labels[order];ss=scores[order]
 ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(labels)-1];tp=np.cumsum(sy)[ends]
 ap=np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/pos
 return {'available_windows':len(labels),'positive_windows':pos,'prevalence':pos/len(labels),'auroc':float(auc),'average_precision':float(ap)}

inlier=q['n_inlier_observed']/np.maximum(q['n_kps'],1)
# Signs make larger scores worse. Fixed past 0.5-second summaries, no tuning.
raw={'low_inlier_fraction':-inlier,'high_reprojection_p90':q['reproj_p90_px'],
 'low_grid_coverage':-q['inlier_grid_coverage'],'low_depth_fraction':-q['depth_valid_fraction'],
 'high_imu_rotation_residual':q['imu_rotation_residual_rad']}
scores={}
for name,v in raw.items():
 scores[name]=np.array([np.nanmedian(v[i-9:i+1]) if np.any(np.isfinite(v[i-9:i+1])) else np.nan for i in indices])
rules={name:{'all_windows':auc_ap(y,s),'clean_forecast_windows':auc_ap(y[clean],s[clean])} for name,s in scores.items()}
normal_reference=indices[(~y)&(~already)&(~context)&(~pastcontext)]
episodes=[]
for g in groups:
 start=int(g[0]);end=int(g[-1]);left=start+1;right=end+20
 anchor=int(left+np.nanargmax(step[left:right+1]))
 before=(t>=t[anchor]-1)&(t<t[anchor]);first=(t>=t[anchor]-1)&(t<t[anchor]-.5);last=(t>=t[anchor]-.5)&(t<t[anchor])
 quality={}
 for name,v in raw.items():
  ref=v[normal_reference];ref=ref[np.isfinite(ref)]
  values=v[before];values=values[np.isfinite(values)]
  valsfirst=v[first];valsfirst=valsfirst[np.isfinite(valsfirst)]
  valslast=v[last];valslast=valslast[np.isfinite(valslast)]
  median=float(np.median(values)) if len(values) else None
  quality[name]={'median_previous_1s':median,'badness_percentile_vs_same_replay_normal':float(np.mean(ref<=median)*100) if median is not None and len(ref) else None,
   'median_1_to_half_second_before':float(np.median(valsfirst)) if len(valsfirst) else None,
   'median_last_half_second_before':float(np.median(valslast)) if len(valslast) else None}
 episodes.append({'positive_windows':len(g),'first_forecast_offset_s':float(t[start]-t[0]),'last_forecast_offset_s':float(t[end]-t[0]),
  'largest_step_input_id':anchor,'largest_step_offset_s':float(t[anchor]-t[0]),'largest_step_error_cm':float(step[anchor]*100),
  'max_future_1s_error_cm':float(np.nanmax(future[g])*100),'same_step_final_export_error_cm':float(saved_rpe(anchor-1,anchor)*100),
  'max_same_window_final_export_error_cm':float(np.nanmax([saved_rpe(i,i+20) for i in g])*100),
  'context_crossing':crossed(start,end+20,context_fields),
  'map_update_crossing':crossed(start,end+20,['map_change_index']),'already_high_error_windows':int((past[g]>threshold).sum()),
  'quality_before_largest_step':quality})
report={'scope':'Single V202 replay, exploratory descriptive diagnostics; no training, tuning, or independent performance claim.',
 'threshold_m':threshold,'threshold_source':'Frozen earlier V101 training threshold; unchanged.',
 'history_seconds':2,'future_seconds':1,'window_stride_seconds':.25,'windows':len(indices),'positive_windows':int(y.sum()),'episodes':episodes,
 'all_eligible_error':stats(future[indices]),'without_future_big_context_change_error':stats(future[indices[~context]]),
 'same_window_final_export_error':stats([saved_rpe(i,i+20) for i in indices]),
 'windows_without_any_future_map_update':int((~updates).sum()),'error_without_any_future_map_update':stats(future[indices[~updates]]),
 'positive_future_context_crossing':int(np.sum(y&context)),'positive_future_map_update_crossing':int(np.sum(y&updates)),
 'positive_already_high_error':int(np.sum(y&already)),'clean_forecast_windows':int(clean.sum()),'clean_forecast_positive_windows':int(np.sum(y&clean)),
 'quality_rules':rules,'map_big_change_offsets_s':(t[np.flatnonzero(np.r_[False,np.diff(q['big_change_index'])!=0])]-t[0]).tolist(),
 'notes':['Future ground truth and map-change flags are used only for label diagnostics, not prediction inputs.',
 'Final exported trajectory uses later optimization; used only to diagnose jumps, not as online input or validated risk label.',
 'Clean forecast diagnostic excludes windows already in high error and those crossing big context changes; it is a retrospective subset.',
 'Quality percentiles use post-hoc same-replay normal reference and are not validated deployment thresholds.',
 'Map update counters do not establish causation or uniquely classify BA/loop corrections.',
 'Largest-step anchor is selected with GT after the run; before-anchor comparisons alone do not prove predictive lead.',
 'Overlapping windows and very few episodes cannot support reliable neural-model comparisons.']}
checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()};assert all(checks.values())
(ROOT/'quality_error_analysis.json').write_text(json.dumps(report,indent=2))
(ROOT/'quality_analysis_integrity.json').write_text(json.dumps(checks,indent=2))
np.savez_compressed(ROOT/'quality_error_arrays.npz',input_id=indices,future_error=future[indices],past_error=past[indices],positive=y,
 context=context,map_update=updates,clean=clean,**scores)
print(json.dumps(report,indent=2))
