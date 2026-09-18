from pathlib import Path
import json,numpy as np,yaml
from scipy.spatial.transform import Rotation as R,Slerp
root=Path(__file__).resolve().parent; name='V103'; base=Path('/opt/slam-study/datasets')/name/'mav0'
g=np.loadtxt(base/'state_groundtruth_estimate0/data.csv',delimiter=',',comments='#'); tg=g[:,0]/1e9
T=np.array(yaml.safe_load((base/'cam0/sensor.yaml').read_text())['T_BS']['data']).reshape(4,4)
def scale_align(ps,pg):
 aa=ps-ps.mean(0); bb=pg-pg.mean(0); u,z,v=np.linalg.svd(bb.T@aa/len(aa)); d=np.diag([1,1,np.sign(np.linalg.det(u@v))])
 return np.trace(d@np.diag(z))/np.mean(np.sum(aa*aa,axis=1))
paths={'new_online':root/'matched/V103/V103_slam_tum.txt','new_export':root/'matched/V103/f_V103.txt',
 'legacy_export':Path('/opt/slam-study/legacy/data/V103_slam_tum.txt')}
loaded={}; allstats=[]; sensitivity=[]
for kind,path in paths.items():
 a=np.atleast_2d(np.loadtxt(path)); raw_n=len(a)
 if np.median(a[:,0])>1e12: a[:,0]/=1e9
 a=a[np.isfinite(a).all(1)&(np.linalg.norm(a[:,4:8],axis=1)>.5)]
 keep=np.unique(a[:,0],return_index=True)[1]; keep.sort(); a=a[keep]; a=a[(a[:,0]>=tg[0])&(a[:,0]<=tg[-1])]
 t=a[:,0]; rs=R.from_quat(a[:,4:8]); rb=Slerp(tg,R.from_quat(g[:,[5,6,7,4]]))(t)
 rg=rb*R.from_matrix(T[:3,:3]); pg=np.column_stack([np.interp(t,tg,g[:,k]) for k in [1,2,3]])+rb.apply(T[:3,3])
 cal=int(.2*len(t)); scale=scale_align(a[:cal,1:4],pg[:cal]); norm=[]; keyed={}
 for end in range(cal+41,len(t)):
  start=np.searchsorted(t,t[end]-.5,side='left')
  if start<=cal or start==end or np.max(np.diff(t[end-40:end+1]))>.11: continue
  slam=scale*rs[start].inv().apply(a[end,1:4]-a[start,1:4]); truth=rg[start].inv().apply(pg[end]-pg[start])
  err=np.linalg.norm(slam-truth); norm.append(err); keyed[round(float(t[end]),4)]=err
 loaded[kind]=keyed
 allstats.append({'kind':kind,'raw_rows':raw_n,'clean_rows':len(t),'scale':float(scale),'n_valid_windows':len(norm),
  'local_rms_m':float(np.sqrt(np.mean(np.array(norm)**2))),'quantiles_m':np.quantile(norm,[.5,.9,.99,1]).tolist()})
 for fraction in [.1,.2,.3,.5,1.0]:
  sc=scale_align(a[:int(fraction*len(t)),1:4],pg[:int(fraction*len(t))]); errors=[]
  for end in range(cal+41,len(t)):
   start=np.searchsorted(t,t[end]-.5,side='left')
   if start<=cal or start==end or np.max(np.diff(t[end-40:end+1]))>.11: continue
   slam=sc*rs[start].inv().apply(a[end,1:4]-a[start,1:4]); truth=rg[start].inv().apply(pg[end]-pg[start])
   errors.append(np.linalg.norm(slam-truth))
  sensitivity.append({'kind':kind,'calibration_fraction':fraction,'scale':float(sc),'same_window_local_rms_m':float(np.sqrt(np.mean(np.array(errors)**2)))})
common=set.intersection(*(set(v) for v in loaded.values()))
commonstats=[{'kind':kind,'n_common_endpoints':len(common),'rms_m':float(np.sqrt(np.mean([vals[t]**2 for t in common])))} for kind,vals in loaded.items()]
(root/'export_vs_online_diagnostic.json').write_text(json.dumps({'all_valid_windows':allstats,'common_endpoints':commonstats,'scale_sensitivity':sensitivity,
 'caveat':'Per-source initial20% GT-derived scale; common endpoints may still have different start frames and calibration prefixes. This identifies run/export sensitivity, not a causal code-fix attribution.'},indent=2))
print(json.dumps({'all_valid_windows':allstats,'common_endpoints':commonstats},indent=2))
print('SCALE_SENSITIVITY',json.dumps(sensitivity))
