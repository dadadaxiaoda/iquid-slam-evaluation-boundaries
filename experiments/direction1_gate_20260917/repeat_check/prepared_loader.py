"""Recipe2 offline probe: causal IMU physical prior and liquid residuals. No MD output."""
from pathlib import Path
import sys,json,time,hashlib,copy,csv
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation as R,Slerp
import yaml
from ncps.torch import CfC

ROOT=Path(__file__).resolve().parent/'repeat_check'; W=40; START=time.time()
torch.set_num_threads(2); torch.set_num_interop_threads(2)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
data={}; manifest=[]; audit={}; records=[]
def align_scale(a,b):
 aa=a-a.mean(0); bb=b-b.mean(0); u,z,v=np.linalg.svd(bb.T@aa/len(a))
 d=np.diag([1,1,np.sign(np.linalg.det(u@v))]); return np.trace(d@np.diag(z))/np.mean(np.sum(aa*aa,axis=1))

def preintegrate(imu,ti,t0,t1,Rcb,gravity):
 # Zero-order hold over [t0,t1], including the latest measurement before t0.
 lo=max(0,np.searchsorted(ti,t0,side='right')-1); hi=np.searchsorted(ti,t1,side='left')
 rows=imu[lo:hi,1:7]; edges=np.r_[t0,ti[lo+1:hi],t1]
 rot=np.eye(3); velocity=np.zeros(3); position=np.zeros(3)
 for row,dt in zip(rows,np.diff(edges)):
  omega=Rcb@row[:3]; accel=Rcb@row[3:]
  half=rot@R.from_rotvec(omega*dt*.5).as_matrix()
  acc=half@accel-gravity
  position+=velocity*dt+.5*acc*dt*dt; velocity+=acc*dt
  rot=rot@R.from_rotvec(omega*dt).as_matrix()
 return position,velocity,R.from_matrix(rot).as_rotvec()

for name in ['V103']:
 base=Path('/opt/slam-study/datasets')/name/'mav0'
 paths=[Path('/opt/slam-study/experiments/direction1_gate_20260917/repeat_check')/name/f'{name}_slam_tum.txt',base/'state_groundtruth_estimate0/data.csv',base/'cam0/sensor.yaml',base/'imu0/data.csv']
 paths.extend([Path('/opt/slam-study/experiments/direction1_gate_20260917/repeat_check')/name/'frame_signals.csv',Path('/opt/slam-study/experiments/direction1_gate_20260917/repeat_check')/name/'online_state.csv'])
 for p in paths: manifest.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 a=np.loadtxt(paths[0]); a=a[np.isfinite(a).all(1)&(np.linalg.norm(a[:,4:8],axis=1)>.5)]
 keep=np.unique(a[:,0],return_index=True)[1]; keep.sort(); a=a[keep]
 gt=np.loadtxt(paths[1],delimiter=',',comments='#'); tg=gt[:,0]/1e9
 a=a[(a[:,0]>=tg[0])&(a[:,0]<=tg[-1])]; t=a[:,0]; ps=a[:,1:4]; rs=R.from_quat(a[:,4:8]); n=len(t); cal=int(.2*n)
 Tbc=np.array(yaml.safe_load(paths[2].read_text())['T_BS']['data']).reshape(4,4)
 rb=Slerp(tg,R.from_quat(gt[:,[5,6,7,4]]))(t); rgt=rb*R.from_matrix(Tbc[:3,:3])
 pg=np.column_stack([np.interp(t,tg,gt[:,k]) for k in [1,2,3]])+rb.apply(Tbc[:3,3])
 scale=align_scale(ps[:cal],pg[:cal]); imu=np.loadtxt(paths[3],delimiter=',',comments='#'); ti=imu[:,0]/1e9
 Rcb=Tbc[:3,:3].T
 ci=np.searchsorted(ti,t[:cal]); ci=np.clip(ci,0,len(ti)-1)
 # Calibration estimates gravity from measured specific force, no GT orientation/velocity.
 gw=np.mean(rs[:cal].apply(imu[ci,4:7]@Rcb.T),axis=0)
 gw=gw/max(np.linalg.norm(gw),1e-8)*9.81
 # Compute physical interval prior once per nominal target; no learned sensor calibration.
 targets={}
 for end in range(cal+W+1,n):
  j=int(np.searchsorted(t,t[end]-.5,side='left')); prev=int(np.searchsorted(t,t[j]-.25,side='left'))
  if prev<=cal or j==end or prev==j: continue
  local=scale*rs[j].inv().apply(ps[end]-ps[j]); truth=rgt[j].inv().apply(pg[end]-pg[j])
  v0=scale*rs[j].inv().apply(ps[j]-ps[prev])/(t[j]-t[prev])
  ip,iv,ir=preintegrate(imu,ti,t[j],t[end],Rcb,rs[j].inv().apply(gw))
  physical=v0*(t[end]-t[j])+ip
  corr=physical-local
  # Scale-equivariant output normalization based only on observed quantities.
  mag=max(.05,float(np.linalg.norm(local)),float(np.linalg.norm(physical)))
  targets[end]={'y':truth-local,'phy':corr,'mag':mag,'horizon':t[end]-t[j],
    'summary':np.r_[local,physical,corr,iv,ir,v0,t[end]-t[j]],'j':j}
 quality=np.genfromtxt(paths[4],delimiter=',',names=True,usecols=range(17))
 qi=np.argsort(quality['ts']); quality=quality[qi]; qt=quality['ts']
 nk=np.maximum(quality['n_kps'],1)
 qvec=np.c_[quality['state'],np.log1p(nk),quality['n_map_pts']/nk,
  quality['n_inliers']/nk,quality['n_lost_kps']/nk,quality['reproj_mean_px'],
  quality['reproj_std_px'],quality['reproj_median_px'],quality['n_reproj']/nk,
  quality['obs_per_mp'],np.log1p(quality['n_local_mps']),np.log1p(quality['n_local_kfs']),quality['is_kf']]
 qvec=np.nan_to_num(qvec)
 states=np.genfromtxt(paths[5],delimiter=',',names=True)
 scenarios={}
 for cond,drop in [('nominal',0),('drop50',.5)]:
  rng=np.random.default_rng(825); kept=np.flatnonzero(rng.random(n)>=drop); tt=t[kept]
  features=np.zeros((len(kept),25))
  for k in range(1,len(kept)):
   p,e=kept[k-1],kept[k]; dt=t[e]-t[p]
   visual=scale*rs[p].inv().apply(ps[e]-ps[p]); vr=(rs[p].inv()*rs[e]).as_rotvec()
   ip,iv,ir=preintegrate(imu,ti,t[p],t[e],Rcb,rs[p].inv().apply(gw))
   lo=np.searchsorted(ti,t[p],side='right'); hi=np.searchsorted(ti,t[e],side='right'); vv=imu[lo:hi,1:7]
   integ=(vv*np.diff(np.r_[t[p],ti[lo:hi]])[:,None]).sum(0)+vv[-1]*(t[e]-ti[hi-1])
   pv=int(kept[max(0,k-6)])
   v0=scale*rs[p].inv().apply(ps[p]-ps[pv])/max(t[p]-t[pv],1e-8)
   features[k]=np.r_[dt,visual,vr,integ,vv.mean(0),visual-(v0*dt+ip),vr-ir]
  rows=[]
  for k in range(W+1,len(kept)):
   e=int(kept[k])
   if e not in targets or kept[k-W]<=cal: continue
   target=targets[e].copy()
   # With removed observations, use only targets whose start pose remains available.
   if drop and target['j'] not in kept[max(0,k-W):k+1]: continue
   if drop:
    j=target['j']; prev_k=max(0,np.searchsorted(tt,t[j]-.25,side='right')-1); prev=int(kept[prev_k])
    if prev==j: continue
    local=scale*rs[j].inv().apply(ps[e]-ps[j])
    v0=scale*rs[j].inv().apply(ps[j]-ps[prev])/(t[j]-t[prev])
    ip,iv,ir=preintegrate(imu,ti,t[j],t[e],Rcb,rs[j].inv().apply(gw))
    physical=v0*(t[e]-t[j])+ip; corr=physical-local
    target['phy']=corr; target['mag']=max(.05,float(np.linalg.norm(local)),float(np.linalg.norm(physical)))
    target['summary']=np.r_[local,physical,corr,iv,ir,v0,t[e]-t[j]]
   seq=features[k-W+1:k+1].copy(); mag=target['mag']
   seq[:,1:4]/=mag; seq[:,19:22]/=mag
   summary=target['summary'].copy(); summary[:12]/=mag; summary[15:18]/=mag
   # Entire input window must belong to uninterrupted tracking state OK.
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
   rows.append({'x':seq,'s':summary,'y':target['y'],'phy':target['phy'],'mag':mag,
    't':t[e],'start':t[kept[k-W]],'idx':e,'horizon':target['horizon']})
  scenarios[cond]={key:np.array([r[key] for r in rows],dtype=np.float32 if key in ['x','s','y','phy','mag'] else None) for key in rows[0]}
 data[name]=scenarios
 audit[name]={'n':n,'scale':float(scale),'gravity_norm':float(np.linalg.norm(gw)),
  'rows':{c:len(ds['y']) for c,ds in scenarios.items()},'rms_m':float(np.sqrt(np.mean(np.sum(scenarios['nominal']['y']**2,axis=1))))}
 print('DATA',name,json.dumps(audit[name]),flush=True)
(ROOT/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
(ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2))

