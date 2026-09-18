"""Recipe2 offline probe: causal IMU physical prior and liquid residuals. No MD output."""
from pathlib import Path
import sys,json,time,hashlib,copy,csv
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation as R,Slerp
import yaml
from ncps.torch import CfC

ROOT=Path(__file__).resolve().parent; W=40; START=time.time()
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

for name in ['V101','V102','V201','V103','V202']:
 base=Path('/opt/slam-study/datasets')/name/'mav0'
 paths=[Path('/opt/slam-study/legacy/data')/f'{name}_slam_tum.txt',base/'state_groundtruth_estimate0/data.csv',base/'cam0/sensor.yaml',base/'imu0/data.csv']
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
   rows.append({'x':seq,'s':summary,'y':target['y'],'phy':target['phy'],'mag':mag,
    't':t[e],'start':t[kept[k-W]],'idx':e,'horizon':target['horizon']})
  scenarios[cond]={key:np.array([r[key] for r in rows],dtype=np.float32 if key in ['x','s','y','phy','mag'] else None) for key in rows[0]}
 data[name]=scenarios
 audit[name]={'n':n,'scale':float(scale),'gravity_norm':float(np.linalg.norm(gw)),
  'rows':{c:len(ds['y']) for c,ds in scenarios.items()},'rms_m':float(np.sqrt(np.mean(np.sum(scenarios['nominal']['y']**2,axis=1))))}
 print('DATA',name,json.dumps(audit[name]),flush=True)
(ROOT/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
(ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2))

tr=[]; va=[]; tests={}
def subset(ds,mask): return {k:v[mask] for k,v in ds.items()}
for name,scenarios in data.items():
 for cond,ds in scenarios.items():
  if name in ['V101','V102']: tr.append(ds)
  elif name=='V201': va.append(ds)
  elif name=='V202': tests[f'V202_independent_{cond}']=ds
  else:
   # Same physical boundary for both sampling variants, derived from nominal timestamps.
   bounds=np.quantile(data['V103']['nominal']['t'],[.5,.7]); lo,hi=bounds
   tr.append(subset(ds,ds['t']<=lo)); va.append(subset(ds,(ds['start']>lo)&(ds['t']<=hi)))
   tests[f'V103_future_{cond}']=subset(ds,ds['start']>hi)
def concat(parts): return {k:np.concatenate([d[k] for d in parts]) for k in parts[0]}
train=concat(tr); val=concat(va)
np.savez_compressed(ROOT/'split_arrays.npz',train_t=train['t'],train_start=train['start'],val_t=val['t'],val_start=val['start'],v103_bounds=bounds)
xm=train['x'].mean((0,1)); xs=train['x'].std((0,1))+1e-6
sm=train['s'].mean(0); ss=train['s'].std(0)+1e-6
def inputs(ds): return (ds['x']-xm)/xs,(ds['s']-sm)/ss
X,S=inputs(train); VX,VS=inputs(val)
unit=float(np.median(train['x'][:,:,0]))
def save():
 (ROOT/'results.json').write_text(json.dumps({'records':records,'elapsed_s':time.time()-START,
  'train':'V101,V102 plus V103 first50% after calibration','validation':'V201 plus purged V103 next20%',
  'test':'V202 independent; purged V103 final30%','n_train':len(X),'n_val':len(VX),
  'device':DEV,'window':W,'seeds':[0,1,2],'caveats':[
  'Offline saved monocular trajectories; GT initial20% scale calibration; no live SLAM integration.',
  'Physical prior uses past visual velocity, IMU zero bias, gravity from initial measured specific force; not a full VIO estimator.',
  'V103 future is same-sequence test, not independent severe-environment generalization.',
  'Artificial input removal does not rerun underlying SLAM.',
  'No online reprojection/feature geometry logs available for these monocular trajectories.',
  'CfC and GRU use different parameter counts; exploratory fixed training budget.']},indent=2))
def record(kind,seed,predict,detail=None):
 for name,ds in tests.items():
  pred=predict(ds).astype(float); y=ds['y'].astype(float)
  b=float(np.sqrt(np.mean(np.sum(y*y,axis=1)))); a=float(np.sqrt(np.mean(np.sum((pred-y)**2,axis=1))))
  rec={'model':kind,'seed':seed,'test':name,'n':len(y),'before_m':b,'after_m':a,
   'gain_pct':100*(1-a/b),'r2':float(1-np.sum((pred-y)**2)/np.sum((y-y.mean(0))**2)),**(detail or {})}
  records.append(rec); print(json.dumps(rec),flush=True)
  np.savez_compressed(ROOT/f'{kind}_{seed}_{name}.npz',prediction=pred,target=y,physical=ds['phy'],mag=ds['mag'],t=ds['t'],start=ds['start'],idx=ds['idx'])
 save()
record('Zero',-1,lambda ds:np.zeros_like(ds['y']))
record('Physical',-1,lambda ds:ds['phy'])
# Validation-only global shrinkage; avoids assuming raw physical correction must be trusted.
gates=[0,.1,.25,.5,.75,1]
alpha=min(gates,key=lambda g:np.mean(np.sum((val['y']-g*val['phy'])**2,axis=1)))
record('Physical_shrunk',-1,lambda ds:alpha*ds['phy'],{'gate':alpha})

def ridge_features(ds):
 xx,ssn=inputs(ds); return np.c_[xx.reshape(len(xx),-1),ssn,np.ones(len(xx))]
z=ridge_features(train); zv=ridge_features(val)
e,u=np.linalg.eigh((z.T@z).astype(float)); target=(train['y']-alpha*train['phy'])/train['mag'][:,None]
uy=u.T@(z.T@target); best=None
for reg in [1,10,100,1000,10000]:
 coef=u@(uy/(e[:,None]+reg)); vp=alpha*val['phy']+(zv@coef)*val['mag'][:,None]
 loss=np.mean(np.sum((vp-val['y'])**2,axis=1))
 if best is None or loss<best[0]: best=(loss,reg,coef)
record('Physical_Ridge_residual',-1,lambda ds:alpha*ds['phy']+(ridge_features(ds)@best[2])*ds['mag'][:,None],{'reg':best[1],'physical_gate':alpha})

class Network(nn.Module):
 def __init__(self,kind):
  super().__init__(); self.kind=kind
  if kind.startswith('CfC'):
   self.cell=CfC(25,64,batch_first=True,backbone_units=64,backbone_layers=1)
   forward=self.cell.rnn_cell.forward
   def adapter(inputs,hx,ts):
    if torch.is_tensor(ts) and ts.ndim==1: ts=ts[:,None]
    return forward(inputs,hx,ts)
   self.cell.rnn_cell.forward=adapter
  else: self.cell=nn.GRU(25,64,batch_first=True)
  self.head=nn.Sequential(nn.Linear(64+19,64),nn.ReLU(),nn.Linear(64,3))
 def forward(self,x,s,ts):
  if self.kind.startswith('CfC'): z,_=self.cell(x,timespans=None if self.kind=='CfC_fixed_residual' else ts)
  else: z,_=self.cell(x)
  return self.head(torch.cat([z[:,-1],s],dim=-1))

for kind in ['CfC_direct','CfC_time_residual','CfC_fixed_residual','GRU_residual']:
 residual=kind!='CfC_direct'; offset=alpha if residual else 0
 yy=(train['y']-offset*train['phy'])/train['mag'][:,None]
 ym=yy.mean(0); ys=yy.std(0)+1e-6; YY=(yy-ym)/ys
 for seed in [0,1,2]:
  torch.manual_seed(seed); model=Network(kind).to(DEV); opt=torch.optim.Adam(model.parameters(),lr=5e-4)
  xt=torch.tensor(X,device=DEV); st=torch.tensor(S,device=DEV); yt=torch.tensor(YY,device=DEV)
  tt=torch.tensor(train['x'][:,:,0]/unit,device=DEV)
  vx=torch.tensor(VX,device=DEV); vs=torch.tensor(VS,device=DEV); vt=torch.tensor(val['x'][:,:,0]/unit,device=DEV)
  state=None; bestloss=float('inf'); wait=0; begun=time.time()
  for ep in range(80):
   model.train(); perm=torch.randperm(len(X),device=DEV)
   for lo in range(0,len(X),256):
    ii=perm[lo:lo+256]; loss=nn.functional.mse_loss(model(xt[ii],st[ii],tt[ii]),yt[ii])
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
   model.eval()
   with torch.no_grad():
    vp=model(vx,vs,vt).cpu().numpy()*ys+ym
   vp=offset*val['phy']+vp*val['mag'][:,None]
   # Validation averages per-sequence MSE rather than treating easy windows as independent evidence.
   losses=[]
   for part in va:
    px,psn=inputs(part)
    with torch.no_grad(): pp=model(torch.tensor(px,device=DEV),torch.tensor(psn,device=DEV),torch.tensor(part['x'][:,:,0]/unit,device=DEV)).cpu().numpy()*ys+ym
    pp=offset*part['phy']+pp*part['mag'][:,None]; losses.append(np.mean(np.sum((pp-part['y'])**2,axis=1)))
   vl=float(np.mean(losses))
   if not np.isfinite(vl): raise RuntimeError('nonfinite loss')
   if vl<bestloss-1e-6: bestloss=vl; state=copy.deepcopy(model.state_dict()); wait=0; bestep=ep+1
   else: wait+=1
   if (ep+1)%10==0: print('PROGRESS',kind,seed,ep+1,vl,flush=True)
   if wait>=12: break
  model.load_state_dict(state); model.eval()
  def predict(ds):
   px,psn=inputs(ds)
   with torch.no_grad(): pp=model(torch.tensor(px,device=DEV),torch.tensor(psn,device=DEV),torch.tensor(ds['x'][:,:,0]/unit,device=DEV)).cpu().numpy()*ys+ym
   return offset*ds['phy']+pp*ds['mag'][:,None]
  record(kind,seed,predict,{'best_epoch':bestep,'epochs_run':ep+1,'seconds':time.time()-begun,
   'parameters':sum(p.numel() for p in model.parameters()),'physical_gate':offset})
  # Apply output shrinkage selected solely on validation, separately reported from raw output.
  gatescore=[]
  for gate in gates:
   losses=[np.mean(np.sum((offset*part['phy']+gate*(predict(part)-offset*part['phy'])-part['y'])**2,axis=1)) for part in va]
   gatescore.append(float(np.mean(losses)))
  gate=gates[int(np.argmin(gatescore))]
  record(kind+'_gated',seed,lambda ds:offset*ds['phy']+gate*(predict(ds)-offset*ds['phy']),{'output_gate':gate,'physical_gate':offset})
  torch.save({'state':state,'xm':xm,'xs':xs,'sm':sm,'ss':ss,'ym':ym,'ys':ys,'unit':unit,'physical_gate':offset,'output_gate':gate},ROOT/f'{kind}_{seed}.pt')
  del model,xt,st,yt,tt,vx,vs,vt; torch.cuda.empty_cache()
integrity=[{**m,'unchanged':hashlib.sha256(Path(m['path']).read_bytes()).hexdigest()==m['sha256']} for m in manifest]
assert all(m['unchanged'] for m in integrity)
(ROOT/'integrity_check.json').write_text(json.dumps(integrity,indent=2)); save()
print('COMPLETE',time.time()-START,flush=True)
