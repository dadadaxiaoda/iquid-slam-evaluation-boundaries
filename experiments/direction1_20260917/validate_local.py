"""Offline local translation error experiment. No live SLAM or raw sensor outage claim."""
from pathlib import Path
import sys,json,time,copy,hashlib
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation as R,Slerp
import yaml

OUT=Path(__file__).resolve().parent
# Reuse audited loader/network definitions only; do not run the legacy experiments.
sys.argv=['reference','--out',str(OUT/'reference_init')]
source=(OUT/'legacy_runner_reference.py').read_text()
exec(compile(source.split("raws={n:load(n)")[0],str(OUT/'legacy_runner_reference.py'),'exec'),globals())
OUT=Path(__file__).resolve().parent
START=time.time(); results=[]; datasets={}; provenance=[]
for name in ['V101','V102','V201','V103','V202']:
 raw=load(name); base=Path('/opt/slam-study/datasets')/name/'mav0'
 paths=[P/f'{name}_slam_tum.txt',base/'state_groundtruth_estimate0/data.csv',base/'cam0/sensor.yaml',base/'imu0/data.csv']
 for path in paths:
  provenance.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 gt=np.loadtxt(paths[1],delimiter=',',comments='#'); tg=gt[:,0]/1e9
 Tbc=np.array(yaml.safe_load(paths[2].read_text())['T_BS']['data']).reshape(4,4)
 rgt=Slerp(tg,R.from_quat(gt[:,[5,6,7,4]]))(raw['t'])*R.from_matrix(Tbc[:3,:3])
 rs=R.from_quat(raw['q']); n=len(raw['t']); cal=int(.2*n)
 scale,_,_=align(raw['ps'][:cal],raw['pg'][:cal])
 imu=np.loadtxt(paths[3],delimiter=',',comments='#'); ti=imu[:,0]/1e9
 scenarios={}
 for condition,drop in [('nominal',0),('drop25',.25),('drop50',.5)]:
  rng=np.random.default_rng(100+int(drop*100)); kept=np.flatnonzero(rng.random(n)>=drop)
  tt=raw['t'][kept]; feats=np.zeros((len(kept),19))
  feats[1:,0]=np.diff(tt)
  feats[1:,1:4]=scale*rs[kept[:-1]].inv().apply(np.diff(raw['ps'][kept],axis=0))
  feats[1:,4:7]=(rs[kept[:-1]].inv()*rs[kept[1:]]).as_rotvec()
  for k in range(1,len(kept)):
   lo=np.searchsorted(ti,tt[k-1],side='right'); hi=np.searchsorted(ti,tt[k],side='right')
   vv=imu[lo:hi,1:7]; it=ti[lo:hi]; dt=np.diff(np.r_[tt[k-1],it])
   integ=(vv*dt[:,None]).sum(0)+vv[-1]*(tt[k]-it[-1])
   feats[k,7:]=np.r_[integ,vv.mean(0)]
  xx=[]; yy=[]; baseline=[]; timepoints=[]; starts=[]; horizons=[]; indices=[]
  for k in range(21,len(kept)):
   end=kept[k]
   start_k=int(np.searchsorted(tt,raw['t'][end]-.5,side='left'))
   j=int(kept[start_k])
   if kept[k-20]<=cal or j<=cal: continue
   # Both relative translations use their own start-camera axes (same physical convention).
   slam=scale*rs[j].inv().apply(raw['ps'][end]-raw['ps'][j])
   truth=rgt[j].inv().apply(raw['pg'][end]-raw['pg'][j])
   xx.append(feats[k-19:k+1]); yy.append(truth-slam); baseline.append(slam)
   timepoints.append(raw['t'][end]); starts.append(raw['t'][kept[k-20]])
   horizons.append(raw['t'][end]-raw['t'][j]); indices.append(end)
  scenarios[condition]={'x':np.array(xx,dtype=np.float32),'y':np.array(yy,dtype=np.float32),
    'slam':np.array(baseline),'t':np.array(timepoints),'start':np.array(starts),
    'horizon':np.array(horizons),'idx':np.array(indices)}
 datasets[name]=scenarios
 audit[name].update(scale_initial20=scale,calibration_frames=cal,rows={k:len(v['y']) for k,v in scenarios.items()})

(OUT/'input_manifest.json').write_text(json.dumps(provenance,indent=2))
(OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))
def concatenate(names):
 return {k:np.concatenate([datasets[n][c][k] for n in names for c in ['nominal','drop25','drop50']]) for k in ['x','y']}
train=concatenate(['V101','V102']); val=concatenate(['V201'])
xm=train['x'].mean((0,1)); xs=train['x'].std((0,1))+1e-6
ym=train['y'].mean(0); ys=train['y'].std(0)+1e-6
unit=float(np.median(train['x'][:,:,0])); X=(train['x']-xm)/xs; XV=(val['x']-xm)/xs
Y=(train['y']-ym)/ys; YV=(val['y']-ym)/ys
def save_results():
 (OUT/'results.json').write_text(json.dumps({'records':results,'elapsed_seconds':time.time()-START,
  'train':['V101','V102'],'validation':['V201'],'test':['V103','V202'],
  'device':DEV,'n_train':len(Y),'n_validation':len(YV),'seeds':[0,1,2],
  'caveats':['Saved retrospective monocular SLAM trajectories, not live validation.',
   'Initial 20% GT-derived scale calibration per sequence; remaining frames only used for training/evaluation.',
   'Artificial observation removal does not rerun SLAM and cannot represent actual sensor outage.',
   'Targets are retrospective ~0.5s local translation corrections, not future prediction; no integration claim.',
   'Overlapping windows and three sampling versions are not independent statistical replications.']},indent=2))
def record(kind,seed,predict,details=None):
 for name in ['V103','V202']:
  for condition,ds in datasets[name].items():
   pred=predict(ds); target=ds['y'].astype(float)
   b=float(np.sqrt(np.mean(np.sum(target**2,axis=1))))
   a=float(np.sqrt(np.mean(np.sum((target-pred)**2,axis=1))))
   r2=float(1-np.sum((target-pred)**2)/np.sum((target-target.mean(0))**2))
   rec={'model':kind,'seed':seed,'sequence':name,'condition':condition,'n':len(target),
    'before_m':b,'after_m':a,'gain_pct':100*(1-a/b),'r2':r2,**(details or {})}
   results.append(rec); print(json.dumps(rec),flush=True)
   np.savez_compressed(OUT/f'{kind}_{seed}_{name}_{condition}.npz',prediction=pred,target=target,
     timestamp=ds['t'],start=ds['start'],horizon=ds['horizon'],idx=ds['idx'],slam=ds['slam'])
 save_results()
record('Zero',-1,lambda ds:np.zeros_like(ds['y']))
record('Mean',-1,lambda ds:np.broadcast_to(ym,ds['y'].shape))
z=np.c_[X.reshape(len(X),-1),np.ones(len(X))]; zv=np.c_[XV.reshape(len(XV),-1),np.ones(len(XV))]
e,u=np.linalg.eigh((z.T@z).astype(float)); uy=u.T@(z.T@Y); best=None
for alpha in [.1,1,10,100,1000]:
 coef=u@(uy/(e[:,None]+alpha)); loss=np.mean((zv@coef-YV)**2)
 if best is None or loss<best[0]: best=(loss,alpha,coef)
record('Ridge',-1,lambda ds:np.c_[((ds['x']-xm)/xs).reshape(len(ds['x']),-1),np.ones(len(ds['x']))]@best[2]*ys+ym,{'alpha':best[1]})
for kind in ['MLP','GRU','CfC_fixed_head','CfC_time_head']:
 for seed in [0,1,2]:
  torch.manual_seed(seed); np.random.seed(seed); model=Net(kind,19).to(DEV)
  opt=torch.optim.Adam(model.parameters(),lr=5e-4)
  xt=torch.tensor(X,device=DEV); yt=torch.tensor(Y,device=DEV)
  ts=torch.tensor(train['x'][:,:,0]/unit,device=DEV)
  vx=torch.tensor(XV,device=DEV); vy=torch.tensor(YV,device=DEV)
  vt=torch.tensor(val['x'][:,:,0]/unit,device=DEV)
  state=None; bestloss=float('inf'); wait=0; epochbest=0; started=time.time()
  for ep in range(60):
   model.train(); perm=torch.randperm(len(xt),device=DEV)
   for lo in range(0,len(xt),256):
    ii=perm[lo:lo+256]; loss=nn.functional.mse_loss(model(xt[ii],ts[ii]),yt[ii])
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
   model.eval()
   with torch.no_grad(): vl=float(nn.functional.mse_loss(model(vx,vt),vy))
   if not np.isfinite(vl): raise RuntimeError('nonfinite loss')
   if vl<bestloss-1e-6: bestloss=vl; state=copy.deepcopy(model.state_dict()); wait=0; epochbest=ep+1
   else: wait+=1
   if (ep+1)%10==0: print(f'PROGRESS {kind} seed={seed} epoch={ep+1} val={vl:.6f}',flush=True)
   if wait>=10: break
  model.load_state_dict(state); model.eval()
  def predict(ds):
   with torch.no_grad():
    return model(torch.tensor((ds['x']-xm)/xs,device=DEV),torch.tensor(ds['x'][:,:,0]/unit,device=DEV)).cpu().numpy()*ys+ym
  record(kind,seed,predict,{'best_epoch':epochbest,'epochs':ep+1,'seconds':time.time()-started,
    'parameters':sum(p.numel() for p in model.parameters())})
  torch.save({'state':state,'xm':xm,'xs':xs,'ym':ym,'ys':ys,'time_unit':unit},OUT/f'{kind}_{seed}.pt')
  del model,xt,yt,ts,vx,vy,vt; torch.cuda.empty_cache()
integrity=[]
for item in provenance:
 integrity.append({**item,'unchanged':hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']})
(OUT/'integrity_check.json').write_text(json.dumps(integrity,indent=2))
assert all(x['unchanged'] for x in integrity)
save_results(); print('COMPLETE',time.time()-START,flush=True)
