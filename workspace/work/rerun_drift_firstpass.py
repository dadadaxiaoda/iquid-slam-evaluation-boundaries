"""Read-only legacy sources; corrected residual-prediction experiments.
Run with the existing WSL venv. Outputs are confined to --out.
Saved trajectories have retrospective optimization: this is NOT live SLAM validation.
"""
from pathlib import Path
import argparse, json, time, hashlib, csv, copy
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation as R, Slerp
from ncps.torch import CfC
from ncps.wirings import AutoNCP

P=Path('/opt/slam-study/legacy/data'); W=20
ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True)
ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--seeds',default='0,1,2')
ap.add_argument('--probe',action='store_true'); args=ap.parse_args()
OUT=Path(args.out); OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2); torch.set_num_interop_threads(2)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
SEEDS=[int(x) for x in args.seeds.split(',')]; records=[]; audit={}; start=time.time()

def align(a,b):
 ac=a-a.mean(0); bc=b-b.mean(0); u,z,v=np.linalg.svd(bc.T@ac/len(a))
 d=np.diag([1,1,np.sign(np.linalg.det(u@v))]); rg=u@d@v
 s=np.trace(d@np.diag(z))/np.mean(np.sum(ac*ac,axis=1))
 return s,rg,b.mean(0)-s*rg@a.mean(0)

def load(name):
 if name.startswith('K'):
  with np.load(P/f'{name}_scale_label.npz') as d:
   t=d['t'].copy(); ps=d['p_slam'].copy(); pg=d['p_gt'].copy(); q=d['q_slam'].copy()
  extra=None
  f=P/f'{name}_dataset_instr.npz'
  if f.exists():
   with np.load(f) as d: extra=d['X_full'][:,[7,8,9,10,11,12,13,14,17]].copy()
  raw_n=len(t); keep=np.unique(t,return_index=True)[1]; keep.sort()
  t,ps,pg,q=t[keep],ps[keep],pg[keep],q[keep]
  if extra is not None: extra=extra[keep]
 else:
  import yaml
  a=np.loadtxt(P/f'{name}_slam_tum.txt'); raw_n=len(a)
  finite=np.isfinite(a).all(1)&(np.linalg.norm(a[:,4:8],axis=1)>0.5); a=a[finite]
  keep=np.unique(a[:,0],return_index=True)[1]; keep.sort(); a=a[keep]
  base=Path('/opt/slam-study/datasets')/name/'mav0'
  g=np.loadtxt(base/'state_groundtruth_estimate0/data.csv',delimiter=',',comments='#')
  tg=g[:,0]/1e9; valid=(a[:,0]>=tg[0])&(a[:,0]<=tg[-1]); a=a[valid]
  t=a[:,0]; ps=a[:,1:4]; q=a[:,4:8]
  rb=Slerp(tg,R.from_quat(g[:,[5,6,7,4]]))(t)
  # Ground truth is IMU/body. SLAM monocular export is camera.
  T=np.array(yaml.safe_load((base/'cam0/sensor.yaml').read_text())['T_BS']['data']).reshape(4,4)
  pg=np.column_stack([np.interp(t,tg,g[:,k]) for k in [1,2,3]])+rb.apply(T[:3,3])
  imu=np.loadtxt(base/'imu0/data.csv',delimiter=',',comments='#'); ti=imu[:,0]/1e9
  extra=np.zeros((len(t),12))
  for i in range(1,len(t)):
   # Only samples at/before current frame; no future endpoint interpolation.
   lo=np.searchsorted(ti,t[i-1],side='right'); hi=np.searchsorted(ti,t[i],side='right')
   if hi<=lo: raise ValueError('missing IMU interval')
   vv=imu[lo:hi,1:7]; tt=ti[lo:hi]; dt=np.diff(np.r_[t[i-1],tt])
   integ=(vv*dt[:,None]).sum(0)+vv[-1]*(t[i]-tt[-1])
   extra[i]=np.r_[integ,vv.mean(0)]
  extra[0]=extra[1]
 assert np.all(np.diff(t)>0) and len(t)>2*W
 assert np.isfinite(ps).all() and np.isfinite(pg).all()
 q=q/np.linalg.norm(q,axis=1,keepdims=True)
 audit[name]={'raw_rows':raw_n,'clean_rows':len(t),'removed_rows':raw_n-len(t),
  'duration_s':float(t[-1]-t[0]),'median_dt_s':float(np.median(np.diff(t))),
  'extra_dimensions':0 if extra is None else extra.shape[1]}
 return dict(name=name,t=t,ps=ps,pg=pg,q=q,extra=extra)

def dataset(raw,calibration):
 s,rg,tg=align(raw['ps'][calibration],raw['pg'][calibration])
 pa=s*raw['ps']@rg.T+tg; rr=R.from_matrix(rg)*R.from_quat(raw['q'])
 y=rr.inv().apply(raw['pg']-pa)
 # Translation and relative rotation both expressed in previous camera frame.
 original=R.from_quat(raw['q']); n=len(pa); motion=np.zeros((n,7))
 motion[1:,0]=np.diff(raw['t'])
 motion[1:,1:4]=s*original[:-1].inv().apply(np.diff(raw['ps'],axis=0))
 motion[1:,4:7]=(original[:-1].inv()*original[1:]).as_rotvec()
 motion[0]=motion[1]
 extra=raw['extra']; full=motion
 if extra is not None:
  full=np.c_[motion,extra,np.vstack([np.zeros((1,extra.shape[1])),np.diff(extra,axis=0)])] if raw['name'].startswith('K') else np.c_[motion,extra]
 x=np.stack([full[i-W:i] for i in range(W,n)]).astype('float32')
 ts=x[:,:,0].copy(); assert np.all(ts>0)
 ds=dict(x=x,ts=ts,y=y[W:].astype('float32'),pa=pa[W:],pg=raw['pg'][W:],
         rot=rr.as_matrix()[W:],idx=np.arange(W,n),time=raw['t'][W:],scale=s)
 # Coordinate round-trip: adding the exact body residual restores GT.
 np.testing.assert_allclose(ds['pa']+np.einsum('nij,nj->ni',ds['rot'],ds['y']),ds['pg'],atol=2e-4)
 return ds

class Net(nn.Module):
 def __init__(self,kind,dim):
  super().__init__(); self.kind=kind
  if kind.startswith('CfC'):
   self.cell=CfC(dim,AutoNCP(64,3),batch_first=True)
   # ncps 1.0.1 squeezes batched timespans to (B,), while wired cells
   # multiply tensors of (B,H). Local adapter; installed package unchanged.
   original_forward=self.cell.rnn_cell.forward
   def batched_time_forward(inputs,hx,ts):
    if torch.is_tensor(ts) and ts.ndim==1: ts=ts[:,None]
    return original_forward(inputs,hx,ts)
   self.cell.rnn_cell.forward=batched_time_forward
  elif kind=='GRU': self.cell=nn.GRU(dim,64,batch_first=True); self.head=nn.Linear(64,3)
  else: self.cell=nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3))
 def forward(self,x,ts):
  if self.kind.startswith('CfC'):
   z,_=self.cell(x,timespans=ts if self.kind=='CfC_time' else None); return z[:,-1]
  if self.kind=='GRU': z,_=self.cell(x); return self.head(z[:,-1])
  return self.cell(x[:,-1])

def score(pred,ds,ids):
 y=ds['y'][ids].astype(float); pred=pred.astype(float)
 corrected=ds['pa'][ids]+np.einsum('nij,nj->ni',ds['rot'][ids],pred)
 base=np.linalg.norm(ds['pg'][ids]-ds['pa'][ids],axis=1)
 after=np.linalg.norm(ds['pg'][ids]-corrected,axis=1)
 b=np.sqrt(np.mean(base**2)); a=np.sqrt(np.mean(after**2))
 sse=np.sum((pred-y)**2); sst=np.sum((y-y.mean(0))**2)
 # Fixed alignment, same baseline and corrected positions, no test GT refitting.
 return dict(r2=float(1-sse/max(sst,1e-15)),ate_before_m=float(b),ate_after_m=float(a),
             improvement_pct=float(100*(1-a/b)),n_test=len(ids),
             rmse_residual_m=float(np.sqrt(np.mean(np.sum((pred-y)**2,axis=1)))))

def save():
 (OUT/'results.json').write_text(json.dumps(dict(records=records,audit=audit,
   device=DEV,elapsed_s=time.time()-start,epochs=args.epochs,seeds=SEEDS,window=W,
   caveat='Retrospectively optimized saved SLAM trajectories; initial GT calibration for causal tests; not live deployment.'),indent=2))

def run_case(label,train_ds,tr,va,test_sets,configs):
 assert not np.intersect1d(tr,va).size
 assert len(tr)>20 and len(va)>10
 for cfg in configs:
  cols=np.arange(7) if cfg=='motion' else np.arange(train_ds['x'].shape[2])
  xx=train_ds['x'][:,:,cols]; yy=train_ds['y']
  xm=xx[tr].mean((0,1)); xs=xx[tr].std((0,1))+1e-6
  ym=yy[tr].mean(0); ys=yy[tr].std(0)+1e-6
  xn=(xx-xm)/xs; yn=(yy-ym)/ys
  # Time scale learned from training dt only. Constant-dt CfC uses the same architecture.
  time_unit=float(np.median(train_ds['ts'][tr])); tn=train_ds['ts']/time_unit
  def record(kind,seed,test_name,ds,te,pred,detail=None):
   rec=dict(case=label,config=cfg,model=kind,seed=seed,test_sequence=test_name,
    n_train=len(tr),n_val=len(va),time_unit_s=time_unit,**score(pred,ds,te),**(detail or {}))
   records.append(rec)
   print(json.dumps(rec),flush=True)
   key=f'{label}_{cfg}_{kind}_{seed}_{test_name}'
   np.savez_compressed(OUT/(key+'.npz'),prediction=pred,target=ds['y'][te],
    t=ds['time'][te],baseline=ds['pa'][te],gt=ds['pg'][te],rotation=ds['rot'][te])
   save()
  for test_name,ds,te in test_sets:
   record('Zero',-1,test_name,ds,te,np.zeros((len(te),3)))
   record('Mean',-1,test_name,ds,te,np.broadcast_to(ym,(len(te),3)))
   # Offline truth-history diagnostic only. Never counted as deployable success.
   prev=ds['y'][te-1]; world=np.einsum('nij,nj->ni',ds['rot'][te-1],prev)
   oracle=np.einsum('nji,nj->ni',ds['rot'][te],world)
   record('Oracle_previous_GT',-1,test_name,ds,te,oracle)
  z=xn[tr].reshape(len(tr),-1); zv=xn[va].reshape(len(va),-1)
  # Dual ridge avoids inversion of a 20*D matrix repeatedly.
  z=np.c_[z,np.ones(len(z))]; zv=np.c_[zv,np.ones(len(zv))]
  gram=z@z.T; e,u=np.linalg.eigh(gram.astype('float64')); uy=u.T@yn[tr]
  best=None
  for alpha in [0.1,1,10,100,1000]:
   coef=z.T@(u@(uy/(e[:,None]+alpha)))
   err=np.mean((zv@coef-yn[va])**2)
   if best is None or err<best[0]: best=(err,alpha,coef)
  for test_name,ds,te in test_sets:
   xt=(ds['x'][te][:,:,cols]-xm)/xs; zz=np.c_[xt.reshape(len(te),-1),np.ones(len(te))]
   record('Ridge',-1,test_name,ds,te,(zz@best[2])*ys+ym,dict(alpha=best[1]))
  for kind in ['MLP','GRU','CfC_fixed','CfC_time']:
   for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    m=Net(kind,len(cols)).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=5e-4)
    X=torch.tensor(xn[tr],device=DEV); Y=torch.tensor(yn[tr],device=DEV)
    T=torch.tensor(tn[tr],device=DEV); XV=torch.tensor(xn[va],device=DEV)
    TV=torch.tensor(tn[va],device=DEV); YV=torch.tensor(yn[va],device=DEV)
    best_loss=float('inf'); state=None; wait=0; best_ep=0; tm=time.time()
    for ep in range(args.epochs):
     m.train(); perm=torch.randperm(len(tr),device=DEV)
     for lo in range(0,len(tr),256):
      ii=perm[lo:lo+256]; loss=nn.functional.mse_loss(m(X[ii],T[ii]),Y[ii])
      opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),5); opt.step()
     m.eval()
     with torch.no_grad(): vl=float(nn.functional.mse_loss(m(XV,TV),YV))
     if not np.isfinite(vl): raise RuntimeError('nonfinite validation')
     if vl<best_loss-1e-6: best_loss=vl; state=copy.deepcopy(m.state_dict()); wait=0; best_ep=ep+1
     else: wait+=1
     if wait>=15: break
    m.load_state_dict(state); m.eval()
    for test_name,ds,te in test_sets:
     xt=(ds['x'][te][:,:,cols]-xm)/xs
     with torch.no_grad(): pred=m(torch.tensor(xt,device=DEV),torch.tensor(ds['ts'][te]/time_unit,device=DEV)).cpu().numpy()*ys+ym
     record(kind,seed,test_name,ds,te,pred,dict(best_epoch=best_ep,epochs_run=ep+1,
       train_seconds=time.time()-tm,parameters=sum(p.numel() for p in m.parameters())))
    torch.save(dict(state=state,xm=xm,xs=xs,ym=ym,ys=ys,time_unit=time_unit,cols=cols),
               OUT/f'{label}_{cfg}_{kind}_{seed}.pt')
    del m,X,Y,T,XV,TV,YV; torch.cuda.empty_cache() if DEV=='cuda' else None

raws={n:load(n) for n in ['K00i','K04i','K03','V101','V102','V103','V201','V202']}
# Meaningful check of time wiring and batched broadcasting.
torch.manual_seed(0); check=Net('CfC_time',7).to(DEV)
inp=torch.randn(4,W,7,device=DEV); tt=torch.ones(4,W,device=DEV)
check.eval()
with torch.no_grad():
 out1=check(inp,tt); out2=check(inp,tt*2)
 assert out1.shape==(4,3) and not torch.allclose(out1,out2)
del check
save(); print('AUDIT',json.dumps(audit),flush=True)
if args.probe:
 raw=raws['K00i']; ds=dataset(raw,np.arange(len(raw['t'])))
 n=len(ds['y']); tr=np.arange(int(.6*n)); va=np.arange(int(.6*n)+W,int(.75*n))
 te=np.arange(int(.75*n)+W,n)
 run_case('probe',ds,tr,va,[('K00i',ds,te)],['rich']); raise SystemExit

for name in ['K00i','V103']:
 raw=raws[name]; nraw=len(raw['t']); n=nraw-W
 # Offline diagnostic: comparable all-sequence alignment, independent early-stop set.
 ds=dataset(raw,np.arange(nraw)); tr=np.arange(int(.6*n))
 va=np.arange(int(.6*n)+W,int(.75*n)); te=np.arange(int(.75*n)+W,n)
 run_case(name+'_global_diagnostic',ds,tr,va,[(name,ds,te)],['motion','rich'])
 # Three nonoverlapping future test blocks; one initial calibration is frozen.
 ds=dataset(raw,np.arange(int(.2*nraw)))
 for fold,(a,b) in enumerate([(.5,.65),(.65,.8),(.8,1.0)]):
  end=int(a*n); valstart=end-int(.1*n)
  tr=np.arange(valstart-W); va=np.arange(valstart,end-W)
  te=np.arange(end,n if b==1 else int(b*n))
  assert tr[-1]+W<va[0] and va[-1]+W<te[0]
  run_case(name+f'_future{fold+1}',ds,tr,va,[(name,ds,te)],['motion','rich'])

# KITTI transfer, motion-only (K03 has no tracking features).
ds=dataset(raws['K00i'],np.arange(int(.2*len(raws['K00i']['t'])))); n=len(ds['y'])
tests=[]
for name in ['K03','K04i']:
 raw=raws[name]; cal=int(.2*len(raw['t'])); dd=dataset(raw,np.arange(cal))
 tests.append((name,dd,np.flatnonzero(dd['idx']>=cal+W)))
run_case('KITTI_transfer',ds,np.arange(int(.7*n)-W),np.arange(int(.7*n),n),tests,['motion'])

# EuRoC leave-sequence-out: V101/V102/V201 train, V202 validation, V103 test.
parts=[dataset(raws[k],np.arange(int(.2*len(raws[k]['t'])))) for k in ['V101','V102','V201']]
val=dataset(raws['V202'],np.arange(int(.2*len(raws['V202']['t']))))
# Drop calibration prefixes for each sequence to test meaningful residuals.
sel=[np.flatnonzero(p['idx']>=int(.2*len(raws[k]['t']))+W) for p,k in zip(parts,['V101','V102','V201'])]
vi=np.flatnonzero(val['idx']>=int(.2*len(raws['V202']['t']))+W)
combo={k:np.concatenate([p[k][i] for p,i in zip(parts,sel)]+[val[k][vi]]) for k in ['x','ts','y','pa','pg','rot','idx','time']}
nt=sum(len(i) for i in sel); nv=len(vi)
test=dataset(raws['V103'],np.arange(int(.2*len(raws['V103']['t']))))
te=np.flatnonzero(test['idx']>=int(.2*len(raws['V103']['t']))+W)
run_case('EuRoC_transfer',combo,np.arange(nt),np.arange(nt,nt+nv),[('V103',test,te)],['motion','rich'])
save(); print('COMPLETE',time.time()-start,flush=True)
