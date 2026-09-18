"""Bounded exploratory risk forecasting; no SLAM integration or GT scale fit."""
from pathlib import Path
import json, hashlib, time, copy
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from ncps.torch import CfC

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
W=40; H=20; STRIDE=5
manifest={}
def average_precision_score(y,p):
 order=np.argsort(-p,kind='stable'); sy=y[order]; scores=p[order]
 ends=np.r_[np.flatnonzero(np.diff(scores)!=0),len(y)-1]
 tp=np.cumsum(sy)[ends]; precision=tp/(ends+1)
 return float(np.sum(np.diff(np.r_[0,tp])*precision)/y.sum())
def roc_auc_score(y,p):
 n=y.sum(); m=len(y)-n
 return float((rankdata(p)[y==1].sum()-n*(n+1)/2)/(n*m))
def read(path, **kw):
 manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 return np.genfromtxt(path,**kw)

def load(name):
 folder=ROOT/name
 a=read(folder/'online.csv',delimiter=',',skip_header=1)
 gt=read(Path('/opt/slam-study/datasets')/name/'mav0/state_groundtruth_estimate0/data.csv',delimiter=',',skip_header=1)
 imu=read(Path('/opt/slam-study/datasets')/name/'mav0/imu0/data.csv',delimiter=',',skip_header=1)
 gt[:,0]/=1e9; imu[:,0]/=1e9
 assert np.all(np.diff(a[:,0])>0) and np.all(np.diff(gt[:,0])>0)
 t=a[:,0]; validgt=(t>=gt[0,0])&(t<=gt[-1,0]); clipped=np.clip(t,gt[0,0],gt[-1,0])
 gr=Slerp(gt[:,0],Rotation.from_quat(gt[:,[5,6,7,4]]))(clipped).as_matrix()
 gp=np.stack([np.interp(clipped,gt[:,0],gt[:,j]) for j in [1,2,3]],axis=1)
 # EuRoC T_body_camera, independently checked against sensor.yaml.
 rb=np.array([[.0148655429818,-.999880929698,.00414029679422],[.999557249008,.0149672133247,.025715529948],[-.0257744366974,.00375618835797,.999660727178]])
 pb=np.array([-.0216401454975,-.064676986768,.00981073058949])
 gp=gp+np.einsum('nij,j->ni',gr,pb)
 cfg=Path('/opt/slam-study/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC.yaml')
 manifest[str(cfg)]=hashlib.sha256(cfg.read_bytes()).hexdigest()
 rectpath=ROOT/'rectification.json'
 manifest[str(rectpath)]=hashlib.sha256(rectpath.read_bytes()).hexdigest()
 gr=gr@rb@np.asarray(json.loads(rectpath.read_text())).T
 sr=Rotation.from_quat(a[:,5:9]).as_matrix(); sp=a[:,2:5]
 dt=np.r_[.05,np.diff(t)]
 delta=np.zeros((len(a),3)); delta[1:]=np.einsum('nij,nj->ni',sr[:-1].transpose(0,2,1),np.diff(sp,axis=0))/dt[1:,None]
 angular=np.zeros_like(delta); angular[1:]=Rotation.from_matrix(sr[:-1].transpose(0,2,1)@sr[1:]).as_rotvec()/dt[1:,None]
 observed=np.full((len(a),6),np.nan)
 for j in range(6):
  # Mean of IMU measurements available up to this image, no future sample.
  for i,ti in enumerate(t):
   lo=np.searchsorted(imu[:,0],ti-dt[i],side='right'); hi=np.searchsorted(imu[:,0],ti,side='right')
   observed[i,j]=np.mean(imu[lo:hi,j+1]) if hi>lo else np.nan
 rbc=rb@np.asarray(json.loads(rectpath.read_text())).T
 gyro_camera=observed[:,:3]@rbc
 residual=angular-gyro_camera
 # Difference of interval velocities, expressed in the current interval start frame.
 acceleration=np.zeros_like(delta)
 for i in range(2,len(a)):
  previous_velocity=sr[i-1].T@sr[i-2]@delta[i-1]
  acceleration[i]=(delta[i]-previous_velocity)/dt[i]
 features=np.c_[delta,angular,dt,observed,np.linalg.norm(observed[:,:3],axis=1),
  np.linalg.norm(observed[:,3:],axis=1),residual,np.linalg.norm(residual,axis=1),
  acceleration,np.linalg.norm(acceleration,axis=1),np.linalg.norm(delta,axis=1),np.linalg.norm(angular,axis=1)]
 assert features.shape[1]==25
 rows=[]; risks=[]; lost=[]; endtimes=[]; errors=[]; timespans=[]
 for i in range(W-1,len(a)-H,STRIDE):
  l=i-W+1; j=i+H
  if t[l]-t[0]<15 or not np.all(a[l:i+1,1]==2): continue
  if not np.all(np.isfinite(features[l:i+1])) or not np.all(validgt[[i,j]]): continue
  if np.max(np.diff(t[l:j+1]))>.075: continue
  fail=np.any(a[i+1:j+1,1]!=2)
  ds=sr[i].T@(sp[j]-sp[i]); dg=gr[i].T@(gp[j]-gp[i])
  error=np.linalg.norm(ds-dg) if not fail else np.nan
  rows.append(features[l:i+1]); timespans.append(dt[l:i+1]); lost.append(fail); errors.append(error); endtimes.append(t[i])
 x=np.asarray(rows,dtype=np.float32)
 assert len(x)>50, (name,len(x))
 return dict(x=x,dt=np.asarray(timespans,dtype=np.float32),lost=np.asarray(lost),error=np.asarray(errors),t=np.asarray(endtimes))

def summary(x): return np.concatenate([x[:,-1],x.mean(1),x.std(1)],axis=1)
def metrics(y,p,cut):
 pred=p>=cut; pos=y==1; neg=~pos
 return {'auroc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
         'recall':float(pred[pos].mean()),'false_positive_rate':float(pred[neg].mean()),'cutoff':float(cut)}
def cutoff(y,p): return float(np.quantile(p[y==0],.9,method='higher')+1e-9)
def event_count(y): return int(np.sum(np.diff(np.r_[0,y])==1))

data={n:load(n) for n in ['V101','V201','V202']}
tr,va,te=[data[n] for n in ['V101','V201','V202']]
threshold=float(max(.02,np.quantile(tr['error'][np.isfinite(tr['error'])],.9)))
audit={}
for name,d in data.items():
 d['y']=(d['lost'] | (d['error']>threshold)).astype(np.int64)
 audit[name]={'windows':len(d['y']),'positive':int(d['y'].sum()),'positive_fraction':float(d['y'].mean()),
  'risk_episodes':event_count(d['y']),'future_tracking_loss_windows':int(d['lost'].sum()),
  'local_translation_rmse_m':float(np.sqrt(np.nanmean(d['error']**2)))}
protocol={'train':'V101','validation':'V201','test':'V202','history_seconds':2,'forecast_seconds':1,
 'stride_seconds':.25,'risk_threshold_m':threshold,'threshold_selection':'training 90th percentile, floor 2cm',
 'target':'future 1-second camera-frame relative translation error above threshold OR future non-OK tracking',
 'model_input':'25 past/current motion and IMU features, including gyro consistency; no GT input; no reprojection/inlier quality',
 'scale_alignment':'none; stereo metric scale; camera/IMU extrinsic conversion for GT labels',
 'fit_budget':5,'neural_seeds':[0,1],'epochs_max':30,'hidden_units':32,
 'limitations':['Previously inspected sequences, exploratory only','One replay per sequence','Overlapping windows, not independent samples',
 'Tracking-OK histories only; startup first15seconds excluded','No SLAM intervention; no drift reduction claim',
 'Existing library logs quality only on monocular path; VI quality file is empty; motion/IMU-only pilot does not test full quality-based proposal']}
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
(ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2))
(ROOT/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items()}
(ROOT/'input_integrity.json').write_text(json.dumps(unchanged,indent=2)); assert all(unchanged.values())
np.savez_compressed(ROOT/'label_audit.npz',**{f'{name}_{key}':d[key] for name,d in data.items() for key in ['y','error','lost','t']})
print('DATA_AUDIT',json.dumps(audit),flush=True)
if any(min(d['y'].sum(),len(d['y'])-d['y'].sum())<20 for d in data.values()):
 (ROOT/'result.json').write_text(json.dumps({'status':'insufficient_positive_or_negative_windows','audit':audit},indent=2))
 print('STOP_INSUFFICIENT_CLASSES',flush=True); raise SystemExit(0)
mean=tr['x'].mean((0,1)); std=np.maximum(tr['x'].std((0,1)),1e-3)
for d in data.values(): d['z']=np.clip((d['x']-mean)/std,-10,10).astype(np.float32)
records=[]; predictions={}; yv=va['y']; yt=te['y']
def record(kind,seed,pv,pt,extra=None):
 c=cutoff(yv,pv)
 row={'model':kind,'seed':seed,'validation':metrics(yv,pv,c),'test':metrics(yt,pt,c)}
 if extra: row.update(extra)
 records.append(row); predictions[f'{kind}_{seed}_val']=pv; predictions[f'{kind}_{seed}_test']=pt
 print('RESULT',json.dumps(row),flush=True)

# Two simple, causal rules; choose one by validation AP only.
rules=[('high_gyro_inconsistency',va['x'][:,-1,18],te['x'][:,-1,18]),
       ('high_visual_acceleration',va['x'][:,-1,22],te['x'][:,-1,22])]
best=max(rules,key=lambda r:average_precision_score(yv,r[1])); record('Rule',-1,best[1],best[2],{'selected_rule':best[0]})
sx=np.c_[summary(tr['z']),np.ones(len(tr['y']))]; yy=tr['y']
weights=np.where(yy==1,len(yy)/(2*yy.sum()),len(yy)/(2*(len(yy)-yy.sum())))
def objective(beta):
 logits=sx@beta; penalty=.01
 value=np.mean(weights*(np.logaddexp(0,logits)-yy*logits))+.5*penalty*np.sum(beta[:-1]**2)
 grad=sx.T@(weights*(expit(logits)-yy))/len(yy); grad[:-1]+=penalty*beta[:-1]
 return value,grad
fit=minimize(objective,np.zeros(sx.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500})
assert fit.success,fit.message
def linear_probability(d): return expit(np.c_[summary(d['z']),np.ones(len(d['y']))]@fit.x)
record('Logistic',0,linear_probability(va),linear_probability(te))

class Net(nn.Module):
 def __init__(self,kind):
  super().__init__(); self.kind=kind
  if kind=='CfC':
   self.cell=CfC(25,32,batch_first=True,backbone_units=32,backbone_layers=1)
   original=self.cell.rnn_cell.forward
   def adapter(x,h,ts):
    if torch.is_tensor(ts) and ts.ndim==1: ts=ts[:,None]
    return original(x,h,ts)
   self.cell.rnn_cell.forward=adapter
  else: self.cell=nn.GRU(25,32,batch_first=True)
  self.head=nn.Linear(32,1)
 def forward(self,x,dt):
  z,_=self.cell(x,timespans=dt) if self.kind=='CfC' else self.cell(x)
  return self.head(z[:,-1]).squeeze(-1)

device='cuda' if torch.cuda.is_available() else 'cpu'
tx=torch.tensor(tr['z'],device=device); td=torch.tensor(tr['dt'],device=device); ty=torch.tensor(tr['y'],dtype=torch.float32,device=device)
vx=torch.tensor(va['z'],device=device); vd=torch.tensor(va['dt'],device=device)
ex=torch.tensor(te['z'],device=device); ed=torch.tensor(te['dt'],device=device)
for kind in ['GRU','CfC']:
 for seed in [0,1]:
  started=time.time(); torch.manual_seed(seed); np.random.seed(seed)
  model=Net(kind).to(device); opt=torch.optim.Adam(model.parameters(),lr=.001)
  loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((ty==0).sum()/(ty==1).sum()),device=device))
  bestap=-1; stale=0; beststate=None; epochbest=0
  for epoch in range(30):
   model.train(); order=torch.randperm(len(tx),device=device)
   for ix in order.split(128):
    opt.zero_grad(); err=loss(model(tx[ix],td[ix]),ty[ix]); err.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
   model.eval()
   with torch.no_grad(): pv=model(vx,vd).sigmoid().cpu().numpy()
   ap=average_precision_score(yv,pv)
   if ap>bestap+1e-5: bestap=ap; beststate=copy.deepcopy(model.state_dict()); stale=0; epochbest=epoch+1
   else: stale+=1
   if stale>=6: break
  model.load_state_dict(beststate); model.eval()
  with torch.no_grad(): pv=model(vx,vd).sigmoid().cpu().numpy(); pt=model(ex,ed).sigmoid().cpu().numpy()
  torch.save({'state':beststate,'mean':mean,'std':std},ROOT/f'{kind}_{seed}.pt')
  record(kind,seed,pv,pt,{'best_epoch':epochbest,'seconds':time.time()-started,'parameters':sum(p.numel() for p in model.parameters())})

np.savez_compressed(ROOT/'predictions.npz',y_validation=yv,y_test=yt,t_test=te['t'],**predictions)
(ROOT/'result.json').write_text(json.dumps({'status':'complete','device':device,'protocol':protocol,'audit':audit,'records':records},indent=2))
unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items()}
(ROOT/'input_integrity.json').write_text(json.dumps(unchanged,indent=2)); assert all(unchanged.values())
print('COMPLETE',flush=True)
