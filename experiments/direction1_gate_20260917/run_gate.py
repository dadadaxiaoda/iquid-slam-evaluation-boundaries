"""Exploratory liquid per-window gate. No Markdown artifacts, no new external test claim."""
from pathlib import Path
root=Path(__file__).resolve().parent
src=(root/'recipe2_reference.py').read_text().split("record('Zero',")[0]
exec(compile(src,str(root/'recipe2_reference.py'),'exec'),globals())
ROOT=root; records=[]; begun_all=time.time()
fit_parts=[]; gate_parts=[]; val_parts=va; split_boundaries={}
for name in ['V101','V102','V103']:
 nominal=data[name]['nominal']; frac=.6 if name!='V103' else .3
 cut=float(np.quantile(nominal['t'],frac)); upper=float(np.quantile(nominal['t'],.5)) if name=='V103' else float(nominal['t'].max())
 split_boundaries[name]={'candidate_fit_end':cut,'gate_train_end':upper}
 for cond,ds in data[name].items():
  fit_parts.append(subset(ds,ds['t']<=cut))
  part=subset(ds,(ds['start']>cut)&(ds['t']<=upper))
  part['group']=np.full(len(part['y']),['V101','V102','V103'].index(name))
  gate_parts.append(part)
fit=concat(fit_parts); gate_train=concat(gate_parts)
for group,name in enumerate(['V101','V102','V103']):
 mask=gate_train['group']==group
 assert np.all(gate_train['start'][mask]>split_boundaries[name]['candidate_fit_end'])
 assert np.all(gate_train['t'][mask]<=split_boundaries[name]['gate_train_end'])
np.savez_compressed(ROOT/'gate_training_split.npz',candidate_fit_t=fit['t'],candidate_fit_start=fit['start'],
 gate_train_t=gate_train['t'],gate_train_start=gate_train['start'],gate_train_group=gate_train['group'])
xm=fit['x'].mean((0,1)); xs=fit['x'].std((0,1))+1e-6; sm=fit['s'].mean(0); ss=fit['s'].std(0)+1e-6
def inputs(ds): return (ds['x']-xm)/xs,(ds['s']-sm)/ss
def feature(ds):
 xx,sv=inputs(ds); return np.c_[xx.reshape(len(xx),-1),sv,np.ones(len(xx))]
def relative_validation(predict):
 return float(np.mean([np.mean(np.sum((predict(p)-p['y'])**2,axis=1))/max(np.mean(np.sum(p['y']**2,axis=1)),1e-6) for p in val_parts]))
alpha=min([0,.1,.25,.5,.75,1],key=lambda a:relative_validation(lambda ds:a*ds['phy']))
z=feature(fit); e,u=np.linalg.eigh((z.T@z).astype(float)); yy=(fit['y']-alpha*fit['phy'])/fit['mag'][:,None]
uy=u.T@(z.T@yy); best=None
for reg in [10,100,1000,10000]:
 coef=u@(uy/(e[:,None]+reg))
 def candidate(ds): return alpha*ds['phy']+(feature(ds)@coef)*ds['mag'][:,None]
 loss=relative_validation(candidate)
 if best is None or loss<best[0]: best=(loss,reg,coef)
coef=best[2]
def candidate(ds): return alpha*ds['phy']+(feature(ds)@coef)*ds['mag'][:,None]
def make_gate_data(ds):
 xx,sv=inputs(ds); c=candidate(ds); norm=ds['mag'][:,None]
 aux=np.c_[sv,c/norm,np.linalg.norm(c,axis=1)/ds['mag']]
 return xx.astype('float32'),aux.astype('float32'),c
gx,gs,gc=make_gate_data(gate_train)
oracle=np.clip(np.sum(gate_train['y']*gc,axis=1)/np.maximum(np.sum(gc*gc,axis=1),1e-12),0,1)
vdata=[(*make_gate_data(p),p) for p in val_parts]
denom=np.empty(len(gc)); weights=np.empty(len(gc))
for g in [0,1,2]:
 mask=gate_train['group']==g
 denom[mask]=max(np.mean(np.sum(gate_train['y'][mask]**2,axis=1)),1e-6)
 weights[mask]=len(gc)/(3*np.sum(mask))
unit=float(np.median(gate_train['x'][:,:,0]))
(ROOT/'gate_protocol.json').write_text(json.dumps({'candidate_fit_rows':len(fit['y']),'gate_train_rows':len(gc),
 'validation_rows':sum(len(p['y']) for p in val_parts),'physical_gate':alpha,'ridge_reg':best[1],
 'validation_objective':'mean per-part error normalized by Zero baseline, normal and severe equally represented',
 'test_status':'V103 future and V202 are previously inspected exploratory diagnostics, not fresh validation',
 'signals':'40 observations motion/IMU consistency plus observed-only summary; no mismatched VI quality logs',
 'oracle_fraction_positive':float(np.mean(oracle>.01))},indent=2))
(ROOT/'gate_split_boundaries.json').write_text(json.dumps(split_boundaries,indent=2))
def save_gate():
 (ROOT/'results.json').write_text(json.dumps({'records':records,'elapsed_s':time.time()-begun_all,
  'seeds':[0,1,2],'device':DEV,'caveats':['Offline saved mono trajectories, initial20% GT scale calibration.',
  'Candidate fit and gate-training time spans separated with input-window purge.',
  'Same-sequence V103 future and previously inspected V202 are diagnostic only; no fresh independent test.',
  'No full reprojection/feature geometry signals collected in this probe.',
  'Gating can abstain; coverage and oracle diagnostics reported to avoid claiming all-zero output as compensation.']},indent=2))
def record_gate(kind,seed,predict,detail=None):
 for name,ds in tests.items():
  c=candidate(ds); w=np.asarray(predict(ds)).reshape(-1); pred=c*w[:,None]; y=ds['y'].astype(float)
  b=float(np.sqrt(np.mean(np.sum(y*y,axis=1)))); a=float(np.sqrt(np.mean(np.sum((pred-y)**2,axis=1))))
  improve=np.sum(y*y,axis=1)-np.sum((pred-y)**2,axis=1)
  used=w>.01
  rec={'model':kind,'seed':seed,'test':name,'n':len(y),'before_m':b,'after_m':a,
   'gain_pct':100*(1-a/b),'coverage_pct':float(100*np.mean(used)),
   'mean_gate':float(np.mean(w)),'harmful_used_pct':float(100*np.mean(improve[used]<-1e-8)) if np.any(used) else 0,
   **(detail or {})}
  records.append(rec); print(json.dumps(rec),flush=True)
  np.savez_compressed(ROOT/f'{kind}_{seed}_{name}.npz',target=y,candidate=c,gate=w,prediction=pred,t=ds['t'],start=ds['start'])
 save_gate()
record_gate('Zero',-1,lambda ds:np.zeros(len(ds['y'])))
record_gate('Always',-1,lambda ds:np.ones(len(ds['y'])))
# Truth-only oracle establishes whether this candidate even admits useful protected corrections.
record_gate('Oracle_GT_only',-1,lambda ds:np.clip(np.sum(ds['y']*candidate(ds),axis=1)/np.maximum(np.sum(candidate(ds)**2,axis=1),1e-12),0,1))
# Threshold signal uses observed correction/motion scale only. Select on validation.
thresholds=[0,.02,.05,.1,.2,.3,.5,.75,1,2,4,1e9]
def signal(ds): return np.linalg.norm(candidate(ds),axis=1)/ds['mag']
threshold=min(thresholds,key=lambda th:relative_validation(lambda ds:candidate(ds)*(signal(ds)>=th)[:,None]))
record_gate('Threshold',-1,lambda ds:(signal(ds)>=threshold).astype(float),{'threshold':threshold})

class GateNet(nn.Module):
 def __init__(self,kind):
  super().__init__(); self.kind=kind
  if kind.startswith('CfC'):
   self.cell=CfC(25,48,batch_first=True,backbone_units=48,backbone_layers=1)
   forward=self.cell.rnn_cell.forward
   def adapter(x,h,ts):
    if torch.is_tensor(ts) and ts.ndim==1: ts=ts[:,None]
    return forward(x,h,ts)
   self.cell.rnn_cell.forward=adapter
  elif kind=='GRU': self.cell=nn.GRU(25,48,batch_first=True)
  else: self.cell=None
  self.head=nn.Sequential(nn.Linear((48 if self.cell is not None else 25)+gs.shape[1],48),nn.ReLU(),nn.Linear(48,1))
 def forward(self,x,s,ts):
  if self.cell is None: z=x[:,-1]
  elif self.kind.startswith('CfC'):
   out,_=self.cell(x,timespans=ts if self.kind=='CfC_time' else None); z=out[:,-1]
  else: out,_=self.cell(x); z=out[:,-1]
  return self.head(torch.cat([z,s],dim=-1)).squeeze(-1)

for kind in ['MLP','GRU','CfC_fixed','CfC_time']:
 for seed in [0,1,2]:
  torch.manual_seed(seed); model=GateNet(kind).to(DEV); opt=torch.optim.Adam(model.parameters(),lr=5e-4)
  xt=torch.tensor(gx,device=DEV); st=torch.tensor(gs,device=DEV); tt=torch.tensor(gate_train['x'][:,:,0]/unit,device=DEV)
  ct=torch.tensor(gc,dtype=torch.float32,device=DEV); yt=torch.tensor(gate_train['y'],device=DEV)
  dt=torch.tensor(denom,dtype=torch.float32,device=DEV); wt=torch.tensor(weights,dtype=torch.float32,device=DEV)
  ot=torch.tensor(oracle,dtype=torch.float32,device=DEV)
  bestloss=float('inf'); state=None; wait=0; begun=time.time()
  for ep in range(80):
   model.train(); perm=torch.randperm(len(gc),device=DEV)
   for lo in range(0,len(gc),256):
    ii=perm[lo:lo+256]; logits=model(xt[ii],st[ii],tt[ii]); gate=torch.sigmoid(logits)
    risk=((yt[ii]-gate[:,None]*ct[ii])**2).sum(1)/dt[ii]
    soft=nn.functional.binary_cross_entropy_with_logits(logits,ot[ii],reduction='none')
    loss=((risk+.1*soft)*wt[ii]).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
   model.eval(); losses=[]
   with torch.no_grad():
    for px,psn,c,p in vdata:
     w=torch.sigmoid(model(torch.tensor(px,device=DEV),torch.tensor(psn,device=DEV),torch.tensor(p['x'][:,:,0]/unit,device=DEV))).cpu().numpy()
     losses.append(np.mean(np.sum((p['y']-w[:,None]*c)**2,axis=1))/max(np.mean(np.sum(p['y']**2,axis=1)),1e-6))
   vl=float(np.mean(losses))
   if not np.isfinite(vl): raise RuntimeError('nonfinite validation')
   if vl<bestloss-1e-6: bestloss=vl; state=copy.deepcopy(model.state_dict()); bestep=ep+1; wait=0
   else: wait+=1
   if (ep+1)%10==0: print('PROGRESS',kind,seed,ep+1,vl,flush=True)
   if wait>=12: break
  model.load_state_dict(state); model.eval()
  def weights_for(ds):
   xx,sv,c=make_gate_data(ds)
   with torch.no_grad(): return torch.sigmoid(model(torch.tensor(xx,device=DEV),torch.tensor(sv,device=DEV),torch.tensor(ds['x'][:,:,0]/unit,device=DEV))).cpu().numpy()
  record_gate(kind,seed,weights_for,{'best_epoch':bestep,'epochs_run':ep+1,'seconds':time.time()-begun,
    'parameters':sum(p.numel() for p in model.parameters())})
  # Validation-only conservative gate threshold, raw soft output also retained.
  choices=[0,.1,.25,.5,.75,.9,.99,1.01]
  cut=min(choices,key=lambda cut:relative_validation(lambda ds:candidate(ds)*(weights_for(ds)*(weights_for(ds)>=cut))[:,None]))
  record_gate(kind+'_conservative',seed,lambda ds:weights_for(ds)*(weights_for(ds)>=cut),{'probability_cutoff':cut})
  torch.save({'state':state,'xm':xm,'xs':xs,'sm':sm,'ss':ss,'unit':unit,'ridge_coef':coef,'ridge_reg':best[1],'physical_gate':alpha,'probability_cutoff':cut},ROOT/f'{kind}_{seed}.pt')
  del model,xt,st,tt,ct,yt,dt,wt,ot; torch.cuda.empty_cache()
integrity=[{**m,'unchanged':hashlib.sha256(Path(m['path']).read_bytes()).hexdigest()==m['sha256']} for m in manifest]
assert all(m['unchanged'] for m in integrity)
(ROOT/'integrity_check.json').write_text(json.dumps(integrity,indent=2)); save_gate()
print('COMPLETE',time.time()-begun_all,flush=True)
