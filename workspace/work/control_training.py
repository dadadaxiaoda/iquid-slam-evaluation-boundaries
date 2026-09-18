"""Positive control: recover a known mapping on independent synthetic sequences."""
from pathlib import Path
import sys,json,time
root=Path('/opt/slam-study/workspace')
sys.argv=['control','--out',str(root/'work/control')]
ns={}; exec((root/'outputs/rerun_drift.py').read_text().split('\nraws=')[0],ns)
np=ns['np']; torch=ns['torch']; Net=ns['Net']; nn=ns['nn']; device=ns['DEV']
torch.manual_seed(42); rng=np.random.default_rng(42)
x=rng.normal(size=(600,10,7)).astype('float32'); ts=rng.uniform(.5,1.5,size=(600,10)).astype('float32')
y=x[:,-1,:3]*.6
X=torch.tensor(x,device=device); T=torch.tensor(ts,device=device); Y=torch.tensor(y,device=device)
result=[]
for kind in ['MLP','CfC_time_head']:
 torch.manual_seed(42); np.random.seed(42); m=Net(kind,7).to(device)
 opt=torch.optim.Adam(m.parameters(),lr=3e-3); best=1e9; state=None; wait=0
 for ep in range(180):
  m.train(); perm=torch.randperm(400,device=device)
  for lo in range(0,400,128):
   ii=perm[lo:lo+128]; loss=nn.functional.mse_loss(m(X[ii],T[ii]),Y[ii])
   opt.zero_grad(); loss.backward(); opt.step()
  m.eval()
  with torch.no_grad(): v=float(nn.functional.mse_loss(m(X[400:500],T[400:500]),Y[400:500]))
  if v<best: best=v; state={k:p.clone() for k,p in m.state_dict().items()}; wait=0
  else: wait+=1
  if wait>=20: break
 m.load_state_dict(state)
 with torch.no_grad(): p=m(X[500:],T[500:]).cpu().numpy()
 truth=y[500:]; r2=float(1-((p-truth)**2).sum()/((truth-truth.mean(0))**2).sum())
 rec=dict(model=kind,synthetic_test_r2=r2,epochs=ep+1,n_train=400,n_val=100,n_test=100)
 print(rec,flush=True); result.append(rec)
 assert r2>.7,rec
(root/'outputs/drift_rerun/positive_control.json').write_text(json.dumps(result,indent=2))
