"""One additional baseline replay and frozen gate evaluation; same environment, not fresh dataset."""
from pathlib import Path
import subprocess,os,json,time,resource
import numpy as np
root=Path(__file__).resolve().parent; dst=root/'repeat_check/V103'; dst.mkdir(parents=True,exist_ok=True)
repo=Path('/opt/slam-study/ORB_SLAM3'); env=os.environ.copy(); env['ORB_SLAM3_INSTR_DIR']=str(dst)
resource.setrlimit(resource.RLIMIT_CORE,(0,0)); start=time.time()
command=[str(root/'matched/mono_euroc_headless'),str(repo/'Vocabulary/ORBvoc.txt'),str(repo/'Examples/Monocular/EuRoC.yaml'),'/opt/slam-study/datasets/V103',str(repo/'Examples/Monocular/EuRoC_TimeStamps/V103.txt'),'V103']
print('REPEAT_REPLAY_START',flush=True)
with (dst/'run.log').open('w') as log: done=subprocess.run(command,cwd=dst,env=env,stdout=log,stderr=subprocess.STDOUT)
assert done.returncode==0, f'Baseline repeat failed: {done.returncode}'
a=np.loadtxt(dst/'online_camera_tum.txt'); np.savetxt(dst/'V103_slam_tum.txt',a,fmt='%.12f')
print('REPEAT_REPLAY_DONE',time.time()-start,len(a),flush=True)
source=(root/'quality_gate/recipe2_reference.py').read_text().split('tr=[]; va=[]; tests={}')[0]
source=source.replace("/'quality_gate'; W=40", "/'repeat_check'; W=40")
source=source.replace("for name in ['V101','V102','V201','V103','V202']:", "for name in ['V103']:")
source=source.replace('/direction1_gate_20260917/matched','/direction1_gate_20260917/repeat_check')
(root/'repeat_check/prepared_loader.py').write_text(source)
exec(compile(source,str(root/'repeat_check/prepared_loader.py'),'exec'),globals())
gx=np.zeros((1,40,38)); gs=np.zeros((1,23))
code=(root/'quality_gate/generated_run_gate.py').read_text()
classcode=code[code.index('class GateNet(nn.Module):'):code.index("for kind in ['MLP'")]
exec(compile(classcode,'frozen_GateNet','exec'),globals())
with np.load(root/'quality_gate/split_arrays.npz') as split: hi=float(split['v103_bounds'][1])
rows=[]
for cond,ds0 in data['V103'].items():
 ds={k:v[ds0['start']>hi] for k,v in ds0.items()}
 assert len(ds['y'])>20
 def features(checkpoint):
  x=(ds['x']-checkpoint['xm'])/checkpoint['xs']; s=(ds['s']-checkpoint['sm'])/checkpoint['ss']
  z=np.c_[x.reshape(len(x),-1),s,np.ones(len(x))]
  c=checkpoint['physical_gate']*ds['phy']+(z@checkpoint['ridge_coef'])*ds['mag'][:,None]
  aux=np.c_[s,c/ds['mag'][:,None],np.linalg.norm(c,axis=1)/ds['mag']]
  return x.astype('float32'),aux.astype('float32'),c
 checkpoint=torch.load(root/'quality_gate/CfC_time_0.pt',map_location='cpu',weights_only=False)
 x,s,c=features(checkpoint)
 def record(kind,seed,w):
  y=ds['y'].astype(float); pred=c*np.asarray(w)[:,None]
  b=float(np.sqrt(np.mean(np.sum(y*y,axis=1)))); a=float(np.sqrt(np.mean(np.sum((pred-y)**2,axis=1))))
  row={'model':kind,'seed':seed,'condition':cond,'n':len(y),'before_m':b,'after_m':a,'gain_pct':100*(1-a/b),'coverage_pct':float(100*np.mean(np.asarray(w)>.01))}
  rows.append(row); print(json.dumps(row),flush=True)
  np.savez_compressed(root/'repeat_check'/f'{kind}_{seed}_{cond}.npz',target=y,candidate=c,gate=w,prediction=pred)
 record('Zero',-1,np.zeros(len(c))); record('Always',-1,np.ones(len(c))); record('FixedHalf',-1,np.full(len(c),.5))
 for kind in ['GRU','CfC_time']:
  for seed in [0,1,2]:
   checkpoint=torch.load(root/'quality_gate'/f'{kind}_{seed}.pt',map_location='cpu',weights_only=False)
   x,s,_=features(checkpoint); model=GateNet(kind).to(DEV).eval(); model.load_state_dict(checkpoint['state'])
   with torch.no_grad(): w=torch.sigmoid(model(torch.tensor(x,device=DEV),torch.tensor(s,device=DEV),torch.tensor(ds['x'][:,:,0]/checkpoint['unit'],device=DEV))).cpu().numpy()
   record(kind,seed,w); del model
integrity=[{**m,'unchanged':hashlib.sha256(Path(m['path']).read_bytes()).hexdigest()==m['sha256']} for m in manifest]
assert all(m['unchanged'] for m in integrity)
(root/'repeat_check/integrity_check.json').write_text(json.dumps(integrity,indent=2))
(root/'repeat_check/results.json').write_text(json.dumps({'records':rows,'exit_code':done.returncode,'replay_seconds':time.time()-start,
 'protocol':'Frozen models and original first-run future boundary; no retraining; same images/environment, not independent environment validation.'},indent=2))
print('REPEAT_CHECK_COMPLETE',flush=True)
