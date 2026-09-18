from pathlib import Path
import json,csv
root=Path(__file__).resolve().parent
source=(root/'run.py').read_text().split("for name in ['V101'")[0]
exec(compile(source,str(root/'run.py'),'exec'),globals())
ti=np.arange(0,1.01,.01); imu=np.c_[ti,np.zeros((len(ti),3)),np.tile([0,0,9.81],(len(ti),1))]
g=np.array([0,0,9.81]); t0=.012; t1=.509; dt=t1-t0
p,v,r=preintegrate(imu,ti,t0,t1,np.eye(3),g)
np.testing.assert_allclose(p,0,atol=1e-12); np.testing.assert_allclose(v,0,atol=1e-12)
imu[:,4]=.3
p,v,r=preintegrate(imu,ti,t0,t1,np.eye(3),g)
np.testing.assert_allclose(p,[.5*.3*dt**2,0,0],atol=1e-12)
np.testing.assert_allclose(v,[.3*dt,0,0],atol=1e-12)
imu[:,4]=0; imu[:,3]=.2
p,v,r=preintegrate(imu,ti,t0,t1,np.eye(3),g)
np.testing.assert_allclose(r,[0,0,.2*dt],atol=1e-12)
np.testing.assert_allclose(p,0,atol=1e-12)
if '--fixtures-only' in sys.argv:
 print('PREINTEGRATION_FIXTURES_PASSED'); sys.exit(0)
data=json.loads((root/'results.json').read_text()); checks=[]; groups={}
with np.load(root/'split_arrays.npz') as split:
 lo,hi=split['v103_bounds']
 raw=np.loadtxt('/opt/slam-study/legacy/data/V103_slam_tum.txt')
 low,high=raw[:,0].min(),raw[:,0].max()
 tr=(split['train_t']>=low)&(split['train_t']<=high)
 va=(split['val_t']>=low)&(split['val_t']<=high)
 assert np.all(split['train_t'][tr]<=lo)
 assert np.all(split['val_start'][va]>lo) and np.all(split['val_t'][va]<=hi)
for rec in data['records']:
 path=root/f"{rec['model']}_{rec['seed']}_{rec['test']}.npz"
 with np.load(path) as d:
  y=d['target']; pred=d['prediction']; b=float(np.sqrt(np.mean(np.sum(y*y,axis=1))))
  a=float(np.sqrt(np.mean(np.sum((pred-y)**2,axis=1))))
  assert np.isfinite(pred).all() and np.isfinite(y).all() and np.all(d['mag']>=.04999)
  if rec['test'].startswith('V103_future'): assert np.all(d['start']>hi)
  assert abs(a-rec['after_m'])<1e-7 and abs(b-rec['before_m'])<1e-7
  checks.append({'file':str(path),'max_metric_difference':max(abs(a-rec['after_m']),abs(b-rec['before_m']))})
 groups.setdefault((rec['model'],rec['test']),[]).append(rec)
rows=[]
for (model,test),rr in groups.items():
 rows.append({'model':model,'test':test,'runs':len(rr),'before_m':rr[0]['before_m'],
  'after_m':float(np.mean([r['after_m'] for r in rr])),
  'gain_mean':float(np.mean([r['gain_pct'] for r in rr])),
  'gain_std':float(np.std([r['gain_pct'] for r in rr]))})
with (root/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
integrity=json.loads((root/'integrity_check.json').read_text()); assert all(r['unchanged'] for r in integrity)
(root/'verification.json').write_text(json.dumps({'preintegration_checks':'stationary,constant acceleration,constant angular velocity passed',
 'temporal_purge_check':'passed','predictions':checks,'unchanged_inputs':len(integrity)},indent=2))
print('VERIFIED',len(checks),'UNCHANGED_INPUTS',len(integrity))
for row in rows:
 if row['test'].endswith('nominal'): print(json.dumps(row),flush=True)
