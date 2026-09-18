from pathlib import Path
import json,csv
import numpy as np
root=Path(__file__).resolve().parent
for folder in [root,root/'quality_gate']:
 if not (folder/'results.json').exists(): continue
 data=json.loads((folder/'results.json').read_text()); groups={}; checks=[]
 for rec in data['records']:
  p=folder/f"{rec['model']}_{rec['seed']}_{rec['test']}.npz"
  with np.load(p) as d:
   assert np.isfinite(d['gate']).all() and np.all(d['gate']>=0) and np.all(d['gate']<=1)
   pred=d['candidate']*d['gate'][:,None]; np.testing.assert_allclose(pred,d['prediction'],atol=1e-8)
   before=float(np.sqrt(np.mean(np.sum(d['target']**2,axis=1))))
   after=float(np.sqrt(np.mean(np.sum((d['target']-pred)**2,axis=1))))
   coverage=float(100*np.mean(d['gate']>.01))
   assert abs(before-rec['before_m'])<1e-8 and abs(after-rec['after_m'])<1e-8 and abs(coverage-rec['coverage_pct'])<1e-8
   checks.append({'file':str(p),'verified':True})
  groups.setdefault((rec['model'],rec['test']),[]).append(rec)
 rows=[]
 for (model,test),rr in groups.items():
  rows.append({'model':model,'test':test,'runs':len(rr),'before_m':rr[0]['before_m'],
   'after_m':float(np.mean([r['after_m'] for r in rr])),'gain_mean':float(np.mean([r['gain_pct'] for r in rr])),
   'gain_std':float(np.std([r['gain_pct'] for r in rr])),'coverage_mean':float(np.mean([r['coverage_pct'] for r in rr])),
   'mean_gate':float(np.mean([r['mean_gate'] for r in rr]))})
 with (folder/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
 integrity=json.loads((folder/'integrity_check.json').read_text()); assert all(e['unchanged'] for e in integrity)
 (folder/'verification.json').write_text(json.dumps({'records_verified':len(checks),'inputs_unchanged':len(integrity),'checks':checks},indent=2))
 print('VERIFIED',folder,len(checks),flush=True)
 for row in rows:
  if row['test'].endswith('nominal') and row['model'] in ['Always','Oracle_GT_only','Threshold','GRU','CfC_fixed','CfC_time','CfC_time_conservative']: print(json.dumps(row),flush=True)
folder=root/'repeat_check'
if (folder/'results.json').exists():
 data=json.loads((folder/'results.json').read_text()); groups={}
 for rec in data['records']:
  with np.load(folder/f"{rec['model']}_{rec['seed']}_{rec['condition']}.npz") as d:
   np.testing.assert_allclose(d['candidate']*d['gate'][:,None],d['prediction'],atol=1e-8)
   before=np.sqrt(np.mean(np.sum(d['target']**2,axis=1)))
   after=np.sqrt(np.mean(np.sum((d['target']-d['prediction'])**2,axis=1)))
   assert abs(before-rec['before_m'])<1e-8 and abs(after-rec['after_m'])<1e-8
  groups.setdefault((rec['model'],rec['condition']),[]).append(rec)
 rows=[]
 for (model,condition),rr in groups.items():
  rows.append({'model':model,'condition':condition,'runs':len(rr),'before_m':rr[0]['before_m'],
   'after_m':float(np.mean([r['after_m'] for r in rr])),'gain_mean':float(np.mean([r['gain_pct'] for r in rr])),
   'gain_std':float(np.std([r['gain_pct'] for r in rr]))})
 with (folder/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
 assert all(r['unchanged'] for r in json.loads((folder/'integrity_check.json').read_text()))
 (folder/'verification.json').write_text(json.dumps({'records_verified':len(data['records']),'inputs_unchanged':6,'exit_code':data['exit_code']}))
 print('REPEAT_VERIFIED',len(data['records']),flush=True)
 for row in rows:
  if row['condition']=='nominal': print(json.dumps(row),flush=True)
