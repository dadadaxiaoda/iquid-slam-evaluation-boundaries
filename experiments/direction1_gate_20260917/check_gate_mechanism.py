from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parent; rows=[]; variance=[]
for label,folder in [('old_saved_input',root),('fresh_online_quality',root/'quality_gate')]:
 for test in ['V103_future_nominal','V202_independent_nominal']:
  with np.load(folder/f'Always_-1_{test}.npz') as d:
   y=d['target']; c=d['candidate']; b=np.sqrt(np.mean(np.sum(y*y,axis=1)))
   for gate in [.25,.5,.75]:
    after=np.sqrt(np.mean(np.sum((y-gate*c)**2,axis=1)))
    rows.append({'stage':label,'test':test,'fixed_weight':gate,'before_m':float(b),'after_m':float(after),
     'gain_pct':float(100*(1-after/b))})
  for seed in [0,1,2]:
   with np.load(folder/f'CfC_time_{seed}_{test}.npz') as d:
    w=d['gate']; oracle=np.clip(np.sum(d['target']*d['candidate'],axis=1)/np.maximum(np.sum(d['candidate']**2,axis=1),1e-12),0,1)
    variance.append({'stage':label,'test':test,'seed':seed,'mean_weight':float(w.mean()),'std_weight':float(w.std()),
     'weight_range':[float(w.min()),float(w.max())],'oracle_weight_correlation':float(np.corrcoef(w,oracle)[0,1])})
(root/'gate_mechanism_diagnostic.json').write_text(json.dumps({'fixed_weight_controls':rows,'liquid_weight_variation':variance,
 'status':'Post-hoc mechanism diagnostic. Fixed .25/.5/.75 controls are not tuned on test, but were added after primary results; require future validation.'},indent=2))
print(json.dumps([r for r in rows if r['stage']=='fresh_online_quality'],indent=2))
print(json.dumps([r for r in variance if r['stage']=='fresh_online_quality'],indent=2))
