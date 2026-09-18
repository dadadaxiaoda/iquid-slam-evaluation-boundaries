from pathlib import Path
import json,numpy as np,collections,hashlib
root=Path('/opt/slam-study/workspace/outputs/drift_rerun')
d=json.loads((root/'results.json').read_text()); rows=d['records']
keys=[(r['case'],r['config'],r['model'],r['seed'],r['test_sequence']) for r in rows]
assert len(keys)==len(set(keys))==440,(len(keys),len(set(keys)))
group=collections.Counter((r['case'],r['config'],r['test_sequence']) for r in rows)
assert len(group)==20 and set(group.values())=={22},group
assert len(list(root.glob('*.pt')))==342
max_delta=0
for r in rows:
 f=root/f"{r['case']}_{r['config']}_{r['model']}_{r['seed']}_{r['test_sequence']}.npz"
 with np.load(f) as a:
  p=a['prediction']; rot=a['rotation']; pa=a['baseline']; gt=a['gt']; t=a['t']
  assert len(t)==r['n_test'] and np.all(np.diff(t)>0)
  assert all(np.isfinite(a[k]).all() for k in a.files)
  corr=pa+np.einsum('nij,nj->ni',rot,p.astype(float))
  before=float(np.sqrt(np.mean(np.sum((gt-pa)**2,axis=1))))
  after=float(np.sqrt(np.mean(np.sum((gt-corr)**2,axis=1))))
  delta=max(abs(before-r['ate_before_m']),abs(after-r['ate_after_m']))
  max_delta=max(max_delta,delta); assert delta<1e-7,(f,delta)
  assert abs(100*(1-after/before)-r['improvement_pct'])<1e-6
manifest=json.loads((root/'input_manifest.json').read_text())
changed=[p for p,v in manifest.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v['sha256']]
assert not changed,changed
check=dict(records_checked=len(rows),test_configurations=len(group),neural_checkpoints=342,
 max_ate_recomputation_delta_m=max_delta,original_files_checked=len(manifest),changed_original_files=changed)
(root/'result_validation.json').write_text(json.dumps(check,indent=2)); print(check)
