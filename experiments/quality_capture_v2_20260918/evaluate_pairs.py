"""Acceptance and same-replay comparison of coordinate-consistent online pairs."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
base=ROOT.parent/'quality_capture_20260918/analyze_quality.py'
namespace={'__file__':str(ROOT/'evaluate_pairs.py')}
exec(compile(base.read_text().split('indices=[]')[0],str(base),'exec'),namespace)
q=namespace['q'];a=namespace['a'];t=namespace['t'];gtvalid=namespace['gv'];gp=namespace['gp'];gr=namespace['gr']
future=namespace['future'];threshold=namespace['threshold'];n=len(t)
path=ROOT/'V202/relative_pairs.csv';before=hashlib.sha256(path.read_bytes()).hexdigest()
pairs=np.genfromtxt(path,delimiter=',',names=True)
assert len(pairs)==n and np.array_equal(pairs['input_id'],a['input_id']) and np.max(abs(pairs['end_ts']-t))<1e-9
pairvalid=pairs['pair_valid']==1
columns=['dx','dy','dz','qx','qy','qz','qw']
assert all(np.all(np.isnan(pairs[c][~pairvalid])) for c in columns)
assert all(np.all(np.isfinite(pairs[c][pairvalid])) for c in columns)
assert np.all(abs(pairs['end_ts'][pairvalid]-pairs['start_ts'][pairvalid]-1)<.002)
assert np.all(pairs['imu_initialized'][pairvalid]==1)
indices=[];errors=[];rotation_errors=[];raw=[];exported=[];crossbig=[];crossupdate=[]
for i in range(39,n-20,5):
 j=i+20
 if t[i-39]-t[0]<15 or not np.all(q['state'][i-39:j+1]==2):continue
 if not np.all(q['pose_valid'][i-39:j+1]==1) or not gtvalid[i] or not gtvalid[j] or not pairvalid[j]:continue
 if abs(pairs['start_ts'][j]-t[i])>.002:continue
 displacement=np.array([pairs[c][j] for c in ['dx','dy','dz']])
 gt_displacement=gr[i].T@(gp[j]-gp[i])
 error=float(np.linalg.norm(displacement-gt_displacement))
 estimated_rotation=namespace['Rotation'].from_quat([pairs[c][j] for c in ['qx','qy','qz','qw']]).as_matrix()
 expected_rotation=gr[i].T@gr[j]
 rotation_error=float(namespace['Rotation'].from_matrix(expected_rotation.T@estimated_rotation).magnitude())
 indices.append(i);errors.append(error);rotation_errors.append(rotation_error);raw.append(future[i]);exported.append(namespace['saved_rpe'](i,j))
 crossbig.append(namespace['crossed'](i,j,namespace['context_fields']));crossupdate.append(namespace['crossed'](i,j,['map_change_index']))
indices=np.array(indices);errors=np.array(errors);raw=np.array(raw);exported=np.array(exported)
def stats(values):
 v=np.asarray(values);v=v[np.isfinite(v)]
 return {'windows':len(v),'median_cm':float(np.median(v)*100),'p90_cm':float(np.quantile(v,.9)*100),
 'rmse_cm':float(np.sqrt(np.mean(v*v))*100),'max_cm':float(np.max(v)*100),'above_frozen_threshold':int(np.sum(v>threshold))} if len(v) else None
largest=np.argsort(raw)[-5:][::-1]
examples=[{'forecast_start_offset_s':float(t[indices[k]]-t[0]),'raw_error_cm':float(raw[k]*100),
 'consistent_online_pair_error_cm':float(errors[k]*100),'final_export_diagnostic_error_cm':float(exported[k]*100),
 'big_context_crossing':bool(crossbig[k]),'map_update_crossing':bool(crossupdate[k])} for k in largest]
positive=errors>threshold
positive_ids=indices[positive]
groups=np.split(positive_ids,np.flatnonzero(np.diff(positive_ids)>5)+1) if len(positive_ids) else []
report={'scope':'One extra V202 replay, no training, coordinate-label acceptance only.',
 'label_method':'At the 1-second endpoint reconstruct both frame poses from their stored frame-to-reference transforms and current reference-keyframe poses under map update lock.',
 'causality':'Reconstruction uses map state available at endpoint, never final-export poses; final export is diagnostic only.',
 'threshold_m':threshold,'threshold_note':'Old frozen threshold used for comparability only; future training requires a newly verified training-label protocol.',
 'raw_online_absolute_output_difference':stats(raw),'coordinate_consistent_online_pairs':stats(errors),
 'final_export_diagnostic_same_windows':stats(exported),'valid_pair_rows':int(pairvalid.sum()),
 'rotation_error_median_degrees':float(np.median(rotation_errors)*180/np.pi),
 'consistent_label_risk_episodes':len(groups),'largest_raw_error_examples':examples,
 'training_runs':0,'decision':'No model fitting unless verified labels provide enough independent events.',
 'limitations':['Previously inspected sequence, one corrected replay','Only local 1-second relative motion, not global accumulated drift',
 'Past pose is revised with endpoint map state; target is localization after that update, not error of the original immutable historical output',
 'No classification of all individual loop/BA event types','Cannot infer CfC superiority from instrumentation repair']}
original=json.loads((ROOT/'original_manifest.json').read_text())
report['original_files_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in original.items());assert report['original_files_unchanged']
assert hashlib.sha256(path.read_bytes()).hexdigest()==before
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in namespace['hashes'].items())
(ROOT/'pair_evaluation.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(ROOT/'pair_evaluation_arrays.npz',input_id=indices,raw_error=raw,consistent_error=errors,exported_error=exported,positive=positive)
print(json.dumps(report,indent=2))
