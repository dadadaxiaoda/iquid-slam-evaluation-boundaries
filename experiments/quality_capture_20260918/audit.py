from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;folder=ROOT/'V202'
q=np.genfromtxt(folder/'quality.csv',delimiter=',',names=True)
a=np.genfromtxt(folder/'online.csv',delimiter=',',names=True)
stamps=np.loadtxt(folder/'timestamps.txt')/1e9
checks={}
checks['one_record_per_image']=len(q)==len(a)==len(stamps)
checks['input_ids_contiguous']=np.array_equal(q['input_id'],np.arange(len(q))) and np.array_equal(a['input_id'],q['input_id'])
checks['timestamps_match']=np.max(abs(q['ts']-a['ts']))<1e-9 and np.max(abs(q['ts']-stamps))<1e-6
checks['states_match']=np.array_equal(q['state'],a['state'])
valid=q['pose_valid']==1
pose_columns=['px','py','pz','qx','qy','qz','qw']
checks['pose_matches_api_output']=all(np.max(abs(q[c][valid]-a[c][valid]))<1e-8 for c in pose_columns)
checks['pose_missing_marked']=all(np.all(np.isnan(q[c][~valid])) for c in pose_columns)
checks['counts_consistent']=np.all(q['n_inlier_observed']<=q['n_associated']) and np.all(q['n_associated']<=q['n_kps']) and np.all(q['n_reproj']<=q['n_inlier_observed'])
has=q['n_reproj']>0
checks['reprojection_missing_marked']=all(np.all(np.isnan(q[c][~has])) for c in ['reproj_median_px','reproj_p90_px','reproj_mean_px'])
checks['reprojection_finite_when_available']=all(np.all(np.isfinite(q[c][has])) for c in ['reproj_median_px','reproj_p90_px','reproj_mean_px'])
checks['quantiles_ordered']=np.all(q['reproj_median_px'][has]<=q['reproj_p90_px'][has])
for key in ['depth_valid_fraction','inlier_grid_coverage']:
 values=q[key][np.isfinite(q[key])];checks[key+'_bounded']=bool(np.all((values>=0)&(values<=1)))
iv=q['imu_interval_valid']==1
checks['imu_missing_marked']=np.all(np.isnan(q['imu_rotation_residual_rad'][~iv]))
checks['imu_valid_nonnegative']=np.all(np.isfinite(q['imu_rotation_residual_rad'][iv])) and np.all(q['imu_rotation_residual_rad'][iv]>=0)
change=np.r_[False,(np.diff(q['map_id'])!=0)|(np.diff(q['reset_epoch'])!=0)|(np.diff(q['map_create_epoch'])!=0)|(np.diff(q['big_change_index'])!=0)]
checks['context_flags_consistent']=np.array_equal(change,q['context_changed']==1)
checks['reset_epoch_monotonic']=np.all(np.diff(q['reset_epoch'])>=0) and np.all(np.diff(q['map_create_epoch'])>=0)
manifest=json.loads((ROOT/'original_manifest.json').read_text())
checks['all_original_files_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
def stats(x):
 x=x[np.isfinite(x)]
 return {'mean':float(np.mean(x)),'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'max':float(np.max(x))} if len(x) else None
summary={'rows':len(q),'pose_valid_rows':int(valid.sum()),'reprojection_available_rows':int(has.sum()),
 'imu_initialized_rows':int(q['imu_initialized'].sum()),'imu_consistency_available_rows':int(iv.sum()),
 'map_ids':np.unique(q['map_id']).astype(int).tolist(),'reset_events':int(q['reset_epoch'][-1]),'map_create_events':int(q['map_create_epoch'][-1]),
 'map_big_change_transitions':int(np.count_nonzero(np.diff(q['big_change_index']))),'context_change_rows':int(change.sum()),
 'frame_timestamp_mismatch_rows':int(np.count_nonzero(abs(q['frame_ts']-q['ts'])>1e-6)),
 'reprojection_median_px':stats(q['reproj_median_px']),'grid_coverage':stats(q['inlier_grid_coverage']),
 'imu_residual_degrees':stats(q['imu_rotation_residual_rad']*180/np.pi),
 'logger_compute_and_lock_ms':stats(q['logger_ms']),'api_including_logger_ms':stats(a['api_ms']),
 'timing_note':'logger_ms includes metric computation and map-lock waiting; excludes CSV formatting/write. API timing includes entire callback.',
 'context_note':'Map-change counters identify context changes but do not uniquely classify every loop/BA event. last_init_id=-1 until a reset/map creation establishes it.',
 'scope':'One V202 replay, instrumentation acceptance only; no training or SLAM improvement claim.'}
report={'passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'summary':summary}
(ROOT/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));assert all(checks.values())
