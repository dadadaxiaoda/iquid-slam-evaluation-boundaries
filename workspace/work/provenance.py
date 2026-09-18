from pathlib import Path
import hashlib,json,shutil
root=Path('/opt/slam-study/workspace')
base=Path('/opt/slam-study/legacy'); sources=list((base/'data').glob('*scale_label.npz'))
sources+=list((base/'data').glob('*dataset*.npz'))+list((base/'data').glob('V*slam_tum.txt'))
scripts=['kitti_train_multi.py','kitti_dataset_instr.py','vi_dataset_instr.py','a1_dataset.py','a1_seq_analyze.py']
snap=root/'work'/'legacy_sources'; snap.mkdir(exist_ok=True)
for name in scripts:
 f=base/name; shutil.copy2(f,snap/name); sources.append(f)
for seq in ['V101','V102','V103','V201','V202']:
 p=Path('/opt/slam-study/datasets')/seq/'mav0'
 sources += [p/'cam0/sensor.yaml',p/'imu0/data.csv',p/'state_groundtruth_estimate0/data.csv']
manifest={str(f):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sources}
out=root/'outputs'/'drift_rerun'/'input_manifest.json'; out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(manifest,indent=2)); print('hashed inputs',len(manifest))
