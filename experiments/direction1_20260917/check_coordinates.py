"""Independent homogeneous-transform spot check of local camera-frame labels."""
from pathlib import Path
import json,numpy as np,yaml
from scipy.spatial.transform import Rotation,Slerp
root=Path(__file__).resolve().parent; audit=json.loads((root/'data_audit.json').read_text()); checks=[]
for name in ['V103','V202']:
 base=Path('/opt/slam-study/datasets')/name/'mav0'
 a=np.loadtxt(Path('/opt/slam-study/legacy/data')/f'{name}_slam_tum.txt')
 a=a[np.isfinite(a).all(1)&(np.linalg.norm(a[:,4:8],axis=1)>.5)]
 keep=np.unique(a[:,0],return_index=True)[1]; keep.sort(); a=a[keep]
 g=np.loadtxt(base/'state_groundtruth_estimate0/data.csv',delimiter=',',comments='#'); tg=g[:,0]/1e9
 a=a[(a[:,0]>=tg[0])&(a[:,0]<=tg[-1])]; t=a[:,0]
 Tbc=np.array(yaml.safe_load((base/'cam0/sensor.yaml').read_text())['T_BS']['data']).reshape(4,4)
 rb=Slerp(tg,Rotation.from_quat(g[:,[5,6,7,4]]))(t).as_matrix()
 pb=np.column_stack([np.interp(t,tg,g[:,k]) for k in [1,2,3]])
 rs=Rotation.from_quat(a[:,4:8]).as_matrix(); scale=audit[name]['scale_initial20']
 for condition in ['nominal','drop25','drop50']:
  with np.load(root/f'Zero_-1_{name}_{condition}.npz') as d:
   for row in np.linspace(0,len(d['idx'])-1,7,dtype=int):
    end=int(d['idx'][row]); start=int(np.argmin(abs(t-(d['timestamp'][row]-d['horizon'][row]))))
    transforms=[]; slamtransforms=[]
    for k in [start,end]:
     Tb=np.eye(4); Tb[:3,:3]=rb[k]; Tb[:3,3]=pb[k]; transforms.append(Tb@Tbc)
     Ts=np.eye(4); Ts[:3,:3]=rs[k]; Ts[:3,3]=a[k,1:4]; slamtransforms.append(Ts)
    truth=(np.linalg.inv(transforms[0])@transforms[1])[:3,3]
    slam=scale*(np.linalg.inv(slamtransforms[0])@slamtransforms[1])[:3,3]
    delta=float(np.max(abs(truth-slam-d['target'][row])))
    assert delta<1e-5
    checks.append({'sequence':name,'condition':condition,'row':int(row),'max_difference_m':delta})
(root/'coordinate_validation.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
print('INDEPENDENT_COORDINATE_CHECKS',len(checks),'MAX_ERROR',max(r['max_difference_m'] for r in checks))
