from pathlib import Path
import numpy as np, torch, json
p=Path('/opt/slam-study/legacy/data')
for name in ['K00i','K04i','K00','K04','K03','V101','V103','V103vi']:
 for tail in ['scale_label','dataset_instr','dataset_imu']:
  f=p/f'{name}_{tail}.npz'
  if not f.exists(): continue
  with np.load(f,allow_pickle=False) as d:
   print(f.name,{k: (d[k].shape,str(d[k].dtype)) for k in d.files},flush=True)
   if tail=='scale_label':
    for k in ['t','t_abs','S_GLOBAL']:
     if k in d: print(k,d[k].reshape(-1)[:3],d[k].reshape(-1)[-3:])
print('cuda',torch.cuda.is_available(),torch.__version__)
if torch.cuda.is_available():
 x=torch.ones(4,device='cuda'); print('cuda_compute',float(x.sum()))
