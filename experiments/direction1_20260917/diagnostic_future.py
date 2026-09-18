"""Explicit same-sequence future diagnostic, separate from primary cross-sequence test."""
from pathlib import Path
root=Path(__file__).resolve().parent
code=(root/'validate_local.py').read_text()
code=code.replace("OUT=Path(__file__).resolve().parent\nSTART=", "OUT=Path(__file__).resolve().parent/'future_diagnostic'\nOUT.mkdir(exist_ok=True)\nSTART=")
code=code.replace("train=concatenate(['V101','V102']); val=concatenate(['V201'])", """
train_parts=[]; val_parts=[]
for condition,ds in datasets['V103'].items():
 lo,hi=np.quantile(ds['t'],[.6,.8])
 tr=ds['t']<=lo; va=(ds['start']>lo)&(ds['t']<=hi); te=ds['start']>hi
 train_parts.append({k:ds[k][tr] for k in ['x','y']})
 val_parts.append({k:ds[k][va] for k in ['x','y']})
 datasets['V103'][condition]={k:v[te] for k,v in ds.items()}
train={k:np.concatenate([d[k] for d in train_parts]) for k in ['x','y']}
val={k:np.concatenate([d[k] for d in val_parts]) for k in ['x','y']}
""")
code=code.replace("for name in ['V103','V202']:","for name in ['V103']:")
code=code.replace("'train':['V101','V102'],'validation':['V201'],'test':['V103','V202']", "'train':['V103 past60%'],'validation':['V103 next20% purged'],'test':['V103 final20% purged']")
code=code.replace("['MLP','GRU','CfC_fixed_head','CfC_time_head']", "['GRU','CfC_fixed_head','CfC_time_head']")
exec(compile(code,str(root/'validate_local.py'),'exec'),globals())
