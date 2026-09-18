from pathlib import Path
root=Path(__file__).resolve().parent
code=(root/'validate_local.py').read_text().split("train=concatenate(['V101','V102'])")[0]
code=code.replace("OUT=Path(__file__).resolve().parent\nSTART=", "OUT=Path(__file__).resolve().parent/'data_diagnostics'\nOUT.mkdir(exist_ok=True)\nSTART=")
exec(compile(code,str(root/'validate_local.py'),'exec'),globals())
stats=[]
for name,conditions in datasets.items():
 for condition,ds in conditions.items():
  norms=np.linalg.norm(ds['y'],axis=1)
  stats.append({'sequence':name,'condition':condition,'n':len(norms),
   'local_error_rms_m':float(np.sqrt(np.mean(norms**2))),
   'local_error_quantiles_m':np.quantile(norms,[0,.5,.9,.99,1]).tolist(),
   'horizon_range_s':[float(ds['horizon'].min()),float(ds['horizon'].max())],
   'history_range_s':[float((ds['t']-ds['start']).min()),float((ds['t']-ds['start']).max())]})
(root/'data_distribution.json').write_text(json.dumps(stats,indent=2))
print(json.dumps([s for s in stats if s['condition']=='nominal'],indent=2))
