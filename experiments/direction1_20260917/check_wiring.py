from pathlib import Path
import sys,json
root=Path(__file__).resolve().parent
sys.argv=['check','--out',str(root/'wiring_check')]
src=(root/'legacy_runner_reference.py').read_text().split('raws={n:load(n)')[0]
exec(compile(src,str(root/'legacy_runner_reference.py'),'exec'),globals())
torch.manual_seed(7)
a=Net('CfC_time_head',19).cpu().eval(); b=Net('CfC_fixed_head',19).cpu().eval()
b.load_state_dict(a.state_dict()); x=torch.randn(4,20,19); ts=torch.ones(4,20)
with torch.no_grad():
 p=a(x,ts); q=a(x,ts*2); f=b(x,ts); g=b(x,ts*2)
assert p.shape==(4,3) and torch.isfinite(p).all()
assert not torch.allclose(p,q) and torch.allclose(f,g)
(root/'wiring_validation.json').write_text(json.dumps({'batch_shape':list(p.shape),
 'time_model_change':float((p-q).abs().max()),'fixed_model_change':float((f-g).abs().max()),
 'passed':True},indent=2))
print('TIME_WIRING_PASSED')
