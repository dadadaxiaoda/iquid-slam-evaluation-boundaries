"""Architecture sensitivity follow-up; explicitly exploratory after wired CfC results."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parent
mode=sys.argv[1] if len(sys.argv)>1 else 'cross'
if mode=='future':
 # Execute the diagnostic source transformation, intercept its final exec.
 wrapper=(root/'diagnostic_future.py').read_text().split('exec(compile(code,')[0]
 exec(compile(wrapper,str(root/'diagnostic_future.py'),'exec'),globals())
 code=code.replace("/'future_diagnostic'", "/'dense_future'")
else:
 code=(root/'validate_local.py').read_text()
 code=code.replace("OUT=Path(__file__).resolve().parent\nSTART=", "OUT=Path(__file__).resolve().parent/'dense_cross'\nOUT.mkdir(exist_ok=True)\nSTART=")
insert="""
class DenseCfC(nn.Module):
 def __init__(self,kind,dim):
  super().__init__(); self.kind=kind
  self.cell=CfC(dim,64,batch_first=True,backbone_units=64,backbone_layers=1)
  self.head=nn.Linear(64,3)
  original=self.cell.rnn_cell.forward
  def adapter(inputs,hx,ts):
   if torch.is_tensor(ts) and ts.ndim==1: ts=ts[:,None]
   return original(inputs,hx,ts)
  self.cell.rnn_cell.forward=adapter
 def forward(self,x,ts):
  z,_=self.cell(x,timespans=ts if self.kind=='DenseCfC_time' else None)
  return self.head(z[:,-1])
Net=DenseCfC
"""
at=code.index('for kind in [')
code=code[:at]+insert+code[at:]
code=code.replace("['MLP','GRU','CfC_fixed_head','CfC_time_head']", "['DenseCfC_fixed','DenseCfC_time']")
code=code.replace("['GRU','CfC_fixed_head','CfC_time_head']", "['DenseCfC_fixed','DenseCfC_time']")
exec(compile(code,str(root/'validate_local.py'),'exec'),globals())
