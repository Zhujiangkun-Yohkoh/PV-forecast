"""Synthetic forward-only module participation, no checkpoint loading or updates."""
from pathlib import Path
import json,sys
import torch
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'scheme_A_submission_correction'))
import run_corrected_benchmark as b
torch.set_num_threads(2);records=[]
for width in [17,7]:
 for name in b.MODEL_NAMES:
  model=b.make_model(name,width,b.load_config()).eval();seen=set();hooks=[]
  # PyTorch MHA invokes its output projection functionally; hook the MHA owner.
  expected={n for n,m in model.named_modules() if list(m.parameters(recurse=False)) and not n.endswith('.out_proj')}
  for n,m in model.named_modules():
   if n in expected:hooks.append(m.register_forward_hook(lambda module,args,out,n=n:seen.add(n)))
  # Disable fused attention paths so ordinary module hooks are observable.
  torch.backends.mha.set_fastpath_enabled(False)
  with torch.inference_mode():y=model(torch.zeros(2,72,width))
  for hook in hooks:hook.remove()
  assert y.shape==(2,144) and torch.isfinite(y).all();assert expected<=seen,(name,expected-seen)
  records.append(dict(model=name,input_dim=width,parameters=sum(p.numel() for p in model.parameters()),parameter_owning_modules=len(expected),all_modules_observed=True))
out=dict(passed=8,failed=0,skipped=0,checkpoint_loaded=False,training=False,records=records)
(H/'results/MODEL_FORWARD_AUDIT.json').write_text(json.dumps(out,indent=2));print(out)
