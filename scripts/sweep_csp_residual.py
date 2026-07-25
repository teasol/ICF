from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import lightning as L
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.datasets.synthetic_data import RESPONSE_TASK_NAMES
from src.utils.utils import build_datamodule,build_model,merge_train_config
SCALES=(0.0,0.02,0.05,0.10,0.20,0.30,0.50,1.00)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'configs/train_covariance_csp_short8.yaml'); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,default=ROOT/'logs/csp_residual_sweep.csv'); a=p.parse_args()
 c=merge_train_config(a.config); L.seed_everything(int(c.get('seed',42)),workers=True); dm=build_datamodule(c); dm.setup('fit'); interface=build_model(c).cuda().eval(); state=torch.load(a.checkpoint,map_location='cuda',weights_only=False)['state_dict']; interface.load_state_dict(state)
 sums=defaultdict(lambda:defaultdict(float)); counts=defaultdict(int)
 with torch.no_grad():
  for batch in dm.val_dataloader():
   x,y,index=batch[:3]; task=None
   for metadata in batch[3:]:
    value=torch.as_tensor(metadata)
    if value.numel()==1 and not value.is_floating_point(): task=RESPONSE_TASK_NAMES[int(value)]
   logits,aux=interface.model(x,y,index,return_auxiliary=True); relation=aux['covariance_relation_logits']; base=logits-0.02*relation; targets=y[index]
   for scale in SCALES:
    fused=base+scale*relation; diagnostics=interface._binary_query_diagnostics(fused,targets); key=(scale,task)
    sums[key]['ce']+=float(F.cross_entropy(fused,targets))*index.numel(); sums[key]['auroc']+=float(diagnostics['auroc'])*index.numel(); sums[key]['balanced_accuracy']+=float(diagnostics['balanced_accuracy'])*index.numel(); counts[key]+=index.numel()
 rows=[]
 for scale in SCALES:
  for task in RESPONSE_TASK_NAMES:
   key=(scale,task)
   if counts[key]: rows.append({'scale':scale,'task':task,'queries':counts[key],**{name:sums[key][name]/counts[key] for name in ('ce','auroc','balanced_accuracy')}})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print(json.dumps([r for r in rows if r['task']=='covariance'],indent=2)); print(f'saved={a.output}')
if __name__=='__main__': main()
