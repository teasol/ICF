from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import lightning as L
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.utils.utils import build_datamodule,build_model,merge_train_config
from scripts.diagnose_covariance_relations import episode_metrics
from scripts.diagnose_oracle_slot_alignment import query_index

RANKS=(1,2,4,8)
RELATIONS=("prototype_cosine","standardized_distance","multiscale_rbf")
METRICS=("covariance_relation_auroc","covariance_relation_balanced_accuracy",
         "covariance_relation_ce","covariance_relation_logit_std",
         "covariance_relation_class_separation")

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=ROOT/"configs/train_covariance_relation_e0.yaml"); p.add_argument("--output",type=Path,default=ROOT/"logs/covariance_subspace.csv"); a=p.parse_args()
 c=merge_train_config(a.config); L.seed_everything(int(c.get("seed",42)),workers=True)
 dm=build_datamodule(c); dm.setup("fit"); model=build_model(c).model.to("cuda").eval(); agg=model.aggregator; clf=model.meta_classifier
 sums=defaultdict(lambda:defaultdict(float)); counts=defaultdict(int); episodes=defaultdict(int)
 projection=agg._covariance_projection[:,:32].float()
 with torch.no_grad():
  for index in range(len(dm.val_dataset)):
   episode=dm.val_dataset.diagnostic_episode(index)
   if episode.response_task!="covariance": continue
   x,y=episode.x,episode.y; query=query_index(y); context=torch.ones(y.numel(),dtype=torch.bool,device=y.device); context[query]=False
   delta=x.float()-x.float().mean(dim=1,keepdim=True); projected=delta@projection
   covariance=torch.einsum("bni,bnj->bij",projected,projected)/projected.shape[1]
   for whiten in (False,True):
    family="csp" if whiten else "raw"
    for rank in RANKS:
     context_feature,query_feature,eigenvalues=clf._covariance_subspace_features(covariance[context],y[context],covariance[query],rank=rank,whiten=whiten)
     for relation in RELATIONS:
      clf.covariance_relation_mode=relation
      logits,separation=clf._covariance_relation_scores(context_feature,y[context],query_feature)
      metrics,valid=episode_metrics(logits,y[query],separation); key=f"{family}_rank{rank}_{relation}"
      for metric,value in metrics.items(): sums[key][metric]+=value*query.numel()
      sums[key]["selected_eigenvalue_abs"]+=float(eigenvalues.abs().mean())*query.numel(); counts[key]+=query.numel(); episodes[key]+=1
 rows=[]
 for family in ("raw","csp"):
  for rank in RANKS:
   for relation in RELATIONS:
    key=f"{family}_rank{rank}_{relation}"; row={"candidate":key,"family":family,"rank":rank,"relation":relation,"episodes":episodes[key],"queries":counts[key]}; row.update({f"val/{metric}":sums[key][metric]/counts[key] for metric in METRICS}); row["selected_eigenvalue_abs"]=sums[key]["selected_eigenvalue_abs"]/counts[key]; rows.append(row)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("w",newline="") as h:
  w=csv.DictWriter(h,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print(json.dumps(sorted(rows,key=lambda row:row["val/covariance_relation_auroc"],reverse=True),indent=2)); print(f"saved={a.output}")
if __name__=="__main__": main()
