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

DESCRIPTORS=("distance","anisotropy","combined")
RELATIONS=("prototype_cosine","standardized_distance","multiscale_rbf")
METRICS=("covariance_relation_auroc","covariance_relation_balanced_accuracy",
         "covariance_relation_ce","covariance_relation_logit_std",
         "covariance_relation_class_separation")

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=ROOT/"configs/train_covariance_relation_e0.yaml"); p.add_argument("--output",type=Path,default=ROOT/"logs/local_geometry_diagnostics.csv"); a=p.parse_args()
 c=merge_train_config(a.config); L.seed_everything(int(c.get("seed",42)),workers=True)
 dm=build_datamodule(c); dm.setup("fit"); model=build_model(c).model.to("cuda").eval(); agg=model.aggregator; clf=model.meta_classifier
 sums=defaultdict(lambda:defaultdict(float)); queries_count=defaultdict(int); valid_count=defaultdict(int); episodes=defaultdict(int)
 with torch.no_grad():
  for index in range(len(dm.val_dataset)):
   episode=dm.val_dataset.diagnostic_episode(index); x,y=episode.x,episode.y; query=query_index(y); context=torch.ones(y.numel(),dtype=torch.bool,device=y.device); context[query]=False
   classification,_,_=agg._bag_view(x); sketches=agg._local_geometry_sketch(classification)
   for descriptor in DESCRIPTORS:
    for relation in RELATIONS:
     clf.covariance_relation_mode=relation
     logits,separation=clf._covariance_relation_scores(sketches[descriptor][context],y[context],sketches[descriptor][query])
     metrics,valid=episode_metrics(logits,y[query],separation); candidate=f"{descriptor}_{relation}"
     for task in ("all",episode.response_task):
      key=(candidate,task)
      for name,value in metrics.items():
       if name in ("covariance_relation_auroc","covariance_relation_balanced_accuracy"):
        if valid: sums[key][name]+=value*query.numel()
       else: sums[key][name]+=value*query.numel()
      queries_count[key]+=query.numel(); valid_count[key]+=query.numel() if valid else 0; episodes[key]+=1
 rows=[]
 for descriptor in DESCRIPTORS:
  for relation in RELATIONS:
   candidate=f"{descriptor}_{relation}"
   for task in ("all","covariance"):
    key=(candidate,task); row={"candidate":candidate,"descriptor":descriptor,"relation":relation,"task":task,"episodes":episodes[key],"queries":queries_count[key]}
    for name in METRICS:
     denominator=valid_count[key] if name in ("covariance_relation_auroc","covariance_relation_balanced_accuracy") else queries_count[key]
     row[f"val/{name}"]=sums[key][name]/max(1,denominator)
    rows.append(row)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("w",newline="") as h:
  w=csv.DictWriter(h,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print(json.dumps([r for r in rows if r["task"]=="covariance"],indent=2)); print(f"saved={a.output}")
if __name__=="__main__": main()
