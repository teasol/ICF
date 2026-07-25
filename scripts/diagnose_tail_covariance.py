from __future__ import annotations
import argparse,csv,json,math,sys
from collections import defaultdict
from pathlib import Path
import lightning as L
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.utils.utils import build_datamodule,build_model,merge_train_config
from scripts.diagnose_covariance_relations import episode_metrics
from scripts.diagnose_oracle_slot_alignment import query_index

FRACTIONS=(0.01,0.05,0.15)
RELATIONS=("prototype_cosine","standardized_distance","multiscale_rbf")
METRICS=("covariance_relation_auroc","covariance_relation_balanced_accuracy",
         "covariance_relation_ce","covariance_relation_logit_std",
         "covariance_relation_class_separation")

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=ROOT/"configs/train_covariance_relation_e0.yaml"); p.add_argument("--output",type=Path,default=ROOT/"logs/tail_covariance_diagnostics.csv"); a=p.parse_args()
 c=merge_train_config(a.config); L.seed_everything(int(c.get("seed",42)),workers=True)
 dm=build_datamodule(c); dm.setup("fit"); model=build_model(c).model.to("cuda").eval(); agg=model.aggregator; clf=model.meta_classifier
 sums=defaultdict(lambda:defaultdict(float)); counts=defaultdict(int); episodes=defaultdict(int); enrichment=defaultdict(lambda:defaultdict(float)); bag_counts=defaultdict(int)
 with torch.no_grad():
  for index in range(len(dm.val_dataset)):
   episode=dm.val_dataset.diagnostic_episode(index)
   if episode.response_task!="covariance": continue
   x,y,responsive=episode.x,episode.y,episode.responsive_instance_mask
   query=query_index(y); context=torch.ones(y.numel(),dtype=torch.bool,device=y.device); context[query]=False
   classification,_,_=agg._bag_view(x); anchors=agg._context_anchors(list(classification.unbind(0)),context)
   similarity=torch.einsum("bnd,sd->bns",F.normalize(classification.float(),dim=-1),anchors.float())
   novelty=1.0-similarity.max(dim=-1).values
   descriptors={}; per_fraction=[]
   base_fraction=responsive.float().mean(dim=-1)
   for fraction in FRACTIONS:
    count=max(agg.min_tail_instances,int(math.ceil(fraction*x.shape[1])))
    selected=novelty.topk(count,dim=-1).indices
    selected_mask=responsive.gather(1,selected).float()
    precision=selected_mask.mean(dim=-1); recall=selected_mask.sum(dim=-1)/responsive.sum(dim=-1).clamp_min(1)
    name=f"tail_{int(fraction*100):02d}"
    enrichment[name]["precision"]+=float(precision.mean())*y.numel(); enrichment[name]["recall"]+=float(recall.mean())*y.numel(); enrichment[name]["enrichment"]+=float((precision/base_fraction.clamp_min(1e-8)).mean())*y.numel(); bag_counts[name]+=y.numel()
    bag_features=[]
    for bag,indices in zip(x,selected):
     subset=bag[indices]; bag_features.append(agg._covariance_sketch(subset-subset.mean(dim=0,keepdim=True)))
    feature=torch.stack(bag_features); descriptors[name]=feature; per_fraction.append(feature)
   descriptors["tail_concat"]=torch.cat(per_fraction,dim=-1)
   for descriptor,values in descriptors.items():
    for relation in RELATIONS:
     clf.covariance_relation_mode=relation; logits,separation=clf._covariance_relation_scores(values[context],y[context],values[query]); metrics,valid=episode_metrics(logits,y[query],separation); key=f"{descriptor}_{relation}"
     for metric,value in metrics.items(): sums[key][metric]+=value*query.numel()
     counts[key]+=query.numel(); episodes[key]+=1
 rows=[]
 for descriptor in ("tail_01","tail_05","tail_15","tail_concat"):
  for relation in RELATIONS:
   key=f"{descriptor}_{relation}"; row={"candidate":key,"descriptor":descriptor,"relation":relation,"episodes":episodes[key],"queries":counts[key]}; row.update({f"val/{metric}":sums[key][metric]/counts[key] for metric in METRICS})
   if descriptor!="tail_concat": row.update({f"oracle_{metric}":enrichment[descriptor][metric]/bag_counts[descriptor] for metric in ("precision","recall","enrichment")})
   rows.append(row)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("w",newline="") as h:
  fields=sorted({key for row in rows for key in row}); w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
 print(json.dumps(rows,indent=2)); print(f"saved={a.output}")
if __name__=="__main__": main()
