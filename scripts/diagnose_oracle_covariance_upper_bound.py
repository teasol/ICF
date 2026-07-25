from __future__ import annotations
import argparse,csv,json,sys
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

DESCRIPTORS=("observed_covariance","observed_spectral","observed_local_distance",
             "observed_local_anisotropy","observed_oracle_population","latent_dispersion")
RELATIONS=("prototype_cosine","standardized_distance","multiscale_rbf")
METRICS=("covariance_relation_auroc","covariance_relation_balanced_accuracy",
         "covariance_relation_ce","covariance_relation_logit_std",
         "covariance_relation_class_separation")

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=ROOT/"configs/train_covariance_relation_e0.yaml"); p.add_argument("--output",type=Path,default=ROOT/"logs/oracle_covariance_upper_bound.csv"); a=p.parse_args()
 c=merge_train_config(a.config); L.seed_everything(int(c.get("seed",42)),workers=True)
 dm=build_datamodule(c); dm.setup("fit"); model=build_model(c).model.to("cuda").eval(); agg=model.aggregator; clf=model.meta_classifier
 sums=defaultdict(lambda:defaultdict(float)); query_counts=defaultdict(int); episodes=defaultdict(int)
 with torch.no_grad():
  for index in range(len(dm.val_dataset)):
   episode=dm.val_dataset.diagnostic_episode(index)
   if episode.response_task!="covariance": continue
   x,y,mask=episode.x,episode.y,episode.responsive_instance_mask
   if mask is None or episode.response_dispersion_factor is None: continue
   covariance=[]; spectral=[]; local_distance=[]; local_anisotropy=[]
   original_descriptor=agg.slot_covariance_descriptor
   for bag,bag_mask in zip(x,mask):
    selected=bag[bag_mask]
    delta=selected-selected.mean(dim=0,keepdim=True)
    covariance.append(agg._covariance_sketch(delta))
    normalized=F.normalize(delta.float(),dim=-1,eps=1e-6).to(bag.dtype)
    assignment=torch.ones(1,selected.shape[0],1,device=x.device,dtype=x.dtype)
    agg.slot_covariance_descriptor="spectral"
    feature,_=agg._slot_covariance_sketch(assignment,delta.unsqueeze(0))
    spectral.append(feature.squeeze(0).squeeze(0))
    geometry=agg._local_geometry_sketch(normalized.unsqueeze(0),neighbor_counts=(2,4,8))
    local_distance.append(geometry["distance"].squeeze(0)); local_anisotropy.append(geometry["anisotropy"].squeeze(0))
   agg.slot_covariance_descriptor=original_descriptor
   descriptors={
    "observed_covariance":torch.stack(covariance),
    "observed_spectral":torch.stack(spectral),
    "observed_local_distance":torch.stack(local_distance),
    "observed_local_anisotropy":torch.stack(local_anisotropy),
    "observed_oracle_population":episode.oracle_population_features,
    "latent_dispersion":episode.response_dispersion_factor.float().unsqueeze(-1),
   }
   query=query_index(y); context=torch.ones(y.numel(),dtype=torch.bool,device=y.device); context[query]=False
   for descriptor in DESCRIPTORS:
    values=descriptors[descriptor]
    for relation in RELATIONS:
     clf.covariance_relation_mode=relation
     logits,separation=clf._covariance_relation_scores(values[context],y[context],values[query])
     metrics,valid=episode_metrics(logits,y[query],separation); key=f"{descriptor}_{relation}"
     for name,value in metrics.items(): sums[key][name]+=value*query.numel()
     query_counts[key]+=query.numel(); episodes[key]+=1
 rows=[]
 for descriptor in DESCRIPTORS:
  for relation in RELATIONS:
   key=f"{descriptor}_{relation}"; row={"candidate":key,"descriptor":descriptor,"relation":relation,"episodes":episodes[key],"queries":query_counts[key]}
   row.update({f"val/{name}":sums[key][name]/query_counts[key] for name in METRICS}); rows.append(row)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("w",newline="") as h:
  w=csv.DictWriter(h,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print(json.dumps(rows,indent=2)); print(f"saved={a.output}")
if __name__=="__main__": main()
