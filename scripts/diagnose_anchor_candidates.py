from __future__ import annotations

import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import lightning as L
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.utils.utils import build_datamodule,build_model,merge_train_config
from scripts.diagnose_oracle_slot_alignment import query_index

CANDIDATES=("hybrid12","kmeans12","kmeans8","kmeans6")
METRICS=("soft_purity","soft_capture","soft_entropy","soft_query_agreement",
         "hard_purity","hard_capture","hard_entropy","hard_query_agreement")

def alignment(assignment,responsive,context,queries):
    responsive_mass=(assignment*responsive.unsqueeze(-1)).sum(dim=1)
    slot_mass=assignment.sum(dim=1).clamp_min(1e-8)
    purity=responsive_mass/slot_mass
    distribution=responsive_mass/responsive_mass.sum(dim=-1,keepdim=True).clamp_min(1e-8)
    best=responsive_mass.argmax(dim=-1)
    best_purity=purity.gather(1,best[:,None]).squeeze(1).mean()
    capture=distribution.max(dim=-1).values.mean()
    entropy=-(distribution.clamp_min(1e-12)*distribution.clamp_min(1e-12).log()).sum(dim=-1)
    entropy=entropy/torch.log(torch.tensor(assignment.shape[-1],device=assignment.device,dtype=torch.float32))
    context_best=best[context]
    modal=torch.bincount(context_best,minlength=assignment.shape[-1]).argmax()
    agreement=(best[queries]==modal).float().mean()
    return best_purity,capture,entropy.mean(),agreement

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/train_covariance_relation_e0.yaml")
    parser.add_argument("--output",type=Path,default=ROOT/"logs/oracle_anchor_candidates.csv")
    args=parser.parse_args(); config=merge_train_config(args.config)
    L.seed_everything(int(config.get("seed",42)),workers=True)
    dm=build_datamodule(config); dm.setup("fit")
    aggregator=build_model(config).model.aggregator.to("cuda").eval()
    sums=defaultdict(lambda:defaultdict(float)); counts=defaultdict(int); episodes=defaultdict(int)
    with torch.no_grad():
      for index in range(len(dm.val_dataset)):
        episode=dm.val_dataset.diagnostic_episode(index); responsive=episode.responsive_instance_mask
        if responsive is None: continue
        x,y=episode.x,episode.y; queries=query_index(y); context=torch.ones(y.numel(),dtype=torch.bool,device=y.device); context[queries]=False
        classification,_,_=aggregator._bag_view(x); bags=list(classification.unbind(0))
        anchors_by_name={"hybrid12":aggregator._context_anchors(bags,context)}
        for slots in (12,8,6): anchors_by_name[f"kmeans{slots}"]=aggregator._context_spherical_kmeans_anchors(bags,context,slots)
        for name,anchors in anchors_by_name.items():
          similarity=torch.einsum("bnd,sd->bns",F.normalize(classification.float(),dim=-1),anchors.float())
          soft=torch.softmax(similarity/aggregator.assignment_temperature,dim=-1)
          hard=F.one_hot(similarity.argmax(dim=-1),num_classes=anchors.shape[0]).float()
          soft_values=alignment(soft,responsive.float(),context,queries); hard_values=alignment(hard,responsive.float(),context,queries)
          values=dict(zip(METRICS,(*soft_values,*hard_values)))
          for task in ("all",episode.response_task):
            for metric,value in values.items(): sums[(name,task)][metric]+=float(value)*y.numel()
            counts[(name,task)]+=y.numel(); episodes[(name,task)]+=1
    rows=[]
    for name in CANDIDATES:
      for task in ("all","covariance"):
        key=(name,task)
        row={"candidate":name,"task":task,"episodes":episodes[key],"bags":counts[key]}
        row.update({metric:sums[key][metric]/counts[key] for metric in METRICS}); rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as handle:
      writer=csv.DictWriter(handle,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows,indent=2)); print(f"saved={args.output}")
if __name__=="__main__": main()
