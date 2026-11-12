#!/usr/bin/env python3
"""Audit extracted BC NPZ shards and emit reproducible quality artifacts."""
import argparse, glob, json, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def audit(root):
    files=sorted(glob.glob(str(Path(root)/"*.npz")))
    agg=Counter(); decks=Counter(); outcomes=Counter(); selects=Counter(); episodes=defaultdict(lambda: {"n":0,"outcomes":set(),"decks":Counter(),"forced":0,"multi":0})
    violations=Counter(); files_bad=[]
    for fp in files:
        z=np.load(fp); n=len(z["outcome"]); agg["pairs"]+=n
        for k,v in zip(*np.unique(z["deck_id"],return_counts=True)): decks[str(int(k))]+=int(v)
        for k,v in zip(*np.unique(z["outcome"],return_counts=True)): outcomes[str(float(k))]+=int(v)
        for k,v in zip(*np.unique(z["select_type"],return_counts=True)): selects[str(int(k))]+=int(v)
        chosen=(z["chosen"]>0).sum(1); legal=(z["legal_mask"]>0).sum(1)
        bad=chosen>z["max_count"]
        if bad.any(): files_bad.append({"file":Path(fp).name,"violations":int(bad.sum())})
        for st,v in zip(*np.unique(z["select_type"][bad],return_counts=True)): violations[str(int(st))]+=int(v)
        for eid,o,d,m,l in zip(z["episode_id"],z["outcome"],z["deck_id"],z["max_count"],legal):
            e=episodes[int(eid)]; e["n"]+=1; e["outcomes"].add(float(o)); e["decks"][str(int(d))]+=1
            e["forced"]+=int(l==1); e["multi"]+=int(m>1)
    ep_rows=[]
    for eid,e in episodes.items():
        dominant=max(e["decks"],key=e["decks"].get)
        ep_rows.append({"episode_id":eid,"pairs":e["n"],"dominant_deck":int(dominant),"has_both_outcomes":len(e["outcomes"])==2,"forced_fraction":e["forced"]/e["n"],"multi_fraction":e["multi"]/e["n"]})
    ep_rows.sort(key=lambda x:x["episode_id"])
    agg.update({"shards":len(files),"episodes":len(ep_rows),"invalid_outcomes":sum(v for k,v in outcomes.items() if k not in ("-1.0","1.0")),"action_count_violations":sum(violations.values()),"episodes_with_both_outcomes":sum(x["has_both_outcomes"] for x in ep_rows),"episodes_single_outcome":sum(not x["has_both_outcomes"] for x in ep_rows)})
    return {"source_root":str(Path(root).resolve()),"files":files,"aggregate":dict(agg),"deck_pairs":dict(decks),"outcomes":dict(outcomes),"select_types":dict(selects),"action_count_violations_by_select_type":dict(violations),"bad_files":files_bad,"episodes":ep_rows}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out",default="experiments/data_quality_report.json"); args=ap.parse_args()
    report=audit(args.root); Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(report,indent=2)+"\n")
    a=report["aggregate"]; print(json.dumps(a,indent=2))
if __name__=="__main__": main()
