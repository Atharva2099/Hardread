#!/usr/bin/env python3
"""Kaggle-only Stage 1/2 raw replay quality pipeline.

Input is attached under /kaggle/input; output must be /kaggle/working. It keeps
raw replay files out of the repo and preserves provenance in episode_manifest.
"""
from __future__ import annotations
import argparse, collections, glob, json, os, time
from pathlib import Path
from typing import Iterable
import numpy as np
from src.replay import parse_episode, featurize_pair, _flush_shard


def files_under(root: str, max_files: int = 0):
    out=[]
    for p in glob.glob(os.path.join(root, '**', '*.json'), recursive=True):
        try:
            with open(p) as f: x=json.load(f)
            if isinstance(x, dict) and isinstance(x.get('steps'), list) and isinstance(x.get('info'), dict): out.append(p)
        except Exception: pass
        if max_files and len(out) >= max_files: break
    return sorted(out)

def _reward_value(x):
    return float(x) if isinstance(x,(int,float,np.integer,np.floating)) else 0.0

def scan(files):
    rows=[]; stats=collections.Counter(); agent=collections.defaultdict(collections.Counter)
    for i,p in enumerate(files):
        ep=parse_episode(p)
        if ep is None: stats['parse_errors']+=1; continue
        info=ep.get('info') or {}; names=(info.get('TeamNames') or [])[:2]; rewards=ep.get('rewards') or [0,0]
        status=ep.get('statuses') or []
        winner=-1
        if len(rewards)>=2:
            r0, r1 = _reward_value(rewards[0]), _reward_value(rewards[1])
            winner=0 if r0>r1 else 1 if r1>r0 else -1
        for pi,name in enumerate(names):
            if not name: continue
            if winner<0: agent[name]['draws']+=1
            elif pi==winner: agent[name]['wins']+=1
            else: agent[name]['losses']+=1
        row={'episode_id':info.get('EpisodeId',i),'source_file':p,'source_dataset':next((x for x in p.split('/') if 'episodes-' in x),None),'team_names':names,'rewards':rewards[:2],'winner':winner,'statuses':status[:2],'n_steps':len(ep.get('steps') or []),'file_size':os.path.getsize(p)}
        rows.append(row); stats['episodes']+=1; stats[f'winner_{winner}']+=1
    return rows,agent,stats

def rates(agent):
    out={}
    for n,s in agent.items():
        games=s['wins']+s['losses']; out[n]={'wins':s['wins'],'losses':s['losses'],'draws':s['draws'],'games':games,'win_rate':s['wins']/games if games else 0.0}
    return out

def clean_pairs(ep, allowed=None, deck_match=True):
    """Extract only normal option-index actions; exclude deck selection and malformed labels."""
    info=ep.get("info") or {}; names=(info.get("TeamNames") or [])[:2]
    rewards=ep.get("rewards") or [0, 0]; winner=-1
    if len(rewards)>=2:
        r0, r1 = _reward_value(rewards[0]), _reward_value(rewards[1])
        winner=0 if r0>r1 else 1 if r1>r0 else -1
    players={i for i,n in enumerate(names) if allowed is None or n in allowed}
    if not players: return []
    if deck_match:
        # Reuse source parser's deck detector by extracting per-player visible IDs.
        from src.replay import _collect_player_card_ids, match_deck
        dids={i:match_deck(_collect_player_card_ids(ep.get("steps") or [],i)) for i in (0,1)}
    else: dids={0:0,1:0}
    out=[]
    for step in ep.get("steps") or []:
        if not isinstance(step,list): continue
        for pi,entry in enumerate(step[:2]):
            if pi not in players or not isinstance(entry,dict): continue
            obs=entry.get("observation"); action=entry.get("action")
            sel=(obs or {}).get("select") if isinstance(obs,dict) else None
            if not isinstance(sel,dict) or not sel.get("option") or not isinstance(action,list) or not action: continue
            # type 9 and long card-ID lists are deck-selection, not policy actions.
            if int(sel.get("type",-1))==9 or len(action)>=30: continue
            opts=sel.get("option") or []
            if any(not isinstance(i,(int,np.integer)) or not 0<=int(i)<len(opts) for i in action): continue
            if len(set(int(i) for i in action))>int(sel.get("maxCount",len(action))): continue
            if winner<0: outcome=0.0
            else: outcome=1.0 if pi==winner else -1.0
            out.append((obs,[int(i) for i in action],pi,outcome,int((info.get("EpisodeId") or 0)),dids[pi]))
    return out

def build_pairs(files, allowed, out_dir, deck_match=True, max_pairs=0, shard_size=5000):
    Path(out_dir).mkdir(parents=True,exist_ok=True); shard=[]; total=0; rejected=collections.Counter(); deck=collections.Counter(); by_ep=collections.Counter(); idx=0
    for fp in files:
        ep=parse_episode(fp)
        if ep is None: continue
        info=ep.get('info') or {}; names=(info.get('TeamNames') or [])[:2]; eid=info.get('EpisodeId',idx)
        # extract_pairs applies deck-selection and action-count validation.
        pairs=clean_pairs(ep,allowed=allowed,deck_match=deck_match)
        if pairs:
            by_ep[int(eid)]+=len(pairs)
        for obs,action,pidx,outcome,eid2,did in pairs:
            rec=featurize_pair(obs,action,deck_id=did,outcome=outcome,episode_id=eid2)
            if rec is None: rejected['featurize']+=1; continue
            shard.append(rec); deck[int(did)]+=1; total+=1
            if max_pairs and total>=max_pairs: break
            if len(shard)>=shard_size:
                _flush_shard(shard,out_dir,idx); idx+=1; shard=[]
        if max_pairs and total>=max_pairs: break
    if shard: _flush_shard(shard,out_dir,idx); idx+=1
    return {'pairs':total,'shards':idx,'deck_pairs':dict(deck),'episodes_with_pairs':len(by_ep),'rejected':dict(rejected)}

def audit_shards(out_dir):
    """Verify generated labels obey submission-policy action-count invariants."""
    files=sorted(glob.glob(os.path.join(out_dir, 'shard_*.npz')))
    n=viol=0
    for fp in files:
        z=np.load(fp); chosen=(z['chosen']>0).sum(axis=1); max_count=z['max_count']
        n += len(chosen); viol += int((chosen > max_count).sum())
    return {'pairs':n, 'shards':len(files), 'chosen_gt_max_count':viol}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-root',default='/kaggle/input'); ap.add_argument('--out',default='/kaggle/working/quality'); ap.add_argument('--top-pct',type=float,default=.3); ap.add_argument('--max-pairs',type=int,default=0); ap.add_argument('--max-files',type=int,default=0); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); files=files_under(args.input_root, args.max_files); print('raw files',len(files))
    rows,agent,scan_stats=scan(files); ar=rates(agent); ranked=sorted(ar,key=lambda n:(ar[n]['win_rate'],ar[n]['games']),reverse=True)
    min_games=10; eligible=[n for n in ranked if ar[n]['games']>=min_games]; cut=max(1,int(len(eligible)*args.top_pct)); top=set(eligible[:cut])
    for row in rows:
        row['agent_stats']=[ar.get(n,{}) for n in row['team_names']]
        row['quality_tier']='top_empirical_winrate' if any(n in top for n in row['team_names']) else 'all_valid'
    with open(out/'episode_manifest.jsonl','w') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
    summary={'source_root':os.path.abspath(args.input_root),'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'raw_files':len(files),'scan':dict(scan_stats),'agents':ar,'eligible_agents':len(eligible),'top_agents':sorted(top),'top_pct':args.top_pct}
    (out/'quality_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    variants={}
    variants['all_valid']=build_pairs(files,None,str(out/'pairs_all'),max_pairs=args.max_pairs)
    variants['top_empirical_winrate']=build_pairs(files,top,str(out/'pairs_top'),max_pairs=args.max_pairs)
    variants['integrity']={'all_valid':audit_shards(str(out/'pairs_all')), 'top_empirical_winrate':audit_shards(str(out/'pairs_top'))}
    (out/'variant_summary.json').write_text(json.dumps(variants,indent=2)+'\n')
    print(json.dumps({'summary':summary|{'variants':variants}},indent=2)[:12000])
if __name__=='__main__': main()
