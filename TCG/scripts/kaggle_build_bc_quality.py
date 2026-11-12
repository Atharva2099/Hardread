#!/usr/bin/env python3
"""Kaggle-only BC-quality dataset builder; never writes raw data locally."""
from __future__ import annotations
import argparse, collections, glob, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
from src.replay import parse_episode, _collect_player_card_ids, match_deck, featurize_pair, _flush_shard
from src.featurizer import MAX_OPTIONS

def reward(x): return float(x) if isinstance(x,(int,float,np.integer,np.floating)) else 0.0
def wilson_lb(w,n,z=1.96):
    if n<=0:return 0.0
    p=w/n; den=1+z*z/n
    return max(0.0,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/den)
def raw_files(root,max_files):
    out=[]
    for p in glob.glob(os.path.join(root,'**','*.json'),recursive=True):
        try:
            with open(p) as f:x=json.load(f)
            if isinstance(x,dict) and isinstance(x.get('steps'),list) and isinstance(x.get('info'),dict):out.append(p)
        except Exception:pass
        if max_files and len(out)>=max_files:break
    return sorted(out)
def scan(files):
    agents=collections.defaultdict(collections.Counter); eps=[]; seen=set(); duplicates=0
    for fp in files:
        e=parse_episode(fp)
        if not e:continue
        info=e.get('info') or {}; eid=info.get('EpisodeId')
        if eid in seen:duplicates+=1;continue
        seen.add(eid); names=(info.get('TeamNames') or [])[:2]; rs=e.get('rewards') or [0,0];r0,r1=reward(rs[0] if rs else 0),reward(rs[1] if len(rs)>1 else 0)
        winner=0 if r0>r1 else 1 if r1>r0 else -1
        statuses=(e.get('statuses') or [])[:2]
        for p,n in enumerate(names):
            if not n:continue
            if winner<0:agents[n]['draw']+=1
            elif p==winner:agents[n]['win']+=1
            else:agents[n]['loss']+=1
        eps.append((fp,eid,names,winner,statuses))
    stats={n:{'wins':s['win'],'losses':s['loss'],'draws':s['draw'],'games':s['win']+s['loss'],'wilson_lcb':wilson_lb(s['win'],s['win']+s['loss'])} for n,s in agents.items()}
    eligible=[n for n,x in stats.items() if x['games']>=30]
    eligible.sort(key=lambda n:(stats[n]['wilson_lcb'],stats[n]['games']),reverse=True)
    top=set(eligible[:max(1,int(.30*len(eligible)))])
    return eps,stats,top,duplicates
def valid_entries(ep, allowed, stats, variant, min_decisions=8, cap=96):
    info=ep.get('info') or {}; names=(info.get('TeamNames') or [])[:2];rs=ep.get('rewards') or [0,0];r0,r1=reward(rs[0] if rs else 0),reward(rs[1] if len(rs)>1 else 0)
    winner=0 if r0>r1 else 1 if r1>r0 else -1
    statuses=(ep.get('statuses') or [])[:2]
    if winner<0 or len(statuses)<2 or any(s!='DONE' for s in statuses):return [],{'terminal_or_draw':1}
    did={p:match_deck(_collect_player_card_ids(ep.get('steps') or [],p)) for p in(0,1)}
    candidates=[[],[]];rej=collections.Counter()
    for step in ep.get('steps') or []:
      if not isinstance(step,list):continue
      for p,e in enumerate(step[:2]):
        if not isinstance(e,dict) or (variant=='top' and (p>=len(names) or names[p] not in allowed)):continue
        obs=e.get('observation');a=e.get('action');sel=(obs or {}).get('select') if isinstance(obs,dict) else None
        if not isinstance(sel,dict) or not sel.get('option') or not isinstance(a,list) or not a:continue
        opts=sel.get('option') or []; mc=int(sel.get('maxCount',0));mn=int(sel.get('minCount',0))
        if int(sel.get('type',-1))==9 or len(a)>=30:rej['deck_selection']+=1;continue
        if any(not isinstance(i,(int,np.integer)) or not 0<=int(i)<len(opts) for i in a):rej['illegal_index']+=1;continue
        if len(set(map(int,a)))!=len(a):rej['duplicate_action']+=1;continue
        if not mn<=len(a)<=mc:rej['count_violation']+=1;continue
        candidates[p].append((obs,[int(i) for i in a],winner,did[p],len(opts)==1))
    if sum(map(len,candidates))<min_decisions:return [],rej|{'too_short':1}
    out=[]
    for p,rows in enumerate(candidates):
      # deterministic even subsample to prevent a long episode dominating training.
      if len(rows)>cap: rows=[rows[i] for i in np.linspace(0,len(rows)-1,cap,dtype=int)];rej['episode_cap']+=1
      n=names[p] if p<len(names) else ''; st=stats.get(n,{}); strength=float(st.get('wilson_lcb',0.0)); agent_w=max(.20,min(1.0,.30+1.4*strength))
      for obs,a,w,d,forced in rows:
        outcome=1.0 if p==w else -1.0
        rec=featurize_pair(obs,a,deck_id=d,outcome=outcome,episode_id=int(info.get('EpisodeId') or 0))
        if rec is None:rej['featurize']+=1;continue
        # `legal_mask` deliberately uses -inf for padded slots; audit only
        # numeric model inputs that must be finite.
        finite_keys=('board','options','card_ids','option_card','max_count','min_count','select_type','select_ctx','deck_id','chosen','outcome','episode_id')
        if not all(np.isfinite(rec[k]).all() for k in finite_keys):rej['nonfinite']+=1;continue
        order=np.full(MAX_OPTIONS,-1,dtype=np.int32);order[:len(a)]=a
        rec['chosen_order']=order;rec['action_len']=np.int32(len(a));rec['sample_weight']=np.float32(agent_w*(1.0 if outcome>0 else .35)*(.05 if forced else 1.0)*(.5 if d==3 else 1.0));rec['agent_strength']=np.float32(strength);rec['forced_action']=np.int8(forced)
        out.append(rec)
    return out,rej
def digest(r):
    h=hashlib.sha1()
    for k in ('board','card_ids','options','option_card','legal_mask','chosen_order'):h.update(np.ascontiguousarray(r[k]).tobytes())
    return h.digest()
def build(eps,stats,top,out_dir,variant,limit):
    Path(out_dir).mkdir(parents=True,exist_ok=True); shard=[];seen=set(); q=collections.Counter(); deck=collections.Counter(); total=0;idx=0;used=0
    for fp,eid,names,winner,statuses in eps:
      ep=parse_episode(fp)
      rows,rej=valid_entries(ep,top,stats,variant);q.update(rej)
      for r in rows:
        d=digest(r)
        if d in seen:q['exact_duplicate']+=1;continue
        seen.add(d);shard.append(r);total+=1;deck[int(r['deck_id'])]+=1
        if len(shard)>=5000:_flush_shard(shard,out_dir,idx);idx+=1;shard=[]
        if limit and total>=limit:break
      if rows:used+=1
      if limit and total>=limit:break
    if shard:_flush_shard(shard,out_dir,idx);idx+=1
    return {'pairs':total,'shards':idx,'episodes':used,'deck_pairs':dict(deck),'rejections':dict(q)}
def integrity(root):
 n=bad=0
 for f in glob.glob(os.path.join(root,'shard_*.npz')):
  z=np.load(f);n+=len(z['chosen']);bad+=int(((z['chosen']>0).sum(1)>z['max_count']).sum());bad+=int(((z['chosen_order']>=0).sum(1)!=z['action_len']).sum())
 return {'pairs':n,'violations':bad}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-root',default='/kaggle/input');ap.add_argument('--out',default='/kaggle/working/bc_quality_v1');ap.add_argument('--max-files',type=int,default=5000);ap.add_argument('--max-pairs',type=int,default=250000);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 files=raw_files(a.input_root,a.max_files);eps,stats,top,dups=scan(files)
 (out/'agent_quality.json').write_text(json.dumps({'agents':stats,'top_agents':sorted(top),'duplicate_episode_files':dups},indent=2))
 result={'source_files':len(files),'unique_episodes':len(eps),'weighted_all':build(eps,stats,top,str(out/'weighted_all'),'all',a.max_pairs),'weighted_top':build(eps,stats,top,str(out/'weighted_top'),'top',a.max_pairs)}
 result['integrity']={'weighted_all':integrity(str(out/'weighted_all')),'weighted_top':integrity(str(out/'weighted_top'))};(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
