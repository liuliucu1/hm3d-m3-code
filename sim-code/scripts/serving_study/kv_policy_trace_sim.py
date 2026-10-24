"""Historical trace-driven KV replay for the 2026-10-24 exploratory table.

This version infers episode completion from the final request in the complete
trace, which leaks future information into ``term_lru`` and ``phys``. Its
``phys`` policy uses an inter-call EMA, not physical progress. Keep it only to
reproduce the historical table; use kv_progress_replay.py for causal policy
comparisons and evaluate_progress.py for physical-signal prediction quality.

Replays the vLLM block-hash trace captured by install_vllm_cache_probe.py under a
chosen HBM capacity (in 16-token blocks) and compares eviction policies:
  lru       vLLM-like LRU, tail blocks of a request evicted first
  term_lru  LRU, but blocks of robots that already finished are evicted first
  ring      evict the most recently served session first (loop-aware)
  phys      evict the session with the farthest predicted next call
            (EMA of inter-call gap) + finished-robot signal
  opt       block-level Belady with tail-first tie-break (clairvoyant reference)
Hits count only the contiguous resident prefix, as in vLLM APC. No timing
feedback and no in-flight pinning are modeled.

Usage: python3 kv_policy_trace_sim.py <analysis_dir> 3146,2200,1400,1000 lru,ring,phys,opt
"""
import json,sys,collections,heapq
D=sys.argv[1]
E=[json.loads(l) for l in open(D+'/kv_probe_robothor_k8.jsonl')]
R={}
for l in open(D+'/robothor_k8_kvprobe_replay.jsonl'):
    r=json.loads(l); R[r['request_id']]=r
stored=collections.defaultdict(list)
for e in E:
    if e['kind']=='stored': stored[e['request_id']].append(e['hash'])
reqs=[]
for e in E:
    if e['kind']!='lookup': continue
    rid=e['request_id']; cid=rid[len('chatcmpl-'):].rsplit('-',1)[0]
    r=R[cid]
    hs=list(e['hashes']); extra=[h for h in stored[rid] if h not in set(hs)]
    reqs.append(dict(t=e['time_ns']/1e9,look=hs,all=hs+extra,sess=(r['episode_id'],r['role']),role=r['role'],ep=r['episode_id'],pt=r['prompt_tokens'],step=r['step']))
t0=reqs[0]['t']
for q in reqs: q['t']-=t0
# next-use index for OPT
INF=float('inf')
nxt=[{} for _ in reqs]; last={}
for i in range(len(reqs)-1,-1,-1):
    for h in reqs[i]['all']: nxt[i][h]=last.get(h,INF)
    for h in reqs[i]['look']: last[h]=i
# session end (last call index of each session) -> "termination signal"
sess_last={}
for i,q in enumerate(reqs): sess_last[q['sess']]=i
ep_last={}
for i,q in enumerate(reqs): ep_last[q['ep']]=i
def run(C,policy):
    res={}   # hash -> key (smaller = evict first)
    owner={}
    miss=collections.Counter(); tot=collections.Counter(); ev_live=0
    sess_t={}; sess_gap={}
    for i,q in enumerate(reqs):
        hit=0
        for h in q['look']:
            if h in res: hit+=1
            else: break
        tot[q['role']]+=len(q['look']); miss[q['role']]+=len(q['look'])-hit
        s=q['sess']
        if s in sess_t:
            g=q['t']-sess_t[s]; sess_gap[s]=g if s not in sess_gap else 0.5*sess_gap[s]+0.5*g
        sess_t[s]=q['t']
        for pos,h in enumerate(q['all']):
            owner[h]=s
            if policy=='lru': key=(i,-pos)
            elif policy=='opt': key=(-nxt[i][h],-pos)
            else: key=None
            res[h]=key
        for pos,h in enumerate(q['all']): res_lru[h]=(i,-pos)
        while len(res)>C:
            if policy in ('lru','opt'):
                v=min(res,key=lambda h:res[h])
            else:
                def score(h):
                    s2=owner[h]; ep=s2[0]
                    done = (ep_last[ep] < i) if policy!='phys_noterm' else False  # robot finished -> dead
                    if done: return (0,0)
                    if policy=='term_lru': return (1,res_lru[h])

                    if policy=='ring': return (1,-sess_t[s2],res_lru[h][1])
                    pred=sess_t[s2]+sess_gap.get(s2,5.0)   # predicted next use time
                    if pred<q['t']: pred=q['t']+(q['t']-pred)*10   # overdue session: likely idle
                    return (1,-pred,res_lru[h][1])
                v=min(res,key=score)
            if nxt[i].get(v,INF)!=INF and v not in q['all']: ev_live+=1
            del res[v]
    return {k:(miss[k],tot[k]) for k in tot}
res_lru={}
caps=[int(x) for x in sys.argv[2].split(',')]
print('blocks unique',len({h for q in reqs for h in q['all']}),'lookup blocks',sum(len(q['look']) for q in reqs))
for C in caps:
    out=[]
    for p in sys.argv[3].split(','):
        res_lru.clear()
        r=run(C,p)
        pm,pt=r['planner_shadow']; rm,rt=r['target_review']
        out.append(f"{p}: plan miss {pm}/{pt} rev miss {rm}/{rt}")
    print(C,' | '.join(out),flush=True)
