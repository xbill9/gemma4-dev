"""Mean per-step kernel time by kernel family, and each attention call's growth with depth. Prints ANALYSIS."""
import collections, json, statistics as st
def load(m):
    r=collections.defaultdict(list)
    for l in open(f'profile-{m}.jsonl'):
        d=json.loads(l)
        if d['step']=='profile': r[d['depth']].append(d)
    return r
def fam(n):
    for k in ('_dec_split','_dec_combine','_w4_gemv'):
        if k in n: return k
    return 'other'
res={}
for m in ('2b','4b'):
    r=load(m); res[m]={}
    for dep,rows in sorted(r.items()):
        f=collections.Counter(); n=collections.Counter()
        for d in rows:
            for name,ms in d['calls']:
                f[fam(name)]+=ms/len(rows); n[fam(name)]+=1/len(rows)
        wall=st.mean(d['graph_ms_wall'] for d in rows)
        rate=st.mean(d['decode_tok_per_s'] for d in rows)
        print(m,dep,'n',len(rows),'calls',len(rows[0]['calls']),'wall',round(wall,2),'tok/s',round(rate,2),'sum',round(sum(f.values()),2),{k:(round(v,3),round(n[k])) for k,v in f.items()})
        res[m][dep]=(f,n,wall)
    # per-call dec_split, average over passes
    def per(dep):
        rows=r[dep]; out=[]
        for i in range(len(rows[0]['calls'])):
            if '_dec_split' in rows[0]['calls'][i][0]:
                out.append(st.mean(d['calls'][i][1] for d in rows))
        return out
    a,b=per(0),per(32768)
    g=[round(y-x,3) for x,y in zip(a,b)]
    print(m,'dec_split calls',len(a))
    print(' short',[round(x,3) for x in a])
    print(' 32k  ',[round(x,3) for x in b])
    grew=[(i,x) for i,x in enumerate(g) if x>0.05]
    print(' grew >0.05ms',len(grew),'sum',round(sum(x for _,x in grew),3),grew,' short-time of those',round(sum(a[i] for i,_ in grew),3), ' rest at 32k', round(sum(b)-sum(b[i] for i,_ in grew),3))
    print(' 32k ms of grown calls',[round(b[i],3) for i,_ in grew])
    print(' pass spread 32k wall',[d['graph_ms_wall'] for d in r[32768]])
