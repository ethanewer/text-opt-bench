"""Live campaign dashboard — localhost, dependency-free.

Reads runs/<task>/<prefix>*/log.jsonl on every request and serves a clean
single-page dashboard:

  - Task-quality map: headroom (x, log) vs inter-trial spread (y) scatter,
    each task a labeled point colored by an auto-computed quality tier —
    instantly separates strong/discriminating tasks from weak/near-one-shot
    ones. Plus a sortable quality table (baseline, best, headroom, spread,
    gradings-to-converge, tier).
  - Aggregate: every score scaled per task so baseline=1.0 and the best score
    reached=0.0, averaged across all tasks & trials, with error bars = std.
  - Per task: 5 best-so-far curves (one per trial, gradient colored) over a
    shaded min/max spread envelope, with accepted-improvement markers.
    Toggle x-axis iterations<->wall-clock (concurrency normalization),
    y-axis normalized<->raw, and raw y linear<->log.

"best-so-far" is monotone: a regression still shows the best grade reached.

Usage:  python3.12 tools/dashboard.py [--port 8420] [--prefix 5xB-]
"""

import argparse
import json
import math
import statistics as st
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LAUNCH = RUNS / "_campaign" / "launcher.jsonl"


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_trial(log_path):
    baseline = None
    best = None
    wall = 0.0
    pts = []
    for line in log_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == "baseline":
            # A resumed run logs a SECOND baseline event whose score is the
            # adopted best-so-far, not the true baseline — take only the first.
            if baseline is None:
                baseline = e.get("guide_score", e.get("score"))
                best = baseline
                pts.append({"i": 0, "best": baseline, "wall": 0.0, "acc": True})
            continue
        it = e.get("iter")
        if it is None:
            continue
        wall += e.get("seconds") or 0
        gs = e.get("guide_score")
        bgs = e.get("best_guide_score")
        improved = False
        if _num(bgs):
            if best is None or bgs < best:
                improved = bgs < (best if best is not None else bgs)
            best = bgs if best is None else min(best, bgs)
        elif e.get("ok") and _num(gs):
            if best is None or gs < best:
                improved = True
            best = gs if best is None else min(best, gs)
        pts.append({"i": it, "best": best, "wall": round(wall, 1),
                    "acc": bool(e.get("accepted"))})
    return {"run": log_path.parent.name, "baseline": baseline, "pts": pts}


def launcher_status():
    done = launched = 0
    total = None
    done_flag = False
    if LAUNCH.exists():
        for line in LAUNCH.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            ev = e.get("event")
            if ev == "launch":
                launched += 1
            elif ev in ("finish", "timeout"):
                done += 1
            elif ev == "campaign_start":
                d = e.get("detail", "")
                if " jobs" in d:
                    try:
                        total = int(d.split(" jobs")[0].split()[-1])
                    except ValueError:
                        pass
            elif ev == "campaign_done":
                done_flag = True
    return {"done": done, "running": max(0, launched - done),
            "total": total, "finished": done_flag}


def convergence_gradings(pts, final, eps=0.01):
    """First iteration index reaching within eps of the trial's final best."""
    if not _num(final):
        return None
    target = final * (1 + eps) if final > 0 else final * (1 - eps)
    for p in pts:
        if _num(p["best"]) and p["best"] <= target:
            return p["i"]
    return pts[-1]["i"] if pts else None


def tier_of(headroom, spread):
    if not headroom or headroom < 1.3:
        return "weak"
    if spread and spread >= 1.2:
        return "strong"
    if headroom >= 2:
        return "moderate"
    return "weak"


def build_data(prefix):
    tasks = {}
    for log in sorted(RUNS.glob(f"*/{prefix}*/log.jsonl")):
        task = log.parent.parent.name
        trial = load_trial(log)
        if trial["baseline"] is None and not trial["pts"]:
            continue
        tasks.setdefault(task, {"task": task, "trials": []})
        tasks[task]["trials"].append(trial)

    out_tasks = []
    agg = {}
    for task, td in sorted(tasks.items()):
        bases = [t["baseline"] for t in td["trials"] if _num(t["baseline"])]
        baseline = bases[0] if bases else None
        finals = [t["pts"][-1]["best"] for t in td["trials"]
                  if t["pts"] and _num(t["pts"][-1]["best"])]
        optimal = min(finals) if finals else None
        headroom = (baseline / optimal) if (_num(baseline) and optimal) else None
        spread = (max(finals) / min(finals)) if finals and min(finals) else None
        conv = [convergence_gradings(t["pts"], t["pts"][-1]["best"])
                for t in td["trials"] if t["pts"]]
        conv = [c for c in conv if c is not None]
        improved = (_num(baseline) and _num(optimal) and baseline > optimal)
        den = (baseline - optimal) if improved else 0.0
        for t in td["trials"]:
            for p in t["pts"]:
                if _num(p["best"]) and den > 0:
                    agg.setdefault(p["i"], []).append((p["best"] - optimal) / den)
        out_tasks.append({
            "task": task, "baseline": baseline, "optimal": optimal,
            "headroom": headroom, "spread": spread,
            "conv_median": (st.median(conv) if conv else None),
            "tier": tier_of(headroom, spread),
            "n_trials": len(td["trials"]),
            "max_iter": max((p["i"] for t in td["trials"] for p in t["pts"]), default=0),
            "trials": [{"run": t["run"], "pts": t["pts"]} for t in td["trials"]],
        })

    aggregate = [{"i": k, "mean": st.mean(v), "std": st.pstdev(v), "n": len(v)}
                 for k, v in sorted(agg.items()) if len(v) >= 2]

    return {"prefix": prefix, "status": launcher_status(),
            "tasks": out_tasks, "aggregate": aggregate}


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>text-opt-bm campaign</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1d212b;--line:#2a2f3a;--text:#e7eaf0;
  --muted:#8b93a7;--accent:#22d3ee;--grid:#232833;
  --strong:#34d399;--moderate:#fbbf24;--weak:#f87171;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
header{position:sticky;top:0;z-index:5;background:rgba(15,17,21,.92);backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line);padding:14px 24px}
.hrow{display:flex;align-items:center;gap:16px;flex-wrap:wrap;max-width:1500px;margin:0 auto}
h1{font-size:16px;font-weight:650;margin:0}h1 .p{color:var(--muted);font-weight:500}
.pills{display:flex;gap:8px;flex-wrap:wrap}
.pill{background:var(--panel2);border:1px solid var(--line);border-radius:999px;padding:4px 11px;font-size:12px;color:var(--muted)}
.pill b{color:var(--text)}.pill.run b{color:var(--accent)}.pill.done b{color:var(--strong)}
.spacer{flex:1}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.seg{display:inline-flex;background:var(--panel2);border:1px solid var(--line);border-radius:9px;overflow:hidden}
.seg button{background:transparent;border:0;color:var(--muted);padding:6px 12px;font-size:12.5px;cursor:pointer}
.seg button.on{background:var(--accent);color:#05222a;font-weight:650}
.updated{color:var(--muted);font-size:12px;min-width:92px;text-align:right}
main{max-width:1500px;margin:0 auto;padding:22px 24px 60px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 16px 10px;margin-bottom:20px}
.card h2{font-size:14px;margin:0 0 2px;font-weight:600}.card .sub{color:var(--muted);font-size:12px;margin:0 0 10px}
.two{display:grid;grid-template-columns:minmax(380px,1fr) minmax(360px,1fr);gap:20px}
@media(max-width:900px){.two{grid-template-columns:1fr}}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:18px}.grid .card{margin:0}
.stat{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:2px 0 6px}.stat b{color:var(--text)}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin-top:4px}
.legend .k{display:inline-flex;align-items:center;gap:5px}.legend .sw{width:14px;height:3px;border-radius:2px}
svg{display:block;width:100%;height:auto;overflow:visible}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600;cursor:pointer;user-select:none}th:first-child,td:first-child{text-align:left}
tr:hover td{background:var(--panel2)}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600}
.tag.strong{background:rgba(52,211,153,.16);color:var(--strong)}
.tag.moderate{background:rgba(251,191,36,.16);color:var(--moderate)}
.tag.weak{background:rgba(248,113,113,.16);color:var(--weak)}
.tip{position:fixed;pointer-events:none;background:#0b0d11;border:1px solid var(--line);border-radius:8px;
  padding:7px 9px;font-size:11.5px;color:var(--text);opacity:0;transition:opacity .08s;z-index:20;
  box-shadow:0 6px 20px rgba(0,0,0,.4);white-space:nowrap}
.empty{color:var(--muted);padding:40px;text-align:center}.axl{fill:var(--muted);font-size:10.5px}.gl{stroke:var(--grid);stroke-width:1}
</style></head>
<body>
<header><div class="hrow">
  <h1>text-opt-bm <span class="p" id="prefix"></span></h1>
  <div class="pills" id="pills"></div>
  <div class="spacer"></div>
  <div class="controls">
    <div class="seg" id="xseg"><button data-x="i" class="on">Iterations</button><button data-x="wall">Wall-clock</button></div>
    <div class="seg" id="yseg"><button data-y="norm" class="on">Normalized</button><button data-y="raw">Raw</button></div>
    <div class="seg" id="lseg"><button data-l="lin" class="on">Linear</button><button data-l="log">Log</button></div>
  </div>
  <div class="updated" id="updated">—</div>
</div></header>
<main>
  <div class="card">
    <h2>Task-quality map</h2>
    <p class="sub">Headroom (baseline&divide;best, log) vs inter-trial spread (max&divide;min final best). Upper-right = high headroom &amp; high variance = most discriminating. Tier auto-assigned.</p>
    <div class="two">
      <div id="scatter"></div>
      <div style="overflow:auto;max-height:340px"><table id="qtable"></table></div>
    </div>
  </div>
  <div class="card">
    <h2>Aggregate — normalized best-so-far, averaged across all tasks &amp; trials</h2>
    <p class="sub">Score scaled per task: baseline=1.0, best reached=0.0. Curve=mean; error bars=&plusmn;1 std across task-trials.</p>
    <div id="agg"></div>
  </div>
  <div class="grid" id="taskGrid"></div>
</main>
<div class="tip" id="tip"></div>
<script>
const S={x:'i',y:'norm',l:'lin',data:null,sort:'tier'};
const GRAD=[[34,211,238],[59,130,246],[139,92,246],[236,72,153]];
function grad(t){const s=GRAD,seg=(s.length-1)*t,i=Math.min(Math.floor(seg),s.length-2),f=seg-i,
  c=(a,b)=>Math.round(a+(b-a)*f);return `rgb(${c(s[i][0],s[i+1][0])},${c(s[i][1],s[i+1][1])},${c(s[i][2],s[i+1][2])})`;}
const TIER={strong:'#34d399',moderate:'#fbbf24',weak:'#f87171'};
function ticks(mn,mx,n){if(mn===mx)return[mn];const sp=mx-mn,s0=sp/n,mag=Math.pow(10,Math.floor(Math.log10(Math.abs(s0)||1))),
  nm=s0/mag;let st=nm<1.5?1:nm<3?2:nm<7?5:10;st*=mag;const out=[];for(let v=Math.ceil(mn/st)*st;v<=mx+1e-9;v+=st)out.push(v);return out;}
function fmt(v){if(v==null||isNaN(v))return'—';const a=Math.abs(v);
  if(a!==0&&(a<1e-3||a>=1e6))return v.toExponential(2);
  if(a>=1000)return v.toLocaleString(undefined,{maximumFractionDigits:0});
  if(a>=1)return(+v.toFixed(2)).toString();return(+v.toFixed(4)).toString();}
const NS='http://www.w3.org/2000/svg';
function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}

function chart(mount,{series,band,envelope,xLabel,yLabel,yFrom0,ylog}){
  const W=mount.clientWidth||440,H=270,m={l:56,r:14,t:12,b:34},iw=W-m.l-m.r,ih=H-m.t-m.b;
  const tf=ylog?(v=>v>0?Math.log10(v):null):(v=>v);
  const xs=[],ys=[];
  series.forEach(s=>s.pts.forEach(p=>{xs.push(p.x);const y=tf(p.y);if(y!=null)ys.push(y);}));
  if(band)band.forEach(p=>{xs.push(p.x);ys.push(p.mean+p.std);ys.push(Math.max(0,p.mean-p.std));});
  if(envelope)envelope.forEach(p=>{xs.push(p.x);[tf(p.lo),tf(p.hi)].forEach(y=>{if(y!=null)ys.push(y);});});
  if(!xs.length){mount.innerHTML='<div class="empty">no data yet</div>';return;}
  let xmn=Math.min(...xs),xmx=Math.max(...xs),ymn=Math.min(...ys),ymx=Math.max(...ys);
  if(yFrom0&&!ylog){ymn=Math.min(0,ymn);ymx=Math.max(ymx,1);}
  if(xmn===xmx)xmx=xmn+1;if(ymn===ymx)ymx=ymn+1;const pad=(ymx-ymn)*0.06;ymn-=pad;ymx+=pad;
  const X=v=>m.l+(v-xmn)/(xmx-xmn)*iw,Y=v=>m.t+(1-(v-ymn)/(ymx-ymn))*ih;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:'none'});
  ticks(ymn,ymx,5).forEach(t=>{const y=Y(t);svg.appendChild(el('line',{x1:m.l,y1:y,x2:W-m.r,y2:y,class:'gl'}));
    const tx=el('text',{x:m.l-7,y:y+3,'text-anchor':'end',class:'axl'});tx.textContent=ylog?fmt(Math.pow(10,t)):fmt(t);svg.appendChild(tx);});
  ticks(xmn,xmx,6).forEach(t=>{const x=X(t);svg.appendChild(el('line',{x1:x,y1:m.t,x2:x,y2:H-m.b,class:'gl'}));
    const tx=el('text',{x:x,y:H-m.b+16,'text-anchor':'middle',class:'axl'});tx.textContent=fmt(t);svg.appendChild(tx);});
  svg.appendChild(Object.assign(el('text',{x:m.l+iw/2,y:H-2,'text-anchor':'middle',class:'axl'}),{textContent:xLabel}));
  const yl=el('text',{x:13,y:m.t+ih/2,'text-anchor':'middle',class:'axl',transform:`rotate(-90 13 ${m.t+ih/2})`});yl.textContent=yLabel;svg.appendChild(yl);
  if(envelope&&envelope.length){let up='',dn='';
    envelope.forEach((p,i)=>{const y=tf(p.hi);if(y!=null)up+=(up?'L':'M')+X(p.x)+' '+Y(y)+' ';});
    for(let i=envelope.length-1;i>=0;i--){const y=tf(envelope[i].lo);if(y!=null)dn+='L'+X(envelope[i].x)+' '+Y(y)+' ';}
    if(up)svg.appendChild(el('path',{d:up+dn+'Z',fill:'rgba(139,92,246,.10)',stroke:'none'}));}
  if(band&&band.length){let up='',dn='';band.forEach((p,i)=>{up+=(i?'L':'M')+X(p.x)+' '+Y(p.mean+p.std)+' ';});
    for(let i=band.length-1;i>=0;i--){const p=band[i];dn+='L'+X(p.x)+' '+Y(Math.max(ymn,p.mean-p.std))+' ';}
    svg.appendChild(el('path',{d:up+dn+'Z',fill:'rgba(34,211,238,.13)',stroke:'none'}));
    band.forEach(p=>{const x=X(p.x);svg.appendChild(el('line',{x1:x,y1:Y(p.mean+p.std),x2:x,y2:Y(p.mean-p.std),stroke:'rgba(34,211,238,.55)','stroke-width':1.4}));
      [p.mean+p.std,p.mean-p.std].forEach(v=>svg.appendChild(el('line',{x1:x-3,y1:Y(v),x2:x+3,y2:Y(v),stroke:'rgba(34,211,238,.55)','stroke-width':1.4})));});
    let d='';band.forEach((p,i)=>{d+=(i?'L':'M')+X(p.x)+' '+Y(p.mean)+' ';});
    svg.appendChild(el('path',{d,fill:'none',stroke:'#22d3ee','stroke-width':2.4,'stroke-linejoin':'round'}));
    band.forEach(p=>svg.appendChild(el('circle',{cx:X(p.x),cy:Y(p.mean),r:2.6,fill:'#22d3ee'})));}
  series.forEach(s=>{const P=s.pts.map(p=>({x:p.x,y:tf(p.y),acc:p.acc})).filter(p=>p.y!=null);if(!P.length)return;
    let d='';P.forEach((p,i)=>{d+=(i?'L':'M')+X(p.x)+' '+Y(p.y)+' ';});
    svg.appendChild(el('path',{d,fill:'none',stroke:s.color,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round',opacity:.95}));
    P.forEach(p=>{if(p.acc&&p.x>0)svg.appendChild(el('circle',{cx:X(p.x),cy:Y(p.y),r:2.2,fill:s.color}));});});
  const tip=document.getElementById('tip');
  const hl=el('line',{y1:m.t,y2:H-m.b,stroke:'#5b6478','stroke-width':1,opacity:0});svg.appendChild(hl);
  const hit=el('rect',{x:m.l,y:m.t,width:iw,height:ih,fill:'transparent'});svg.appendChild(hit);
  hit.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect();const px=(ev.clientX-r.left)/r.width*W;
    const xv=xmn+(px-m.l)/iw*(xmx-xmn);hl.setAttribute('x1',px);hl.setAttribute('x2',px);hl.setAttribute('opacity',.6);
    const near=pts=>{let b=null,bd=1e18;pts.forEach(p=>{const dd=Math.abs(p.x-xv);if(dd<bd){bd=dd;b=p;}});return b;};let rows=[];
    if(band&&band.length){const p=near(band.map(b=>({x:b.x,y:b.mean,std:b.std})));if(p)rows.push(`<b>mean</b> ${fmt(p.y)} &plusmn;${fmt(p.std)}`);}
    series.forEach(s=>{const p=near(s.pts);if(p)rows.push(`<span style="color:${s.color}">&#9632;</span> ${s.name}: <b>${fmt(p.y)}</b>`);});
    tip.innerHTML=`<div style="color:var(--muted);margin-bottom:3px">${xLabel} &asymp; ${fmt(xv)}</div>`+rows.join('<br>');
    tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+12)+'px';tip.style.opacity=1;});
  hit.addEventListener('mouseleave',()=>{tip.style.opacity=0;hl.setAttribute('opacity',0);});
  mount.innerHTML='';mount.appendChild(svg);
}

function scatter(mount,pts){
  const W=mount.clientWidth||440,H=320,m={l:44,r:16,t:12,b:36},iw=W-m.l-m.r,ih=H-m.t-m.b;
  const P=pts.filter(p=>p.headroom&&p.spread);if(!P.length){mount.innerHTML='<div class="empty">no data</div>';return;}
  const xs=P.map(p=>Math.log10(p.headroom)),ys=P.map(p=>p.spread);
  let xmn=Math.min(...xs,0),xmx=Math.max(...xs),ymn=Math.min(...ys,1),ymx=Math.max(...ys);
  xmx+=(xmx-xmn)*0.12||0.3;ymx+=(ymx-ymn)*0.12||0.1;ymn-=(ymx-ymn)*0.05;
  const X=v=>m.l+(v-xmn)/(xmx-xmn)*iw,Y=v=>m.t+(1-(v-ymn)/(ymx-ymn))*ih;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:'none'});
  [1,2,5,10,20,50,100,200].filter(h=>Math.log10(h)>=xmn&&Math.log10(h)<=xmx).forEach(h=>{const x=X(Math.log10(h));
    svg.appendChild(el('line',{x1:x,y1:m.t,x2:x,y2:H-m.b,class:'gl'}));
    const tx=el('text',{x:x,y:H-m.b+16,'text-anchor':'middle',class:'axl'});tx.textContent=h+'×';svg.appendChild(tx);});
  ticks(ymn,ymx,5).forEach(t=>{const y=Y(t);svg.appendChild(el('line',{x1:m.l,y1:y,x2:W-m.r,y2:y,class:'gl'}));
    const tx=el('text',{x:m.l-6,y:y+3,'text-anchor':'end',class:'axl'});tx.textContent=t.toFixed(1)+'×';svg.appendChild(tx);});
  svg.appendChild(Object.assign(el('text',{x:m.l+iw/2,y:H-2,'text-anchor':'middle',class:'axl'}),{textContent:'headroom (baseline ÷ best, log)'}));
  const yl=el('text',{x:12,y:m.t+ih/2,'text-anchor':'middle',class:'axl',transform:`rotate(-90 12 ${m.t+ih/2})`});yl.textContent='inter-trial spread (max ÷ min)';svg.appendChild(yl);
  const tip=document.getElementById('tip');
  P.forEach(p=>{const x=X(Math.log10(p.headroom)),y=Y(p.spread),c=TIER[p.tier]||'#94a3b8';
    const g=el('g',{});g.appendChild(el('circle',{cx:x,cy:y,r:6,fill:c,'fill-opacity':.85,stroke:'#0b0d11','stroke-width':1}));
    const tx=el('text',{x:x+9,y:y+3,class:'axl',fill:'var(--text)','font-size':10.5});tx.textContent=p.task;g.appendChild(tx);
    g.addEventListener('mousemove',ev=>{tip.innerHTML=`<b>${p.task}</b> <span class="tag ${p.tier}">${p.tier}</span><br>headroom ${fmt(p.headroom)}× &middot; spread ${fmt(p.spread)}×`;
      tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+12)+'px';tip.style.opacity=1;});
    g.addEventListener('mouseleave',()=>tip.style.opacity=0);svg.appendChild(g);});
  mount.innerHTML='';mount.appendChild(svg);
}

function qtable(mount,tasks){
  const cols=[['task','task'],['tier','tier'],['baseline','base'],['optimal','best'],['headroom','head'],['spread','spread'],['conv_median','g→1%']];
  const order={strong:0,moderate:1,weak:2};
  let rows=[...tasks].sort((a,b)=>{
    if(S.sort==='tier')return (order[a.tier]-order[b.tier])||((b.headroom||0)-(a.headroom||0));
    const av=a[S.sort],bv=b[S.sort];if(typeof av==='string')return av.localeCompare(bv);return (bv||0)-(av||0);});
  let h='<tr>'+cols.map(c=>`<th data-k="${c[0]}">${c[1]}</th>`).join('')+'</tr>';
  rows.forEach(t=>{h+='<tr>'+
    `<td>${t.task}</td><td><span class="tag ${t.tier}">${t.tier}</span></td>`+
    `<td>${fmt(t.baseline)}</td><td>${fmt(t.optimal)}</td>`+
    `<td>${t.headroom?fmt(t.headroom)+'×':'—'}</td><td>${t.spread?fmt(t.spread)+'×':'—'}</td>`+
    `<td>${t.conv_median!=null?t.conv_median:'—'}</td></tr>';});
  mount.innerHTML=h;
  mount.querySelectorAll('th').forEach(th=>th.onclick=()=>{S.sort=th.dataset.k;qtable(mount,tasks);});
}

function taskSeries(t){const n=t.trials.length;return t.trials.map((tr,idx)=>({
  name:tr.run.replace(/-gpt.*$/,'').replace(/^.*?-/,''),color:grad(n>1?idx/(n-1):0),
  pts:tr.pts.map(p=>{const x=S.x==='i'?p.i:p.wall;let y=p.best;
    if(S.y==='norm'){const den=t.baseline-t.optimal;y=den>0?(p.best-t.optimal)/den:1;}return {x,y,acc:p.acc};})}));}

function envelopeOf(t){const by={};t.trials.forEach(tr=>tr.pts.forEach(p=>{const x=S.x==='i'?p.i:Math.round(p.wall);
  let y=p.best;if(S.y==='norm'){const den=t.baseline-t.optimal;y=den>0?(p.best-t.optimal)/den:1;}
  (by[x]=by[x]||[]).push(y);}));
  return Object.keys(by).map(Number).sort((a,b)=>a-b).map(x=>({x,lo:Math.min(...by[x]),hi:Math.max(...by[x])}));}

function render(){const d=S.data;if(!d)return;
  document.getElementById('prefix').textContent=d.prefix;const s=d.status,pills=[];
  if(s.total)pills.push(`<span class="pill">jobs <b>${s.total}</b></span>`);
  pills.push(`<span class="pill run">running <b>${s.running}</b></span>`);
  pills.push(`<span class="pill done">done <b>${s.done}${s.total?'/'+s.total:''}</b></span>`);
  pills.push(`<span class="pill">tasks <b>${d.tasks.length}</b></span>`);
  if(s.finished)pills.push(`<span class="pill done"><b>complete</b></span>`);
  document.getElementById('pills').innerHTML=pills.join('');
  document.getElementById('updated').textContent=new Date().toLocaleTimeString();
  scatter(document.getElementById('scatter'),d.tasks);
  qtable(document.getElementById('qtable'),d.tasks);
  chart(document.getElementById('agg'),{series:[],band:d.aggregate.map(a=>({x:a.i,mean:a.mean,std:a.std})),
    xLabel:'iteration',yLabel:'normalized best (1→0)',yFrom0:true});
  const grid=document.getElementById('taskGrid');grid.innerHTML='';
  const xLabel=S.x==='i'?'iteration':'wall-clock (s)';
  const yLabel=S.y==='norm'?'normalized best (1→0)':(S.l==='log'?'best score (log)':'best score');
  const ylog=(S.y==='raw'&&S.l==='log');
  d.tasks.forEach(t=>{const card=document.createElement('div');card.className='card';
    card.innerHTML=`<h2>${t.task} <span class="tag ${t.tier}" style="font-size:10px;vertical-align:middle">${t.tier}</span></h2>
      <div class="stat"><span>baseline <b>${fmt(t.baseline)}</b></span><span>best <b>${fmt(t.optimal)}</b></span>
        <span>headroom <b>${t.headroom?fmt(t.headroom)+'×':'—'}</b></span>
        <span>spread <b>${t.spread?fmt(t.spread)+'×':'—'}</b></span>
        <span>trials <b>${t.n_trials}</b></span></div><div class="plot"></div><div class="legend"></div>`;
    grid.appendChild(card);const series=taskSeries(t);
    chart(card.querySelector('.plot'),{series,envelope:envelopeOf(t),xLabel,yLabel,yFrom0:S.y==='norm',ylog});
    card.querySelector('.legend').innerHTML=series.map(s=>`<span class="k"><span class="sw" style="background:${s.color}"></span>${s.name}</span>`).join('');});
}
async function refresh(){try{const r=await fetch('/api/data');S.data=await r.json();render();}
  catch(e){document.getElementById('updated').textContent='offline';}}
['x','y','l'].forEach(dim=>document.querySelectorAll(`#${dim}seg button`).forEach(b=>b.onclick=()=>{
  S[dim]=b.dataset[dim];document.querySelectorAll(`#${dim}seg button`).forEach(x=>x.classList.toggle('on',x===b));render();}));
window.addEventListener('resize',()=>{clearTimeout(window._rt);window._rt=setTimeout(render,150);});
refresh();setInterval(refresh,10000);
</script></body></html>
"""


def make_handler(prefix):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/api/data"):
                body = json.dumps(build_data(prefix)).encode()
                ct = "application/json"
            else:
                body = PAGE.encode()
                ct = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--prefix", default="5xB-")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(args.prefix))
    print(f"dashboard: http://{args.host}:{args.port}  (prefix {args.prefix})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
