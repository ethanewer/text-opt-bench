"""Live campaign dashboard — localhost, dependency-free.

Reads runs/<task>/<prefix>*/log.jsonl on every request and serves a clean
single-page dashboard:

  - one card per task with 5 best-so-far curves (one per trial), gradient
    colored; toggle x-axis between iterations (concurrency-normalized) and
    wall-clock seconds.
  - an aggregate card: every score scaled so baseline=1.0 and the best score
    reached so far for that task (proxy for optimal)=0.0, then averaged
    across all tasks and trials, with error bars = std across task-trials.

"best-so-far" is monotone: a regression still shows the best grade reached.

Usage:  python3.12 tools/dashboard.py [--port 8420] [--prefix 5xB-]
Then open http://127.0.0.1:8420
"""

import argparse
import json
import statistics as st
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LAUNCH = RUNS / "_campaign" / "launcher.jsonl"


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_trial(log_path):
    """One trial -> {run, baseline, pts:[{i,best,wall,score,acc}]}."""
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
            baseline = e.get("guide_score", e.get("score"))
            best = baseline
            pts.append({"i": 0, "best": baseline, "wall": 0.0,
                        "score": baseline, "acc": True})
            continue
        it = e.get("iter")
        if it is None:
            continue
        sec = e.get("seconds") or 0
        wall += sec
        gs = e.get("guide_score")
        # The loop records the running best per entry; trust it, else fall
        # back to a running min over valid attempts.
        bgs = e.get("best_guide_score")
        if _num(bgs):
            best = bgs if best is None else min(best, bgs)
        elif e.get("ok") and _num(gs):
            best = gs if best is None else min(best, gs)
        pts.append({"i": it, "best": best, "wall": round(wall, 1),
                    "score": gs if _num(gs) else None,
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
    agg = {}  # iteration -> [normalized best across task-trials]
    for task, td in sorted(tasks.items()):
        bases = [t["baseline"] for t in td["trials"] if _num(t["baseline"])]
        baseline = bases[0] if bases else None
        all_best = [p["best"] for t in td["trials"] for p in t["pts"]
                    if _num(p["best"])]
        optimal = min(all_best) if all_best else None
        cur_best = optimal
        improved = (_num(baseline) and _num(optimal) and baseline > optimal)
        den = (baseline - optimal) if improved else 0.0
        for t in td["trials"]:
            for p in t["pts"]:
                if _num(p["best"]) and den > 0:
                    nb = (p["best"] - optimal) / den
                    agg.setdefault(p["i"], []).append(nb)
        out_tasks.append({
            "task": task,
            "baseline": baseline,
            "optimal": optimal,
            "improvement": (baseline / cur_best) if (_num(baseline) and cur_best) else None,
            "n_trials": len(td["trials"]),
            "max_iter": max((p["i"] for t in td["trials"] for p in t["pts"]), default=0),
            "trials": [{"run": t["run"], "pts": t["pts"]} for t in td["trials"]],
        })

    aggregate = []
    for k in sorted(agg):
        v = agg[k]
        if len(v) < 2:
            continue
        aggregate.append({"i": k, "mean": st.mean(v),
                          "std": st.pstdev(v), "n": len(v)})

    return {"prefix": prefix, "status": launcher_status(),
            "tasks": out_tasks, "aggregate": aggregate}


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>text-opt-bm campaign</title>
<style>
:root{
  --bg:#0f1115; --panel:#171a21; --panel2:#1d212b; --line:#2a2f3a;
  --text:#e7eaf0; --muted:#8b93a7; --accent:#22d3ee; --good:#34d399;
  --grid:#232833;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
header{position:sticky;top:0;z-index:5;background:rgba(15,17,21,.92);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:14px 24px}
.hrow{display:flex;align-items:center;gap:18px;flex-wrap:wrap;max-width:1400px;margin:0 auto}
h1{font-size:16px;font-weight:650;margin:0;letter-spacing:.2px}
h1 .p{color:var(--muted);font-weight:500}
.pills{display:flex;gap:8px;flex-wrap:wrap}
.pill{background:var(--panel2);border:1px solid var(--line);border-radius:999px;
  padding:4px 11px;font-size:12px;color:var(--muted)}
.pill b{color:var(--text);font-weight:600}
.pill.run b{color:var(--accent)} .pill.done b{color:var(--good)}
.spacer{flex:1}
.controls{display:flex;gap:8px;align-items:center}
.seg{display:inline-flex;background:var(--panel2);border:1px solid var(--line);
  border-radius:9px;overflow:hidden}
.seg button{background:transparent;border:0;color:var(--muted);padding:6px 12px;
  font-size:12.5px;cursor:pointer}
.seg button.on{background:var(--accent);color:#05222a;font-weight:650}
.updated{color:var(--muted);font-size:12px;min-width:96px;text-align:right}
main{max-width:1400px;margin:0 auto;padding:22px 24px 60px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:16px 16px 8px;margin-bottom:20px}
.card h2{font-size:14px;margin:0 0 2px;font-weight:600}
.card .sub{color:var(--muted);font-size:12px;margin:0 0 8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}
.grid .card{margin:0}
.stat{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:2px 0 6px}
.stat b{color:var(--text)}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin-top:4px}
.legend .k{display:inline-flex;align-items:center;gap:5px}
.legend .sw{width:14px;height:3px;border-radius:2px;display:inline-block}
svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:fixed;pointer-events:none;background:#0b0d11;border:1px solid var(--line);
  border-radius:8px;padding:7px 9px;font-size:11.5px;color:var(--text);opacity:0;
  transition:opacity .08s;z-index:20;box-shadow:0 6px 20px rgba(0,0,0,.4);white-space:nowrap}
.empty{color:var(--muted);padding:40px;text-align:center}
.axl{fill:var(--muted);font-size:10.5px}
.gl{stroke:var(--grid);stroke-width:1}
</style></head>
<body>
<header><div class="hrow">
  <h1>text-opt-bm <span class="p" id="prefix"></span></h1>
  <div class="pills" id="pills"></div>
  <div class="spacer"></div>
  <div class="controls">
    <div class="seg" id="xseg">
      <button data-x="i" class="on">Iterations</button>
      <button data-x="wall">Wall-clock</button>
    </div>
    <div class="seg" id="yseg">
      <button data-y="norm" class="on">Normalized</button>
      <button data-y="raw">Raw score</button>
    </div>
  </div>
  <div class="updated" id="updated">—</div>
</div></header>
<main>
  <div class="card" id="aggCard">
    <h2>Aggregate — normalized best-so-far, averaged across all tasks &amp; trials</h2>
    <p class="sub">Each score scaled per task so baseline = 1.0 and the best score reached so far = 0.0.
       Curve = mean; error bars = &plusmn;1 std across task-trials at each iteration.</p>
    <div id="agg"></div>
  </div>
  <div class="grid" id="taskGrid"></div>
</main>
<div class="tip" id="tip"></div>
<script>
const S={x:'i',y:'norm',data:null};
const GRAD=[[34,211,238],[59,130,246],[139,92,246],[236,72,153]];
function grad(t){const s=GRAD,seg=(s.length-1)*t,i=Math.min(Math.floor(seg),s.length-2),f=seg-i,
  c=(a,b)=>Math.round(a+(b-a)*f);return `rgb(${c(s[i][0],s[i+1][0])},${c(s[i][1],s[i+1][1])},${c(s[i][2],s[i+1][2])})`;}
function ticks(mn,mx,n){if(mn===mx)return[mn];const sp=mx-mn,s0=sp/n,mag=Math.pow(10,Math.floor(Math.log10(s0))),
  nm=s0/mag;let st=nm<1.5?1:nm<3?2:nm<7?5:10;st*=mag;const out=[];for(let v=Math.ceil(mn/st)*st;v<=mx+1e-9;v+=st)out.push(v);return out;}
function fmt(v){if(v==null)return'—';const a=Math.abs(v);
  if(a!==0&&(a<1e-3||a>=1e6))return v.toExponential(2);
  if(a>=1000)return v.toLocaleString(undefined,{maximumFractionDigits:0});
  if(a>=1)return(+v.toFixed(2)).toString();return(+v.toFixed(4)).toString();}
const NS='http://www.w3.org/2000/svg';
function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}

// series: [{name,color,pts:[{x,y}]}]; band:[{x,mean,std}]
function chart(mount,{series,band,xLabel,yLabel,yFrom0}){
  const W=mount.clientWidth||440,H=270,m={l:52,r:14,t:12,b:34};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const xs=[],ys=[];
  series.forEach(s=>s.pts.forEach(p=>{xs.push(p.x);ys.push(p.y);}));
  if(band)band.forEach(p=>{xs.push(p.x);ys.push(p.mean+p.std);ys.push(Math.max(0,p.mean-p.std));});
  if(!xs.length){mount.innerHTML='<div class="empty">no data yet</div>';return;}
  let xmn=Math.min(...xs),xmx=Math.max(...xs),ymn=Math.min(...ys),ymx=Math.max(...ys);
  if(yFrom0){ymn=Math.min(0,ymn);ymx=Math.max(ymx,1);}
  if(xmn===xmx)xmx=xmn+1; if(ymn===ymx)ymx=ymn+1;
  const pad=(ymx-ymn)*0.06;ymn-=pad;ymx+=pad;
  const X=v=>m.l+(v-xmn)/(xmx-xmn)*iw, Y=v=>m.t+(1-(v-ymn)/(ymx-ymn))*ih;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:'none'});
  // grid + y ticks
  ticks(ymn,ymx,5).forEach(t=>{const y=Y(t);
    svg.appendChild(el('line',{x1:m.l,y1:y,x2:W-m.r,y2:y,class:'gl'}));
    const tx=el('text',{x:m.l-7,y:y+3,'text-anchor':'end',class:'axl'});tx.textContent=fmt(t);svg.appendChild(tx);});
  // x ticks
  ticks(xmn,xmx,6).forEach(t=>{const x=X(t);
    svg.appendChild(el('line',{x1:x,y1:m.t,x2:x,y2:H-m.b,class:'gl'}));
    const tx=el('text',{x:x,y:H-m.b+16,'text-anchor':'middle',class:'axl'});tx.textContent=fmt(t);svg.appendChild(tx);});
  // axis labels
  const xl=el('text',{x:m.l+iw/2,y:H-2,'text-anchor':'middle',class:'axl'});xl.textContent=xLabel;svg.appendChild(xl);
  const yl=el('text',{x:14,y:m.t+ih/2,'text-anchor':'middle',class:'axl',
    transform:`rotate(-90 14 ${m.t+ih/2})`});yl.textContent=yLabel;svg.appendChild(yl);
  // band
  if(band&&band.length){
    let up='',dn='';band.forEach((p,i)=>{up+=(i?'L':'M')+X(p.x)+' '+Y(p.mean+p.std)+' ';});
    for(let i=band.length-1;i>=0;i--){const p=band[i];dn+='L'+X(p.x)+' '+Y(Math.max(ymn,p.mean-p.std))+' ';}
    svg.appendChild(el('path',{d:up+dn+'Z',fill:'rgba(34,211,238,.13)',stroke:'none'}));
    band.forEach(p=>{const x=X(p.x);
      svg.appendChild(el('line',{x1:x,y1:Y(p.mean+p.std),x2:x,y2:Y(p.mean-p.std),stroke:'rgba(34,211,238,.55)','stroke-width':1.4}));
      [p.mean+p.std,p.mean-p.std].forEach(v=>svg.appendChild(el('line',{x1:x-3,y1:Y(v),x2:x+3,y2:Y(v),stroke:'rgba(34,211,238,.55)','stroke-width':1.4})));});
    let d='';band.forEach((p,i)=>{d+=(i?'L':'M')+X(p.x)+' '+Y(p.mean)+' ';});
    svg.appendChild(el('path',{d,fill:'none',stroke:'#22d3ee','stroke-width':2.4,'stroke-linejoin':'round'}));
    band.forEach(p=>svg.appendChild(el('circle',{cx:X(p.x),cy:Y(p.mean),r:2.6,fill:'#22d3ee'})));
  }
  // series lines
  series.forEach(s=>{if(!s.pts.length)return;let d='';s.pts.forEach((p,i)=>{d+=(i?'L':'M')+X(p.x)+' '+Y(p.y)+' ';});
    svg.appendChild(el('path',{d,fill:'none',stroke:s.color,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round',opacity:.95}));});
  // hover
  const tip=document.getElementById('tip');
  const hline=el('line',{y1:m.t,y2:H-m.b,stroke:'#5b6478','stroke-width':1,opacity:0});svg.appendChild(hline);
  const hit=el('rect',{x:m.l,y:m.t,width:iw,height:ih,fill:'transparent'});svg.appendChild(hit);
  hit.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect();
    const px=(ev.clientX-r.left)/r.width*W;const xv=xmn+(px-m.l)/iw*(xmx-xmn);
    hline.setAttribute('x1',px);hline.setAttribute('x2',px);hline.setAttribute('opacity',.6);
    let rows=[];const near=(pts)=>{let b=null,bd=1e18;pts.forEach(p=>{const dd=Math.abs(p.x-xv);if(dd<bd){bd=dd;b=p;}});return b;};
    if(band&&band.length){const p=near(band.map(b=>({x:b.x,y:b.mean,std:b.std})));if(p)rows.push(`<b>mean</b> ${fmt(p.y)} &plusmn;${fmt(p.std)}`);}
    series.forEach(s=>{const p=near(s.pts);if(p)rows.push(`<span style="color:${s.color}">&#9632;</span> ${s.name}: <b>${fmt(p.y)}</b>`);});
    tip.innerHTML=`<div style="color:var(--muted);margin-bottom:3px">${xLabel} &asymp; ${fmt(xv)}</div>`+rows.join('<br>');
    tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+12)+'px';tip.style.opacity=1;});
  hit.addEventListener('mouseleave',()=>{tip.style.opacity=0;hline.setAttribute('opacity',0);});
  mount.innerHTML='';mount.appendChild(svg);
}

function taskSeries(t){
  const n=t.trials.length;
  return t.trials.map((tr,idx)=>{
    const color=grad(n>1?idx/(n-1):0);
    const pts=tr.pts.map(p=>{
      const x=S.x==='i'?p.i:p.wall;
      let y=p.best;
      if(S.y==='norm'){const den=(t.baseline-t.optimal);y=den>0?(p.best-t.optimal)/den:1;}
      return {x,y};
    });
    return {name:tr.run.replace(/-gpt.*$/,'').replace(/^.*?-/,''),color,pts};
  });
}

function render(){
  const d=S.data;if(!d)return;
  document.getElementById('prefix').textContent=d.prefix;
  const st=d.status;const pills=[];
  if(st.total)pills.push(`<span class="pill">jobs <b>${st.total}</b></span>`);
  pills.push(`<span class="pill run">running <b>${st.running}</b></span>`);
  pills.push(`<span class="pill done">done <b>${st.done}${st.total?'/'+st.total:''}</b></span>`);
  pills.push(`<span class="pill">tasks <b>${d.tasks.length}</b></span>`);
  if(st.finished)pills.push(`<span class="pill done"><b>campaign complete</b></span>`);
  document.getElementById('pills').innerHTML=pills.join('');
  document.getElementById('updated').textContent=new Date().toLocaleTimeString();

  // aggregate
  chart(document.getElementById('agg'),{
    series:[],band:d.aggregate.map(a=>({x:a.i,mean:a.mean,std:a.std})),
    xLabel:'iteration',yLabel:'normalized best (1=baseline, 0=best)',yFrom0:true});

  const grid=document.getElementById('taskGrid');grid.innerHTML='';
  const xLabel=S.x==='i'?'iteration':'wall-clock (s)';
  const yLabel=S.y==='norm'?'normalized best (1→0)':'best score so far';
  d.tasks.forEach(t=>{
    const card=document.createElement('div');card.className='card';
    const impr=t.improvement?`${t.improvement.toFixed(2)}&times;`:'—';
    card.innerHTML=`<h2>${t.task}</h2>
      <div class="stat"><span>baseline <b>${fmt(t.baseline)}</b></span>
        <span>best <b>${fmt(t.optimal)}</b></span>
        <span>improvement <b>${impr}</b></span>
        <span>trials <b>${t.n_trials}</b></span>
        <span>iters <b>${t.max_iter}</b></span></div>
      <div class="plot"></div>
      <div class="legend"></div>`;
    grid.appendChild(card);
    const series=taskSeries(t);
    chart(card.querySelector('.plot'),{series,xLabel,yLabel,yFrom0:S.y==='norm'});
    card.querySelector('.legend').innerHTML=series.map(s=>
      `<span class="k"><span class="sw" style="background:${s.color}"></span>${s.name}</span>`).join('');
  });
}

async function refresh(){
  try{const r=await fetch('/api/data');S.data=await r.json();render();}
  catch(e){document.getElementById('updated').textContent='offline';}
}
document.querySelectorAll('#xseg button').forEach(b=>b.onclick=()=>{
  S.x=b.dataset.x;document.querySelectorAll('#xseg button').forEach(x=>x.classList.toggle('on',x===b));render();});
document.querySelectorAll('#yseg button').forEach(b=>b.onclick=()=>{
  S.y=b.dataset.y;document.querySelectorAll('#yseg button').forEach(x=>x.classList.toggle('on',x===b));render();});
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
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
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
