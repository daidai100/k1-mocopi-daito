"""Self-contained, interactive human/K1 reference viewer; no web dependencies."""

import json
from pathlib import Path

import numpy as np

from .contracts import MotionClip, PARENTS


def write_viewer(clip_path, output, human_path=None):
    clip = MotionClip.load(clip_path)
    human = None
    if human_path:
        with np.load(human_path, allow_pickle=False) as data:
            human = data["positions"].copy()
            if clip.source_times is not None:
                times = data["times"]
                if len(times) != len(human) or np.any(np.diff(times) <= 0):
                    raise ValueError("Human timestamps must match positions and increase")
                indices = np.searchsorted(times, clip.source_times)
                if np.any(indices >= len(times)) or not np.allclose(
                    times[indices], clip.source_times, rtol=0, atol=1e-9
                ):
                    raise ValueError("Human capture does not contain the reference source timestamps")
                human = human[indices]
        if len(human) != len(clip.times):
            raise ValueError("Human and robot frame counts differ")
        cal = clip.metadata.get("calibration")
        if cal:
            from scipy.spatial.transform import Rotation

            shape = human.shape
            human = (
                Rotation.from_euler("z", -cal["yaw"]).apply((human - cal["origin"]).reshape(-1, 3))
                * cal["scale"]
                + np.asarray(cal["robot_origin"])
            ).reshape(shape)
    payload = {
        "times": clip.times.tolist(),
        "robot": clip.values["landmarks"].round(5).tolist(),
        "human": human.round(5).tolist() if human is not None else None,
        "contacts": clip.values["contacts"].tolist(),
        "valid": clip.values["valid"].tolist(),
        "parents": PARENTS,
        "metadata": clip.metadata,
    }
    data_json = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    html = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>K1 Motion · Reference review</title><style>
*{box-sizing:border-box}body{margin:0;background:#101820;color:#eaf0f6;font:15px system-ui,sans-serif}
main{max-width:1400px;margin:auto;padding:26px}h1{font-size:26px;margin:0}p{color:#a7bac9}header{display:flex;justify-content:space-between;gap:20px;align-items:center}
.badge{padding:8px 12px;border:1px solid #617587;border-radius:20px;color:#c6d5df;font-size:13px}.panels{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel{background:#17232e;border:1px solid #2b3e4d;border-radius:14px;overflow:hidden}.label{padding:18px;font-weight:600}canvas{display:block;width:100%;height:440px;cursor:grab}
.controls{display:flex;gap:16px;align-items:center;padding:20px 0;flex-wrap:wrap}button,select{background:#203846;color:#eef8ff;border:1px solid #477084;border-radius:8px;padding:10px 16px;font:inherit;cursor:pointer}button{background:#237d82}
input[type=range]{flex:1;min-width:200px;accent-color:#5bccc2}.details{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.details>div{padding:18px;background:#17232e;border-radius:10px}.details span{display:block;color:#94acba;font-size:12px;margin-bottom:7px}
#source{overflow-wrap:anywhere}.hint{font-size:13px}a{color:#67d0c4}@media(max-width:700px){.panels{grid-template-columns:1fr}canvas{height:300px}.details{grid-template-columns:1fr 1fr}header{display:block}.badge{display:inline-block;margin-top:15px}}
</style><main><header><div><h1>K1 motion review</h1><p id="source"></p></div><div class="badge">Kinematic reference · physics acceptance separate</div></header>
<div class="panels"><div class="panel"><div class="label">Human capture</div><canvas id="human"></canvas></div><div class="panel"><div class="label">K1 reference</div><canvas id="robot"></canvas></div></div>
<div class="controls"><button id="play">Play</button><input id="seek" type="range" min="0" step="1" aria-label="Motion frame"><select id="speed" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><output id="time"></output></div>
<div class="details"><div><span>Reference status</span><b id="valid"></b></div><div><span>Left / right foot contact estimate</span><b id="feet"></b></div><div><span>Family / split</span><b id="family"></b></div><div><span>Controls</span>Drag to rotate · scroll to zoom</div></div><p class="hint">Body positions are metres, with a shared calibrated scale when available. Foot labels are causal estimates. Invalid reference frames remain visible.</p></main>
<script id="motion" type="application/json">__PAYLOAD__</script><script>
const d=JSON.parse(document.getElementById('motion').textContent), slider=document.getElementById('seek'),play=document.getElementById('play');
slider.max=d.times.length-1;slider.value=0;let index=0,playing=false,clock=d.times[0],last=performance.now(),yaw=-0.9,elev=.4,zoom=330;
document.getElementById('source').textContent=d.metadata.source_motion_id||'Reference';document.getElementById('family').textContent=(d.metadata.family||'—')+' / '+(d.metadata.split||'—');
function draw(canvas,frames){let rect=canvas.getBoundingClientRect(),ratio=devicePixelRatio||1;canvas.width=rect.width*ratio;canvas.height=rect.height*ratio;let ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);let w=rect.width,h=rect.height;
if(!frames){ctx.fillStyle='#9ab1c1';ctx.fillText('No human frame artifact supplied',30,60);return}let p=frames[index],center=p[0],scale=zoom*Math.min(w/600,h/440);
function project(v){let x=v[0]-center[0],y=v[1]-center[1],z=v[2],a=Math.cos(yaw)*x-Math.sin(yaw)*y,b=Math.sin(yaw)*x+Math.cos(yaw)*y;return[w/2+a*scale,h*.84-(z*Math.cos(elev)-b*Math.sin(elev))*scale]}
ctx.lineWidth=1;ctx.strokeStyle='#29404f';for(let a=-1;a<=1.01;a+=.1){for(let axis=0;axis<2;axis++){let u=project([center[0]+(axis?a:-1),center[1]+(axis?-1:a),0]),v=project([center[0]+(axis?a:1),center[1]+(axis?1:a),0]);ctx.beginPath();ctx.moveTo(...u);ctx.lineTo(...v);ctx.stroke()}}
ctx.lineWidth=5;ctx.lineCap='round';p.forEach((v,i)=>{let pt=project(v),par=d.parents[i];if(par>=0){ctx.strokeStyle=[3,4,5,9,10,11,12].includes(i)?'#58d4c2':'#86a9ed';ctx.beginPath();ctx.moveTo(...project(p[par]));ctx.lineTo(...pt);ctx.stroke()}ctx.fillStyle='#f2f5f9';ctx.beginPath();ctx.arc(...pt,4,0,2*Math.PI);ctx.fill()});}
function render(){draw(document.getElementById('human'),d.human);draw(document.getElementById('robot'),d.robot);slider.value=index;document.getElementById('time').textContent=(d.times[index]-d.times[0]).toFixed(2)+' s / '+(d.times.at(-1)-d.times[0]).toFixed(2)+' s';document.getElementById('valid').textContent=d.valid[index]?'Kinematics accepted':'Rejected frame';document.getElementById('feet').textContent=d.contacts[index].map(x=>x?'contact':'swing').join(' / ')}
play.onclick=()=>{playing=!playing;play.textContent=playing?'Pause':'Play';if(playing&&index===d.times.length-1){index=0;clock=d.times[0]}};slider.oninput=()=>{index=Number(slider.value);clock=d.times[index];render()};
for(const canvas of document.querySelectorAll('canvas')){let drag=null;canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.01;elev=Math.max(-.2,Math.min(1.2,elev+(e.clientY-drag[1])*.005));drag=[e.clientX,e.clientY];render()}};canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(100,Math.min(800,zoom*Math.exp(-e.deltaY*.001)));render()}}
function animate(now){let dt=(now-last)/1000;last=now;if(playing){clock+=dt*Number(document.getElementById('speed').value);while(index+1<d.times.length&&d.times[index+1]<=clock)index++;if(index===d.times.length-1){playing=false;play.textContent='Play'}render()}requestAnimationFrame(animate)}addEventListener('resize',render);render();requestAnimationFrame(animate);
</script></html>""".replace("__PAYLOAD__", data_json)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(html)
    return str(output)
