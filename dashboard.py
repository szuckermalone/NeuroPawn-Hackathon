"""Offline interactive EEG session dashboard. Experimental research visualisation only.

Run: python dashboard.py --demo --out dashboard_demo.html
     python dashboard.py --session runs/demo01 --out dashboard.html
The resulting HTML bundles Plotly and does not need a server or internet access.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import welch
from plotly.offline.offline import get_plotlyjs
from signals import CHANNELS, extract
from vigilance import VigilanceMonitor

BANDS = [('Delta', 0.5, 4), ('Theta', 4, 8), ('Alpha', 8, 13), ('Beta', 13, 30), ('Gamma', 30, 45)]


def demo_recording(seconds=110, fs=125, seed=22):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds*fs))/fs
    fade = np.clip((t-38)/30, 0, 1)
    phases = np.linspace(0, 1.4, 8)
    eeg = rng.normal(0, 2.1, (len(t), 8))
    for ch in range(8):
        theta = (2.2 + 11.0*fade) * np.sin(2*np.pi*6*t+phases[ch])
        alpha = (7.0 + 5.2*fade) * np.sin(2*np.pi*10*t+phases[ch]*0.6)
        beta = (4.7 - 2.1*fade) * np.sin(2*np.pi*18*t+phases[ch]*1.7)
        eeg[:, ch] += theta+alpha+beta
    events = [{'elapsed_s': s, 'trial': i+1, 'rt_ms': 410+7*i+(80 if i>4 else 0), 'outcome': 'correct'} for i,s in enumerate(range(17,seconds-5,14))]
    return t, eeg, fs, events, 'SYNTHETIC DEMO', ['F3','F4','C3','C4','P3','P4','O1','O2']


def load_recording(path):
    path=Path(path)
    meta=json.loads((path/'session.json').read_text())
    arr=np.loadtxt(path/'eeg.csv',delimiter=',',skiprows=1,ndmin=2)
    if arr.shape[1] < 3 or len(arr)<20:
        raise ValueError('EEG session missing data or channels')
    if arr.shape[1] != 9:
        raise ValueError('This dashboard expects eight EEG channels plus timestamp, in the saved project format')
    t=arr[:,0]-arr[0,0]
    x=arr[:,1:]
    if np.any(np.diff(t)<=0):
        raise ValueError('EEG timestamps must be strictly increasing')
    events=[]
    if (path/'events.jsonl').exists():
        for line in (path/'events.jsonl').read_text().splitlines():
            if line.strip():
                e=json.loads(line)
                if 'elapsed_s' in e:events.append({k:e.get(k) for k in ['elapsed_s','trial','rt_ms','outcome']})
    labels=meta.get('channels', CHANNELS)
    if len(labels)!=8: labels=CHANNELS
    return t,x,int(meta['fs']),events,'RECORDED SESSION ('+str(meta.get('mode','unknown')).upper()+')',labels


def compute(t,x,fs,events,source,labels):
    dt=2.0
    ends=np.arange(8, t[-1]-0.05, dt)
    if len(ends)<4:
        raise ValueError('At least 16 s of usable EEG is required')
    monitor=VigilanceMonitor(calibration_windows=8)
    windows=[]
    for end in ends:
        features,quality=extract(t,x,end,fs)
        outcome=monitor.update(features,quality)
        ix=np.searchsorted(t,end)
        start=np.searchsorted(t,end-4)
        segment=x[start:ix]
        if len(segment)<2*fs:
            continue
        freqs,psd=welch(segment,fs=fs,nperseg=min(2*fs,len(segment)),axis=0)
        powers={}
        for name,lo,hi in BANDS:
            sel=(freqs>=lo)&(freqs<hi)
            powers[name]=np.round(np.trapezoid(psd[sel],freqs[sel],axis=0),3).tolist() if sel.sum()>1 else [0.0]*8
        lim=(freqs>=1)&(freqs<=45)
        overall=np.mean(psd,axis=1)
        spectral=np.round(10*np.log10(np.maximum(overall[lim],1e-6)),2)
        windows.append({'t':round(float(end),2),'bands':powers,'spectrum':spectral.tolist(),
                        'score':outcome['score'],'state':outcome['state'], 'quality':outcome['signal_quality'],
                        'alert':bool(outcome['alert']), 'calibration':outcome['calibration']})
    if not windows:
        raise ValueError('Not enough EEG for a dashboard')
    decimate=max(1,int(np.ceil(fs/32)))
    sample_indices=np.arange(0,len(t),decimate)
    # Cap the embedded recording to 30,000 points per channel.
    if len(sample_indices)>30000:
        sample_indices=sample_indices[::int(np.ceil(len(sample_indices)/30000))]
    frame_freqs=freqs[lim]
    return {'source':source, 'fs':fs,'channels':list(labels),'time':np.round(t[sample_indices],3).tolist(),
            'eeg':np.round(x[sample_indices],2).tolist(),'windows':windows,
            'freq':np.round(frame_freqs,2).tolist(), 'events':events,
            'full_length':round(float(t[-1]),1)}


HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroGuard | DriverVigilance EEG Lab</title><style>
:root{--bg:#081018;--panel:#101c28;--edge:#21354a;--ink:#dcebf5;--muted:#88a0b4;--cyan:#30d6dc;--warn:#ffb65e;--red:#ff657a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,-apple-system,Segoe UI,sans-serif}
header{border-bottom:1px solid var(--edge);padding:16px 26px;display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;background:#0b1620}
.brand{font-size:21px;font-weight:800;letter-spacing:.3px}.brand span{color:var(--cyan)}.sub{font-size:11px;color:var(--muted);letter-spacing:1px}
.layout{padding:18px;max-width:1750px;margin:auto}.banner{background:#35281b;color:#ffce8e;padding:11px 15px;border:1px solid #684524;border-radius:10px;margin-bottom:15px}
.controls,.panel{background:var(--panel);border:1px solid var(--edge);border-radius:12px;padding:14px}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:14px}button,select{background:#15293c;border:1px solid #39506a;color:#dcebf5;border-radius:7px;padding:8px 12px;cursor:pointer}button:hover{border-color:var(--cyan)}button.active{background:#10566a}input[type=range]{accent-color:var(--cyan);flex:1;min-width:230px}
.stats{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:12px;margin-bottom:14px}.stat{background:var(--panel);border:1px solid var(--edge);border-radius:12px;padding:16px}.stat label{display:block;font-size:11px;color:var(--muted);letter-spacing:.8px;text-transform:uppercase}.stat strong{font-size:23px;display:block;margin-top:6px}.stat small{color:var(--muted)}
.grid{display:grid;grid-template-columns:1.55fr 1fr;gap:14px}.panel h3{font-size:13px;font-weight:650;margin:0 0 10px}.plot{height:260px}.plot.tall{height:355px}.plot.short{height:195px}.full{grid-column:1/-1}.annotations{color:var(--muted);font-size:12px;margin-top:8px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--cyan);margin-right:7px}
@media(max-width:950px){.grid{grid-template-columns:1fr}.full{grid-column:1}.stats{grid-template-columns:repeat(2,minmax(130px,1fr))}.plot.tall{height:300px}}
</style></head><body><header><div><div class="brand"><span>◉</span> NeuroGuard <span>LAB</span></div><div class="sub">DRIVERVIGILANCE / EEG RESEARCH CONSOLE</div></div><div class="sub" id="sourceLabel"></div></header>
<div class="layout"><div class="banner"><b>Research prototype:</b> EEG changes are not a validated sleepiness diagnosis. Not for use to decide whether driving is safe. Synthetic data, when selected, are artificial.</div>
<div class="controls"><button id="play">▶ Play</button><button id="reset">↺ Reset</button><span id="currentTime">00:08</span><input type="range" id="seek" min="0" max="100" step="1" value="0"><span id="duration"></span><label>Speed <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label><label>Waveform window <select id="span"><option value="2">2 s</option><option value="5" selected>5 s</option><option value="10">10 s</option><option value="16">16 s</option></select></label><label>EEG channel <select id="channel"></select></label></div>
<div class="stats"><div class="stat"><label>Vigilance status</label><strong id="state">...</strong><small id="quality">...</small></div><div class="stat"><label>Experimental EEG index</label><strong id="score">...</strong><small>Reference-normalised, not probability</small></div><div class="stat"><label>Calibration</label><strong id="calibration">...</strong><small>Accepted EEG windows</small></div><div class="stat"><label>Sample rate</label><strong id="sampleRate"></strong><small>Hz at acquisition</small></div><div class="stat"><label>Simulator responses</label><strong id="reaction">...</strong><small>Median valid reaction time</small></div></div>
<div class="grid"><section class="panel full"><h3>01 / Eight-channel EEG waveforms · µV</h3><div id="wave" class="plot tall"></div></section>
<section class="panel"><h3>02 / Mean spectral power density · all channels</h3><div id="psd" class="plot"></div></section>
<section class="panel"><h3>03 / Band power distribution · selected channel</h3><div id="bands" class="plot"></div></section>
<section class="panel"><h3>04 / Spectrogram · mean over eight channels</h3><div id="spectrogram" class="plot tall"></div></section>
<section class="panel"><h3>05 / Scalp map · illustrative spatial interpolation</h3><div class="controls" style="margin:0 0 6px;padding:6px;border:none"><label>Map metric <select id="mapBand"><option>Alpha</option><option>Theta</option><option>Beta</option><option>Delta</option></select></label><span style="font-size:11px;color:var(--muted)">Montage must match physical electrodes</span></div><div id="topo" class="plot"></div></section>
<section class="panel full"><h3>06 / Vigilance index over time + recorded steering events</h3><div id="trend" class="plot"></div><div class="annotations">Vertical cursor follows playback. Orange marks indicate recorded steering responses; a red marker indicates a missed response.</div></section></div>
<div class="annotations" style="padding:16px 5px">Dashboard design inspired by PiEEG-server. This is an independent visualisation layer reading the DriverVigilance session format. No PiEEG-server software is bundled.</div></div>
<script>__PLOTLY_JS__</script><script id="eeg-payload" type="application/json">__DATA_JSON__</script><script>
const D=JSON.parse(document.getElementById('eeg-payload').textContent),W=D.windows;
let idx=0,playing=false,clock=null;
const $=id=>document.getElementById(id), ink='#dcebf5',muted='#8da5bb',cyan='#30d6dc';
const layout=(title='')=>({paper_bgcolor:'#101c28',plot_bgcolor:'#101c28',font:{color:ink,size:11},margin:{l:57,r:22,t:18,b:38},showlegend:false,
  xaxis:{gridcolor:'#1f3548',zeroline:false},yaxis:{gridcolor:'#1f3548',zeroline:false},uirevision:'neuroguard'});
const cfg={displayModeBar:true,responsive:true,displaylogo:false};
const fmt=t=>`${Math.floor(t/60).toString().padStart(2,'0')}:${Math.floor(t%60).toString().padStart(2,'0')}`;
const bandcolors=['#6478e2','#30d6dc','#4cda9d','#ffba60','#ed78bd'];
function channelIndex(){let v=$('channel').value;return v==='all'?-1:Number(v)}
function wave(){let end=W[idx].t,start=Math.max(0,end-Number($('span').value));let a=D.time.findIndex(v=>v>=start);if(a<0)a=0;let b=D.time.findIndex(v=>v>end);if(b<0)b=D.time.length;
 const traces=D.channels.map((ch,c)=>({type:'scattergl',mode:'lines',x:D.time.slice(a,b),y:D.eeg.slice(a,b).map(row=>row[c]+c*115),line:{width:1,color:['#30d6dc','#d39afa','#51bfff','#ffad77','#8ade9a','#ffe477','#a3b0ff','#fc86b2'][c]},name:ch,hovertemplate:`${ch} %{x:.2f}s · %{y:.1f} shifted µV<extra></extra>`}));
 let L=layout();L.yaxis={...L.yaxis,tickvals:D.channels.map((_,c)=>c*115),ticktext:D.channels};L.xaxis.title={text:'Session time (s)'};Plotly.react('wave',traces,L,cfg)}
function spectrum(){let w=W[idx],f=D.freq,y=w.spectrum;let ch=channelIndex();
 // Frequency PSD is averaged across all channels, not channel-specific.
 let l=layout();l.xaxis.title={text:'Frequency (Hz)'};l.yaxis.title={text:'Mean power (dB)'};
 Plotly.react('psd',[{type:'scatter',mode:'lines',x:f,y:y,line:{color:cyan,width:2},fill:'tozeroy',fillcolor:'rgba(48,214,220,0.08)'}],l,cfg)}
function powers(){let w=W[idx],ch=channelIndex(),names=['Delta','Theta','Alpha','Beta','Gamma'],values=names.map(n=>ch<0?w.bands[n].reduce((a,b)=>a+b,0)/8:w.bands[n][ch]);let l=layout();l.yaxis.title={text:'Integrated power (µV²)'};
 Plotly.react('bands',[{type:'bar',x:names,y:values,marker:{color:bandcolors},hovertemplate:'%{x}: %{y:.2f} µV²<extra></extra>'}],l,cfg)}
function spectro(){let start=Math.max(0,idx-34);let segment=W.slice(start,idx+1),z=D.freq.map((_,j)=>segment.map(w=>w.spectrum[j]));let l=layout();l.xaxis.title={text:'Time (s)'};l.yaxis.title={text:'Frequency (Hz)'};
 Plotly.react('spectrogram',[{type:'heatmap',x:segment.map(w=>w.t),y:D.freq,z:z,colorscale:'Turbo',showscale:true,colorbar:{title:{text:'dB',side:'right'},thickness:9}}],l,cfg)}
const pos=[[-.5,.8],[.5,.8],[-.55,.12],[.55,.12],[-.53,-.45],[.53,-.45],[-.33,-.83],[.33,-.83]];
function topo(){let w=W[idx],key=$('mapBand').value,v=w.bands[key],z=[],xs=[],ys=[];for(let j=0;j<38;j++){let row=[];let yy=-1.04+2.08*j/37;ys.push(yy);for(let i=0;i<38;i++){let xx=-1.04+2.08*i/37;if(j===0)xs.push(xx);if(xx*xx+yy*yy>1){row.push(null);continue}let sw=0,sv=0;for(let c=0;c<8;c++){let dist=Math.hypot(xx-pos[c][0],yy-pos[c][1]);let wt=1/(dist*dist+.04);sw+=wt;sv+=wt*v[c]}row.push(sv/sw)}z.push(row)}
 let l=layout();l.margin={l:20,r:20,t:4,b:10};l.xaxis={visible:false,range:[-1.08,1.08],scaleanchor:'y'};l.yaxis={visible:false,range:[-1.08,1.08]};
 let traces=[{type:'heatmap',x:xs,y:ys,z,colorscale:'Viridis',showscale:true,colorbar:{thickness:10,title:{text:'µV²'}}},{type:'scatter',mode:'markers+text',x:pos.map(p=>p[0]),y:pos.map(p=>p[1]),text:D.channels,textposition:'top center',marker:{size:7,color:'#fff',line:{color:'#102031',width:1}},textfont:{color:'#fff',size:11},hovertext:v.map(a=>a.toFixed(2)+' µV²'),hoverinfo:'text'}];
 Plotly.react('topo',traces,l,cfg)}
function trend(){let score=W.map(w=>w.score),valid=W.map(w=>w.alert?'#ff657a':cyan),ev=D.events.filter(e=>e.outcome==='miss'||e.rt_ms!=null),l=layout();l.yaxis.title={text:'EEG index (arbitrary units)'};l.xaxis.title={text:'Session time (s)'};
 let vals=score.filter(x=>x!=null),lo=vals.length?Math.min(-1,Math.min(...vals)-.5):-1;
 let traces=[{type:'scatter',mode:'lines+markers',x:W.map(w=>w.t),y:score,line:{color:cyan,width:2},marker:{size:5,color:valid},connectgaps:false,hovertemplate:'%{x}s: %{y:.2f}<extra>Experimental index</extra>'},
 {type:'scatter',mode:'markers',x:ev.map(e=>e.elapsed_s),y:ev.map(e=>lo),marker:{size:9,color:ev.map(e=>e.outcome==='miss'?'#ff657a':'#ffb65e'),symbol:'diamond'},customdata:ev.map(e=>e.rt_ms),hovertemplate:'Steering event: %{x}s · RT %{customdata} ms<extra></extra>'}];
 l.shapes=[{type:'line',x0:W[idx].t,x1:W[idx].t,y0:0,y1:1,yref:'paper',line:{color:'#ffbe6b',dash:'dash',width:2}}];Plotly.react('trend',traces,l,cfg)}
function render(){let w=W[idx];$('seek').value=idx;$('currentTime').textContent=fmt(w.t);$('state').textContent=w.alert?'EEG CHANGE ALERT':w.state;$('state').style.color=w.alert?'#ff657a':cyan;
 $('score').textContent=w.score==null?'—':(w.score>0?'+':'')+w.score.toFixed(2);$('calibration').textContent=w.calibration;
 $('quality').textContent='Signal: '+w.quality;wave();spectrum();powers();spectro();topo();trend()}
function stop(){playing=false;if(clock)clearInterval(clock);clock=null;$('play').textContent='▶ Play'}
$('play').onclick=()=>{if(playing){stop();return}playing=true;$('play').textContent='Ⅱ Pause';clock=setInterval(()=>{if(idx>=W.length-1){stop();return}idx++;render()},2000/Number($('speed').value))};
$('reset').onclick=()=>{stop();idx=0;render()};$('seek').oninput=e=>{idx=Number(e.target.value);render()};$('channel').onchange=render;$('span').onchange=render;$('mapBand').onchange=render;
$('speed').onchange=()=>{if(playing){stop();$('play').click()}};
$('sourceLabel').textContent=D.source+'  /  '+D.channels.length+' CHANNELS';$('duration').textContent=fmt(D.full_length);$('seek').max=W.length-1;$('sampleRate').textContent=D.fs;
const response=D.events.filter(e=>e.rt_ms!=null).map(e=>e.rt_ms).sort((a,b)=>a-b);$('reaction').textContent=response.length?`${Math.round(response[Math.floor(response.length/2)])} ms`:'—';
$('channel').innerHTML='<option value="all">Mean of all channels</option>'+D.channels.map((ch,i)=>`<option value="${i}">${ch}</option>`).join('');
render();
</script></body></html>'''


def build_html(data, destination):
    output=Path(destination)
    output.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    html=HTML.replace('__PLOTLY_JS__',get_plotlyjs()).replace('__DATA_JSON__',payload)
    output.write_text(html,encoding='utf-8')
    return output


def main():
    parser=argparse.ArgumentParser(description='Offline interactive EEG research dashboard')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo',action='store_true',help='Generate synthetic EEG showcase')
    group.add_argument('--session',help='Path to DriverVigilance session folder')
    parser.add_argument('--out',default='neuroguard_dashboard.html')
    args=parser.parse_args()
    record=demo_recording() if args.demo else load_recording(args.session)
    out=build_html(compute(*record),args.out)
    print(f'Created: {out.resolve()} ({out.stat().st_size/1024/1024:.1f} MB)')
    print('Open the file in Chrome, Firefox, Edge or Safari. No server needed.')


if __name__=='__main__':main()
