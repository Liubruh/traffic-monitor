// ── State ─────────────────────────────────────────────────────────
let isRunning   = false;
let pollTimer   = null;
let seenAlerts  = new Set();
const HIDDEN_ALERT_KEYWORDS = ['行人未走斑马线', '乱穿马路'];
let zoneTool    = null;
let zones       = [];
let drawingPts  = [];
let canvas, ctx, videoEl, videoWrap;

// ── Init ──────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  canvas    = document.getElementById('zoneCanvas');
  ctx       = canvas.getContext('2d');
  videoEl   = document.getElementById('videoFeed');
  videoWrap = document.getElementById('videoWrap');
  canvas.addEventListener('click',     onCanvasClick);
  canvas.addEventListener('mousemove', onCanvasMove);
  canvas.addEventListener('dblclick',  onCanvasDblClick);
  window.addEventListener('resize', () => { resizeCanvas(); redrawCanvas(); });
  videoEl.addEventListener('load', () => { resizeCanvas(); redrawCanvas(); });
  setInterval(updateClock, 1000);
  updateClock();
  loadModelStats();
});

const COCO_MAP = { n:'52.4', s:'61.8', m:'67.2', l:'69.8', x:'71.0' };

async function loadModelStats() { updateMapDisplay(); }
document.getElementById('modelSize').addEventListener('change', updateMapDisplay);

async function updateMapDisplay() {
  const sel = document.getElementById('modelSize').value;
  const el  = document.getElementById('mMap');
  const lb  = document.getElementById('mMapLabel');
  if (sel==='finetune') {
    try {
      const r = await fetch('/api/model_stats'), d = await r.json();
      if (d.map50!=null) { el.textContent=d.map50+'%'; lb.textContent='微调'; el.title=`微调 mAP@50`; }
      else { el.textContent='--'; lb.textContent='微调'; el.title='运行 train.py 训练后更新'; }
    } catch(e) { el.textContent='--'; }
  } else {
    el.textContent = (COCO_MAP[sel]||'--')+'%';
    lb.textContent = 'COCO基准';
    el.title = `YOLOv8${sel} COCO 基准 mAP@50`;
  }
}

function updateClock() {
  document.getElementById('clock').textContent = new Date().toTimeString().slice(0,8);
}

function onSourceChange() {
  const t = document.getElementById('sourceType').value;
  document.getElementById('camField').style.display  = t==='camera'?'':'none';
  document.getElementById('rtspField').style.display = t==='rtsp'?'':'none';
  document.getElementById('fileField').style.display = t==='file'?'':'none';
}

// ── File Upload ───────────────────────────────────────────────────
async function onFileUpload(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const fd = new FormData(); fd.append('file',file);
  const r = await fetch('/api/upload_video',{method:'POST',body:fd});
  const d = await r.json();
  if (d.success) {
    document.getElementById('filePath').value = d.path;
    document.getElementById('fileName').textContent = d.filename;
    toast('文件上传成功','success');
  } else toast(d.message,'error');
}

// ── Target classes ────────────────────────────────────────────────
function getClasses() {
  return [...document.querySelectorAll('.cb input:checked')].map(e=>e.value);
}

// ── Start / Stop ──────────────────────────────────────────────────
async function startDetection() {
  const st = document.getElementById('sourceType').value;
  let sv = st==='camera' ? document.getElementById('camIdx').value
         : st==='rtsp'   ? document.getElementById('rtspUrl').value
         : document.getElementById('filePath').value;
  if (!sv && st!=='camera') { toast('请填写视频源','error'); return; }

  const payload = {
    source_type:st, source_value:sv||'0',
    model_size:document.getElementById('modelSize').value,
    conf_threshold:document.getElementById('confSlider').value,
    target_classes:getClasses(),
    use_gpu:document.getElementById('useGPU').checked,
  };

  toast('正在启动...','info');
  const r = await fetch('/api/start',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d = await r.json();
  if (d.success) {
    isRunning=true; setUI(true);
    if (d.device) document.getElementById('deviceInfo').textContent='设备: '+d.device;
    startPoll(); toast('检测已启动','success'); await pushZones();
  } else toast(d.message,'error');
}

async function stopDetection() {
  await fetch('/api/stop',{method:'POST'});
  isRunning=false; setUI(false); stopPoll(); toast('检测已停止','info');
}

function setUI(running) {
  document.getElementById('btnStart').disabled = running;
  document.getElementById('btnStop').disabled  = !running;
  const p = document.getElementById('statusPill');
  const t = document.getElementById('statusText');
  p.className = 'pill'+(running?' on':'');
  t.textContent = running?'检测中':'待机';
}

// ── Polling ───────────────────────────────────────────────────────
function startPoll() { pollTimer = setInterval(fetchStatus, 900); }
function stopPoll()  { clearInterval(pollTimer); pollTimer=null; }

async function fetchStatus() {
  try {
    const r = await fetch('/api/status'), d = await r.json();
    updateMetrics(d.metrics); updateAlerts(d.alerts);
    updateLightIndicator(d.light_color||'unknown');
    if (d.device) document.getElementById('deviceInfo').textContent='设备: '+d.device;
  } catch(e) {}
}

function updateLightIndicator(color) {
  const el = document.getElementById('lightIndicator');
  el.className = 'light '+(color!=='unknown'?color:'');
  document.getElementById('lightText').textContent =
    {red:'红灯',green:'绿灯',yellow:'黄灯'}[color]||'--';
}

// ── Metrics ───────────────────────────────────────────────────────
function updateMetrics(m) {
  if (!m) return;
  document.getElementById('vbFps').textContent    = m.fps;
  document.getElementById('vbObj').textContent    = m.avg_detections;
  document.getElementById('vbFrames').textContent = m.total_frames;
  document.getElementById('vbVio').textContent    = m.total_violations||0;
  document.getElementById('vbElapsed').textContent= m.elapsed;
  document.getElementById('mFps').textContent     = m.fps;
  document.getElementById('mDet').textContent     = m.avg_detections;
  document.getElementById('mFrames').textContent  = m.total_frames.toLocaleString();
  drawSparkline(m.fps_history||[]);
  updateVehicleRecords(m.current_classes||{});
}

function drawSparkline(hist) {
  if (!hist.length) return;
  const W=228, H=36, p=3, mx=Math.max(...hist,1);
  const pts = hist.map((v,i)=>{
    const x=p + i/(hist.length-1||1)*(W-2*p);
    const y=H-p-(v/mx)*(H-2*p);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  document.getElementById('fpsLine').setAttribute('points',pts);
  document.getElementById('fpsFill').setAttribute('points',
    pts+` ${(W-p).toFixed(1)},${H} ${p},${H}`);
}

function updateVehicleRecords(cc) {
  const el = document.getElementById('vehicleRecords');
  const order  = ['car','truck','bus','motorcycle','bicycle','person'];
  const labels = {car:'轿车',truck:'卡车',bus:'公交车',motorcycle:'摩托车',bicycle:'自行车',person:'行人'};
  const colors = {car:'c1',truck:'c4',bus:'c4',motorcycle:'c1',bicycle:'c1',person:'c2'};
  const units  = {person:'人'};
  const arr = order.filter(k=>(cc[k]||0)>0).map(k=>[k,cc[k]]);
  if (!arr.length) { el.innerHTML='<div style="color:var(--text3);font-size:11px">等待检测...</div>'; return; }
  el.innerHTML = arr.map(([k,v])=>`
    <div class="vrow ${colors[k]||'c1'}">
      <span>${labels[k]||k}</span>
      <span class="vc">${v} ${units[k]||'辆'}</span>
    </div>`).join('');
}

// ── Alerts ────────────────────────────────────────────────────────
let _renderedAlertIds = new Set();

function updateAlerts(alerts) {
  const src = alerts||[];
  const vis = src.filter(a=>{
    const t=String(a?.type||''), m=String(a?.message||'');
    return !HIDDEN_ALERT_KEYWORDS.some(k=>t.includes(k)||m.includes(k));
  });

  if (!vis.length) {
    if (_renderedAlertIds.size) {
      document.getElementById('alertList').innerHTML='<div style="color:var(--text3);font-size:11px">暂无违规</div>';
      document.getElementById('alertBadge').style.display='none';
      _renderedAlertIds.clear();
    }
    return;
  }

  const currentIds = new Set(vis.map(a=>a.id));
  const hasNew = currentIds.size !== _renderedAlertIds.size ||
                 [...currentIds].some(id=>!_renderedAlertIds.has(id));
  if (!hasNew) return;

  _renderedAlertIds = currentIds;
  seenAlerts = new Set([...seenAlerts, ...currentIds]);
  const badge = document.getElementById('alertBadge');
  badge.style.display=''; badge.textContent=seenAlerts.size;
  document.getElementById('alertList').innerHTML=vis.map(a=>`
    <div class="acard ${a.level}">
      <div class="at"><span class="att">${a.type}</span><span class="atm">${a.time}</span></div>
      <div class="am">${a.message}</div>
    </div>`).join('');
}

// ── Snapshot / Clear ──────────────────────────────────────────────
async function takeSnapshot() {
  const r=await fetch('/api/snapshot',{method:'POST'}), d=await r.json();
  d.success?toast('截图已保存: '+d.filename,'success'):toast(d.message,'error');
}
async function clearAlerts() {
  await fetch('/api/alerts/clear',{method:'POST'});
  document.getElementById('alertList').innerHTML='<div style="color:var(--text3);font-size:11px">暂无报警</div>';
  document.getElementById('alertBadge').style.display='none'; seenAlerts.clear(); _renderedAlertIds.clear(); toast('已清空','info');
}

// ══════════════════════════════════════════════════════════════════
// Zone Editor — 坐标归一化存储，不受窗口大小影响
// ══════════════════════════════════════════════════════════════════

function resizeCanvas() {
  canvas.width  = videoWrap.offsetWidth;
  canvas.height = videoWrap.offsetHeight;
}

function getVideoRenderRect() {
  const cw=canvas.width||1, ch=canvas.height||1;
  const nw=videoEl.naturalWidth||cw, nh=videoEl.naturalHeight||ch;
  const vr=nw/Math.max(nh,1), cr=cw/Math.max(ch,1);
  let w,h,x,y;
  if (cr>vr) { h=ch; w=h*vr; x=(cw-w)/2; y=0; }
  else       { w=cw; h=w/vr; x=0; y=(ch-h)/2; }
  return {x,y,w,h,nw,nh};
}

// 画布像素 → 归一化坐标 (0~1，相对于视频画面区域)
function toNorm(cx, cy) {
  const r = getVideoRenderRect();
  return [(cx-r.x)/(r.w||1), (cy-r.y)/(r.h||1)];
}

// 归一化坐标 → 当前画布像素
function toCanvas(np) {
  const r = getVideoRenderRect();
  return [r.x+np[0]*r.w, r.y+np[1]*r.h];
}

function setZoneTool(tool) {
  zoneTool=tool; drawingPts=[];
  ['stop','cross','light'].forEach(k=>document.getElementById('zb-'+k).classList.remove('active'));
  const m={stop_line:'stop',crosswalk:'cross',traffic_light_roi:'light'};
  if (tool&&m[tool]) document.getElementById('zb-'+m[tool]).classList.add('active');
  document.getElementById('videoPanel').classList.toggle('zone-editor-active',!!tool);
  resizeCanvas(); redrawCanvas();
  const h={stop_line:'点击第一个端点，再点击第二个端点',crosswalk:'单击添加多边形顶点，双击闭合',traffic_light_roi:'单击圈出红绿灯区域，双击完成'};
  document.getElementById('zoneHint').textContent=tool?h[tool]:'选择类型后在画面上点击绘制';
}

function onCanvasClick(e) {
  if (!zoneTool) return;
  const r=canvas.getBoundingClientRect();
  const cx=Math.round(e.clientX-r.left), cy=Math.round(e.clientY-r.top);
  const vr=getVideoRenderRect();
  if (cx<vr.x||cx>vr.x+vr.w||cy<vr.y||cy>vr.y+vr.h) return;

  if (zoneTool==='stop_line') {
    drawingPts.push([cx,cy]);
    if (drawingPts.length>=2) {
      zones.push({type:'stop_line', pts:[toNorm(...drawingPts[0]), toNorm(...drawingPts[1])]});
      drawingPts=[]; renderZoneList();
    }
    redrawCanvas(); return;
  }
  drawingPts.push([cx,cy]); redrawCanvas();
}

function onCanvasMove(e) {
  if (!zoneTool||drawingPts.length===0) return;
  const r=canvas.getBoundingClientRect(), vr=getVideoRenderRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  redrawCanvas(Math.max(vr.x,Math.min(vr.x+vr.w,mx)), Math.max(vr.y,Math.min(vr.y+vr.h,my)));
}

function onCanvasDblClick(e) {
  if (!zoneTool||zoneTool==='stop_line') return;
  if (drawingPts.length<2) { drawingPts=[]; return; }
  zones.push({type:zoneTool, pts:drawingPts.map(p=>toNorm(p[0],p[1]))});
  drawingPts=[]; renderZoneList(); redrawCanvas();
}

function redrawCanvas(mx, my) {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const colors={stop_line:'#ef4444',crosswalk:'#22c55e',traffic_light_roi:'#f59e0b'};

  zones.forEach(z=>{
    ctx.strokeStyle=colors[z.type]||'#888'; ctx.lineWidth=2; ctx.setLineDash([]);
    if (z.type==='stop_line') {
      if (z.pts&&z.pts.length>=2) {
        const a=toCanvas(z.pts[0]), b=toCanvas(z.pts[1]);
        ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
        ctx.fillStyle=colors[z.type]; ctx.font='11px Inter,sans-serif';
        ctx.fillText('停止线',a[0]+6,a[1]-6);
      }
    } else if (z.pts&&z.pts.length) {
      const cp=z.pts.map(toCanvas);
      ctx.fillStyle=colors[z.type]+'28'; ctx.beginPath();
      ctx.moveTo(cp[0][0],cp[0][1]); cp.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
  });

  if (drawingPts.length) {
    ctx.strokeStyle=colors[zoneTool]||'#888'; ctx.lineWidth=2; ctx.setLineDash([5,3]);
    if (zoneTool==='stop_line') {
      ctx.beginPath(); ctx.moveTo(drawingPts[0][0],drawingPts[0][1]);
      if (mx!==undefined) ctx.lineTo(mx,my); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle=colors[zoneTool];
      ctx.beginPath(); ctx.arc(drawingPts[0][0],drawingPts[0][1],4,0,Math.PI*2); ctx.fill();
    } else {
      ctx.fillStyle=(colors[zoneTool]||'#888')+'18'; ctx.beginPath();
      ctx.moveTo(drawingPts[0][0],drawingPts[0][1]);
      drawingPts.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));
      if (mx!==undefined) ctx.lineTo(mx,my);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      drawingPts.forEach(p=>{
        ctx.fillStyle=colors[zoneTool]; ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(p[0],p[1],4,0,Math.PI*2); ctx.fill();
      });
    }
  }
}

function renderZoneList() {
  const el=document.getElementById('zoneList');
  const m={stop_line:'停止线',crosswalk:'斑马线',traffic_light_roi:'红绿灯区域'};
  if (!zones.length) { el.innerHTML='<div style="color:var(--text3)">尚未配置区域</div>'; return; }
  el.innerHTML=zones.map((z,i)=>`
    <div class="zent"><span>${m[z.type]||z.type} #${i+1}</span>
    <button class="zdel" onclick="removeZone(${i})">×</button></div>`).join('');
}

function removeZone(i) { zones.splice(i,1); renderZoneList(); redrawCanvas(); }

function clearZones() {
  zones=[]; drawingPts=[]; renderZoneList(); redrawCanvas();
  fetch('/api/set_zones',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({zones:[]})});
  toast('已清除所有区域','info');
}

async function pushZones() {
  /*
  将当前 zones 中的归一化坐标转换为视频原始分辨率的像素坐标，并发送到后端保存。
  */
  const fw=Math.max(1,Math.round(videoEl.naturalWidth||640));
  const fh=Math.max(1,Math.round(videoEl.naturalHeight||480));
  const scaled=zones.map(z=>({
    ...z,
    pts: z.pts?z.pts.map(np=>[
      Math.round(Math.max(0,Math.min(fw-1,np[0]*fw))),
      Math.round(Math.max(0,Math.min(fh-1,np[1]*fh)))
    ]):undefined
  }));
  await fetch('/api/set_zones',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({zones:scaled})});
  toast(`已应用 ${zones.length} 个检测区域`,'success');
}

// ── Toast ─────────────────────────────────────────────────────────
function toast(msg, type='info') {
  const c=document.getElementById('toasts'), t=document.createElement('div');
  const ico={success:'✓',error:'✕',info:'ℹ'};
  t.className=`toast ${type}`;
  t.innerHTML=`<span>${ico[type]||'●'}</span><span>${msg}</span>`;
  c.prepend(t); setTimeout(()=>t.remove(),3000);
}
