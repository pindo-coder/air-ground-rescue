"""Interactive Chinese Command & Telemetry Dashboard for 《援来有路》 Air-Ground Rescue.

Supports two modes:
1. Live local web server (browse & switch across all runs in `runs/`, play MP4 videos):
     python tools/rescue_dashboard.py --port 8765
2. Self-contained offline HTML export (zero server required, ideal for competition defense):
     python tools/rescue_dashboard.py --run runs/<run_id> --export-static runs/command_dashboard.html
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import argparse
import base64
import csv
import io
import json
import sys
import webbrowser

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from safety_metrics import compute_run_safety_smoothness, trajectory_velocity_xy


def _encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_file_b64(path: Path, max_w: int = 420) -> str | None:
    if not path.is_file():
        return None
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, int(round(im.height * max_w / im.width))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=86)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _build_map_layers(run_dir: Path) -> dict:
    map_path = run_dir / "uav_map.npz"
    if not map_path.is_file():
        return {}
    with np.load(map_path, allow_pickle=False) as data:
        states = data["states"]
        origin = data["origin_xy_m"].astype(float).tolist()
        res = float(data["resolution_m"])
        slope = data["slope_deg"].astype(float) if "slope_deg" in data else None

    h, w = states.shape
    # Layer 1: Traversability classification (UNKNOWN / FREE / BLOCKED)
    trav_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    trav_rgb[states == -1] = (48, 58, 72)
    trav_rgb[states == 0] = (46, 118, 92)
    trav_rgb[states == 1] = (205, 76, 62)
    trav_img = Image.fromarray(np.flipud(trav_rgb)).resize((w * 4, h * 4), Image.Resampling.NEAREST)

    # Layer 2: VCTP Slope / Risk heatmap
    risk_rgb = np.full((h, w, 3), (36, 44, 56), dtype=np.uint8)
    if slope is not None:
        valid = np.isfinite(slope) & (states == 0)
        norm = np.clip(np.nan_to_num(slope, nan=0.0) / 18.0, 0.0, 1.0)
        # Emerald -> Amber -> Coral ramp
        r = np.clip(35 + 220 * norm, 0, 255).astype(np.uint8)
        g = np.clip(185 - 110 * np.maximum(0.0, norm - 0.45) * 1.8, 40, 220).astype(np.uint8)
        b = np.clip(130 - 110 * norm, 20, 160).astype(np.uint8)
        risk_rgb[valid] = np.stack([r[valid], g[valid], b[valid]], axis=-1)
    risk_rgb[states == 1] = (190, 52, 58)
    risk_img = Image.fromarray(np.flipud(risk_rgb)).resize((w * 4, h * 4), Image.Resampling.NEAREST)

    return dict(
        origin=origin,
        resolution=res,
        rows=int(h),
        cols=int(w),
        extent=[round(w * res, 3), round(h * res, 3)],
        trav_png=_encode_png_b64(trav_img),
        risk_png=_encode_png_b64(risk_img),
    )


def _load_run_payload(run_dir: Path, include_video_url: bool = True) -> dict:
    run_dir = Path(run_dir)
    eval_path = run_dir / "evaluation/summary.json"
    nav_path = run_dir / "rover/nav_status.json"
    if not nav_path.is_file():
        nav_path = run_dir / "rover/rescue_status.json"
    victims_path = run_dir / "aerial_victims.json"
    traj_path = run_dir / "evaluation/trajectory.csv"

    eval_sum = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else {}
    nav_st = json.loads(nav_path.read_text(encoding="utf-8")) if nav_path.is_file() else {}
    victims_doc = json.loads(victims_path.read_text(encoding="utf-8")) if victims_path.is_file() else {"candidates": []}

    traj_xy, t_list, v_list, roll_list, pitch_list, loc_list = [], [], [], [], [], []
    if traj_path.is_file():
        with traj_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        stride = max(1, len(rows) // 360)
        speeds = [float(row["speed_mps"]) for row in rows] if rows and "speed_mps" in rows[0] else []
        if rows and not speeds:
            time_s = np.array([float(row["time_s"]) for row in rows])
            positions = np.array([[float(row["x"]), float(row["y"])] for row in rows])
            speeds = np.linalg.norm(trajectory_velocity_xy(time_s, positions), axis=1).tolist()
        for r, speed in zip(rows[::stride], speeds[::stride]):
            traj_xy.append([round(float(r["x"]), 3), round(float(r["y"]), 3)])
            t_list.append(round(float(r["time_s"]), 2))
            v_list.append(round(speed, 3))
            roll_list.append(round(float(r["roll_deg"]), 2))
            pitch_list.append(round(float(r["pitch_deg"]), 2))
            loc_list.append(round(float(r["loc_error_m"]), 3) if r.get("loc_error_m") else None)

    safety = eval_sum.get("safety_smoothness") or compute_run_safety_smoothness(run_dir)
    local_policy = nav_st.get("local_policy_stats") or nav_st.get("local_policy")
    routes_history = nav_st.get("routes_history") or []
    if not routes_history:
        for route_path in (run_dir / "rover/route.json", run_dir / "route.json"):
            if not route_path.is_file():
                continue
            route_doc = json.loads(route_path.read_text(encoding="utf-8"))
            route = route_doc.get("route_xy_m")
            if not route and route_doc.get("selected_goal_id") is not None:
                selected = next((r for r in route_doc.get("routes", []) if r.get("id") == route_doc["selected_goal_id"]), {})
                route = selected.get("route_xy_m")
            if route:
                routes_history = [route]
                break
    confirm_imgs = []
    for cp in sorted((run_dir / "rover").glob("victim_*_confirm.jpg")):
        b64 = _encode_file_b64(cp, max_w=380)
        if b64:
            confirm_imgs.append(dict(name=cp.name, data=b64))

    kf_b64 = None
    for cand_kf in (
        run_dir / "evaluation/full_video_keyframes_cn.jpg",
        run_dir / "evaluation/twin_crate_release_showcase_cn.jpg",
        run_dir / "evaluation/final_scene.png",
        run_dir / "evaluation/rescue_preview.png",
        run_dir / "rover/chase_cam.jpg",
    ):
        if cand_kf.is_file():
            kf_b64 = _encode_file_b64(cand_kf, max_w=720)
            break
    if kf_b64 is None:
        for folder in ("chase_frames", "chase_cam"):
            chase_frame = max((run_dir / "rover" / folder).glob("*.jpg"), default=None)
            if chase_frame is not None:
                kf_b64 = _encode_file_b64(chase_frame, max_w=720)
                break

    video_rel = None
    if include_video_url and (run_dir / "evaluation/rescue_simulation.mp4").is_file():
        video_rel = f"/media/{run_dir.name}/evaluation/rescue_simulation.mp4"

    return dict(
        run_id=run_dir.name,
        evaluation=eval_sum,
        local_policy=local_policy if isinstance(local_policy, dict) else {},
        routes_history=routes_history,
        aerial_candidates=victims_doc.get("candidates") or [],
        map_layers=_build_map_layers(run_dir),
        telemetry=dict(t=t_list, xy=traj_xy, v=v_list, roll=roll_list, pitch=pitch_list, loc_err=loc_list),
        safety_smoothness=safety,
        confirm_images=confirm_imgs,
        preview_image=kf_b64,
        video_url=video_rel,
    )


def discover_runs(limit: int = 25) -> list[dict]:
    runs_dir = ROOT / "runs"
    if not runs_dir.is_dir():
        return []
    found = []
    for p in sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        ev = p / "evaluation/summary.json"
        if not ev.is_file():
            continue
        try:
            doc = json.loads(ev.read_text(encoding="utf-8"))
        except Exception:
            continue
        found.append(dict(
            run_id=p.name,
            mode=doc.get("mode", "nav"),
            status=doc.get("status", "UNKNOWN"),
            planner=doc.get("planner", ""),
            detector=doc.get("detector", ""),
            has_video=(p / "evaluation/rescue_simulation.mp4").is_file(),
            collisions=doc.get("collisions"),
            delivered=doc.get("delivered_people"),
        ))
        if len(found) >= limit:
            break
    return found


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>援来有路 · 空地协同应急救援自主规划与指挥调度系统</title>
<style>
  :root {
    --bg:#09111a; --card:#101c2b; --card2:#152436; --border:#243952;
    --text:#e8f2fa; --muted:#8fa7be; --cyan:#38bdf8; --teal:#2dd4bf;
    --orange:#fb923c; --red:#f87171; --amber:#fbbf24;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    background:radial-gradient(circle at 20% 0%,#13263d 0%,var(--bg) 55%);
    color:var(--text); font-family:"Microsoft YaHei","PingFang SC",sans-serif;
    min-height:100vh; padding:16px 20px 26px;
  }
  .topbar {
    display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px;
    background:linear-gradient(90deg,#112236,#0e1b2a); border:1px solid var(--border);
    border-radius:12px; padding:14px 20px; margin-bottom:14px;
    box-shadow:0 8px 28px rgba(0,0,0,.35);
  }
  .brand h1 { font-size:21px; letter-spacing:.5px; display:flex; align-items:center; gap:10px; }
  .brand .sub { font-size:12.5px; color:var(--muted); margin-top:4px; }
  .pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700; }
  .pill.ok { background:rgba(45,212,191,.16); color:var(--teal); border:1px solid rgba(45,212,191,.45); }
  .pill.warn { background:rgba(251,191,36,.16); color:var(--amber); border:1px solid rgba(251,191,36,.45); }
  .pill.bad { background:rgba(248,113,113,.16); color:var(--red); border:1px solid rgba(248,113,113,.45); }
  .controls { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  select, button {
    background:var(--card2); color:var(--text); border:1px solid var(--border);
    border-radius:8px; padding:7px 12px; font-size:13px; font-family:inherit; cursor:pointer;
  }
  button.active { background:rgba(56,189,248,.18); border-color:var(--cyan); color:var(--cyan); font-weight:700; }
  .kpis { display:grid; grid-template-columns:repeat(6,minmax(140px,1fr)); gap:12px; margin-bottom:14px; }
  @media (max-width:1200px){ .kpis { grid-template-columns:repeat(3,1fr); } }
  .kpi {
    background:var(--card); border:1px solid var(--border); border-radius:10px;
    padding:12px 14px; position:relative; overflow:hidden;
  }
  .kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--cyan); }
  .kpi.teal::before { background:var(--teal); }
  .kpi.orange::before { background:var(--orange); }
  .kpi .lbl { font-size:12px; color:var(--muted); }
  .kpi .val { font-size:22px; font-weight:700; margin-top:4px; }
  .kpi .hint { font-size:11.5px; color:var(--muted); margin-top:3px; }
  .grid-main { display:grid; grid-template-columns:1.22fr 1fr; gap:14px; margin-bottom:14px; }
  @media (max-width:1100px){ .grid-main { grid-template-columns:1fr; } }
  .panel {
    background:var(--card); border:1px solid var(--border); border-radius:12px;
    padding:14px 16px; display:flex; flex-direction:column; gap:10px;
  }
  .panel-hd { display:flex; align-items:center; justify-content:space-between; gap:8px; }
  .panel-hd h2 { font-size:15.5px; color:#dcebf7; }
  #mapCanvas { width:100%; height:410px; background:#070d15; border-radius:8px; border:1px solid #1d3147; display:block; }
  .legend { display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--muted); }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .dot { width:11px; height:11px; border-radius:3px; display:inline-block; }
  video, .preview-img { width:100%; max-height:275px; border-radius:8px; border:1px solid #1d3147; background:#050a10; object-fit:contain; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th, td { padding:7px 8px; border-bottom:1px solid #1c2e43; text-align:left; }
  th { color:var(--muted); font-weight:600; background:rgba(21,36,54,.65); }
  .thumbs { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:8px; }
  .thumbs figure { background:#0c1622; border:1px solid #1d3147; border-radius:8px; overflow:hidden; }
  .thumbs img { width:100%; height:110px; object-fit:cover; display:block; }
  .thumbs figcaption { font-size:11px; color:var(--muted); padding:4px 7px; }
  .charts { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
  @media (max-width:1050px){ .charts { grid-template-columns:1fr; } }
  .chart-canvas { width:100%; height:180px; background:#081019; border-radius:8px; border:1px solid #1c2f44; display:block; }
</style>
</head>
<body>
  <div class="topbar">
    <div class="brand">
      <h1>援来有路 · 空地协同应急救援指挥与遥测大屏 <span id="statusPill" class="pill ok">加载中...</span></h1>
      <div class="sub">无人机先验建图（DUCPP）· 全色域纯 YOLO 空中搜寻 · VCTP + IGHA* 三维地形规划 · 神经 DAgger 毫秒局规</div>
    </div>
    <div class="controls">
      <select id="runSelect" style="min-width:290px;"></select>
      <button id="btnTrav" class="active" onclick="setLayer('trav')">通行栅格图</button>
      <button id="btnRisk" onclick="setLayer('risk')">VCTP 坡度热力图</button>
    </div>
  </div>

  <div class="kpis" id="kpiRow"></div>

  <div class="grid-main">
    <div class="panel">
      <div class="panel-hd">
        <h2>空地协同态势总图（无人机先验层 + 全局规划 + 实车轨迹 + 伤员投送点）</h2>
        <span id="mapMeta" style="font-size:12px;color:var(--muted);"></span>
      </div>
      <canvas id="mapCanvas" width="760" height="440"></canvas>
      <div class="legend">
        <span><i class="dot" style="background:#2e765c"></i>可通行地面</span>
        <span><i class="dot" style="background:#cd4c3e"></i>障碍/陡坡</span>
        <span><i class="dot" style="background:#303a48"></i>未确认遮挡</span>
        <span><i class="dot" style="background:#38bdf8"></i>VCTP+IGHA* 规划路</span>
        <span><i class="dot" style="background:#ffffff"></i>实车真实轨迹</span>
        <span><i class="dot" style="background:#fb923c;border-radius:50%"></i>空中 YOLO 伤员候选</span>
        <span><i class="dot" style="background:#2dd4bf;border-radius:50%"></i>物资有效投送点</span>
      </div>
    </div>

    <div class="panel">
      <div class="panel-hd">
        <h2>全程实录画面与车载抵近复核监测</h2>
        <span id="policyTag" style="font-size:12px;color:var(--cyan);"></span>
      </div>
      <div id="mediaBox"></div>
      <div id="confirmBox" class="thumbs"></div>
      <div style="overflow-x:auto;">
        <table>
          <thead><tr><th>伤员/目标</th><th>空中误差</th><th>融合定位误差</th><th>车载置信度</th><th>投送距离</th><th>状态</th></tr></thead>
          <tbody id="victimTable"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="charts">
    <div class="panel">
      <div class="panel-hd"><h2>无人车行驶速度曲线 v(t) [m/s]</h2><span id="vStat" style="font-size:12px;color:var(--muted);"></span></div>
      <canvas id="chartSpeed" class="chart-canvas" width="440" height="185"></canvas>
    </div>
    <div class="panel">
      <div class="panel-hd"><h2>车身三维姿态角（横滚 Roll / 俯仰 Pitch）[°]</h2><span id="attStat" style="font-size:12px;color:var(--muted);"></span></div>
      <canvas id="chartAtt" class="chart-canvas" width="440" height="185"></canvas>
    </div>
    <div class="panel">
      <div class="panel-hd"><h2>车载融合定位误差曲线 [m]</h2><span id="locStat" style="font-size:12px;color:var(--muted);"></span></div>
      <canvas id="chartLoc" class="chart-canvas" width="440" height="185"></canvas>
    </div>
  </div>

<script>
const STATIC_PAYLOAD = __STATIC_PAYLOAD__;
const RUNS_INDEX = __RUNS_INDEX__;
let currentData = null;
let currentLayer = 'trav';
const imgCache = {};

function fmt(v, d=2, suf='') { return (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(d) + suf; }

function setLayer(layer) {
  currentLayer = layer;
  document.getElementById('btnTrav').classList.toggle('active', layer === 'trav');
  document.getElementById('btnRisk').classList.toggle('active', layer === 'risk');
  if (currentData) drawMap(currentData);
}

function renderKPIs(d) {
  const ev = d.evaluation || {};
  const lp = d.local_policy || {};
  const isNeural = lp.policy === 'neural' || ev.local_policy === 'neural';
  const dec = lp.decisions || {};
  const totDec = (dec.learner || 0) + (dec.safety_fallback || 0);
  const neuralPct = totDec > 0 ? (100 * (dec.learner || 0) / totDec) : 0;
  const p95 = (lp.model_inference_ms || {}).p95;
  const sm = d.safety_smoothness || {};

  const pill = document.getElementById('statusPill');
  const st = ev.status || 'UNKNOWN';
  const isOk = st === 'MISSION_SUCCESS' || st === 'ARRIVED';
  pill.className = 'pill ' + (isOk ? 'ok' : (st === 'STOPPED_SAFE' ? 'warn' : 'bad'));
  pill.textContent = (isOk ? '任务成功 · ' : '') + st + ` · 碰撞 ${ev.collisions ?? 0}`;

  const kpis = [
    { cls:'teal', lbl:'任务与投送进度', val: ev.mode === 'rescue' ? `${ev.delivered_people ?? 0}/${ev.cargo_capacity ?? ev.target_deliveries ?? 2} 投送` : ((ev.arrived || st === 'ARRIVED') ? '到达停稳' : st),
      hint: ev.mode === 'rescue' ? `空中检出 ${ev.aerial_candidates_total ?? (d.aerial_candidates || []).length} 名 · 跨视角复核 ${ev.valid_cross_view_reports ?? 0} 名` : `终点误差 ${fmt(ev.final_goal_error_m,3,' m')}` },
    { cls:'', lbl:'空地融合定位 RMSE', val: ev.localization_rmse_m ? fmt(ev.localization_rmse_m,3,' m') : fmt((ev.rover_pose_error_m||ev.localization_error_m||{}).median,3,' m'),
      hint:`误报数 ${ev.false_reports ?? ((ev.total_reports ?? 0) - (ev.effective_reports ?? 0))} · 重规划 ${ev.replan_count ?? 0} 次` },
    { cls:'', lbl:'行驶路程与耗时', val: fmt(ev.actual_path_length_m ?? ev.path_length_m,2,' m'),
      hint:`作业耗时 ${fmt(ev.total_rescue_duration_s ?? ev.travel_duration_s,1,' s')}` },
    { cls:'orange', lbl:'神经网络局规占比', val: totDec > 0 ? fmt(neuralPct,1,'%') : (isNeural ? '—' : '规则基线'),
      hint: totDec > 0 ? `P95 推理 ${fmt(p95,2,' ms')} · 回退 ${dec.safety_fallback||0} 步` : (isNeural ? '暂无推理统计' : '未启用神经网络策略') },
    { cls:'teal', lbl:'最小安全裕度 S_margin', val: fmt(sm.min_safety_margin_m,3,' m'),
      hint:`全程碰撞 ${ev.collisions ?? 0} 次 · 底盘触碰 ${ev.chassis_contacts ?? 0} 次` },
    { cls:'', lbl:'避障平滑度 J_avoid', val: sm.j_avoid_integral !== undefined ? fmt(sm.j_avoid_integral,0) : '—',
      hint:`RMS 加加速度 ${fmt(sm.rms_jerk_mps3,2,' m/s³')}` },
  ];
  document.getElementById('kpiRow').innerHTML = kpis.map(k =>
    `<div class="kpi ${k.cls}"><div class="lbl">${k.lbl}</div><div class="val">${k.val}</div><div class="hint">${k.hint}</div></div>`
  ).join('');
  document.getElementById('policyTag').textContent =
    `检测：${ev.detector || '—'} | 全局：${ev.planner || '—'} | 局规：${isNeural ? '神经 DAgger' : '规则控制'}`;
}

function drawMap(d) {
  const cv = document.getElementById('mapCanvas');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  const ml = d.map_layers || {};
  if (!ml.trav_png) {
    ctx.fillStyle = '#8fa7be'; ctx.font = '15px Microsoft YaHei';
    ctx.fillText('当前运行目录无 uav_map.npz 图层', 24, 40);
    return;
  }
  document.getElementById('mapMeta').textContent = `网格 ${ml.cols}×${ml.rows} (${ml.resolution} m) · 范围 ${ml.extent[0]}×${ml.extent[1]} m`;
  const src = currentLayer === 'risk' ? ml.risk_png : ml.trav_png;
  const renderOverlay = (img) => {
    const pad = 16;
    const scale = Math.min((cv.width - 2*pad)/img.width, (cv.height - 2*pad)/img.height);
    const dw = img.width * scale, dh = img.height * scale;
    const ox = (cv.width - dw)/2, oy = (cv.height - dh)/2;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(img, ox, oy, dw, dh);
    const toPx = (xy) => [
      ox + ((xy[0] - ml.origin[0]) / ml.extent[0]) * dw,
      oy + (1.0 - (xy[1] - ml.origin[1]) / ml.extent[1]) * dh,
    ];
    // Planned routes
    ctx.lineWidth = 2.6; ctx.strokeStyle = '#38bdf8';
    for (const r of (d.routes_history || [])) {
      if (r.length < 2) continue;
      ctx.beginPath();
      r.forEach((pt, idx) => { const [x,y] = toPx(pt); idx===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y); });
      ctx.stroke();
    }
    // Actual trajectory
    const txy = (d.telemetry || {}).xy || [];
    if (txy.length >= 2) {
      ctx.lineWidth = 2.4; ctx.strokeStyle = '#ffffff';
      ctx.beginPath();
      txy.forEach((pt, idx) => { const [x,y] = toPx(pt); idx===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y); });
      ctx.stroke();
    }
    // Aerial candidates
    for (const c of (d.aerial_candidates || [])) {
      const [x,y] = toPx(c.mean_enu_m);
      ctx.fillStyle = '#fb923c'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(x, y, 6.5, 0, Math.PI*2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 11px Microsoft YaHei';
      ctx.fillText('C' + c.id, x + 9, y + 4);
    }
    // Deliveries
    for (const dl of ((d.evaluation||{}).deliveries || [])) {
      if (!dl.actual_drop_xy) continue;
      const [x,y] = toPx(dl.actual_drop_xy);
      ctx.fillStyle = '#2dd4bf'; ctx.strokeStyle = '#07131e'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 7.5, 0, Math.PI*2); ctx.fill(); ctx.stroke();
    }
  };
  if (imgCache[src]) { renderOverlay(imgCache[src]); return; }
  const im = new Image();
  im.onload = () => { imgCache[src] = im; renderOverlay(im); };
  im.src = src;
}

function drawLineChart(canvasId, tArr, seriesList, yUnit='') {
  const cv = document.getElementById(canvasId);
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0,0,W,H);
  if (!tArr || tArr.length < 2) return;
  const padL=44, padR=14, padT=16, padB=26;
  const vals = seriesList.flatMap(s => s.data.filter(v => v !== null && v !== undefined));
  if (!vals.length) return;
  let ymin = Math.min(0, ...vals), ymax = Math.max(...vals);
  if (ymax - ymin < 1e-4) { ymax += 1; ymin -= 1; }
  const t0 = tArr[0], t1 = tArr[tArr.length - 1];
  const tx = t => padL + ((t - t0) / Math.max(1e-3, t1 - t0)) * (W - padL - padR);
  const ty = v => H - padB - ((v - ymin) / (ymax - ymin)) * (H - padT - padB);
  // Grid
  ctx.strokeStyle = '#1a2d42'; ctx.lineWidth = 1; ctx.fillStyle = '#7e99b3'; ctx.font = '11px Microsoft YaHei';
  for (let i=0; i<=4; i++) {
    const v = ymin + (ymax - ymin) * i / 4;
    const y = ty(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(v.toFixed(1) + yUnit, 4, y + 4);
  }
  ctx.fillText(t0.toFixed(0)+'s', padL, H - 7);
  ctx.fillText(t1.toFixed(0)+'s', W - padR - 30, H - 7);
  for (const s of seriesList) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.beginPath(); let started = false;
    s.data.forEach((v, idx) => {
      if (v === null || v === undefined) { started = false; return; }
      const x = tx(tArr[idx]), y = ty(v);
      if (!started) { ctx.moveTo(x,y); started = true; } else { ctx.lineTo(x,y); }
    });
    ctx.stroke();
  }
}

function renderMediaAndTable(d) {
  const mediaBox = document.getElementById('mediaBox');
  if (d.video_url) {
    mediaBox.innerHTML = `<video controls preload="metadata" src="${d.video_url}"></video>`;
  } else if (d.preview_image) {
    mediaBox.innerHTML = `<img class="preview-img" src="${d.preview_image}" alt="实拍预览" />`;
  } else {
    mediaBox.innerHTML = `<div style="padding:26px;text-align:center;color:var(--muted);background:#09121d;border-radius:8px;">当前运行未录制视频（可切换至含视频的实录任务）</div>`;
  }
  document.getElementById('confirmBox').innerHTML = (d.confirm_images || []).map(im =>
    `<figure><img src="${im.data}" /><figcaption>车载前视 YOLO 抵近复核：${im.name}</figcaption></figure>`
  ).join('');

  const ev = d.evaluation || {};
  const legs = ev.rover_legs || [];
  const dels = ev.deliveries?.length ? ev.deliveries : (ev.delivery_checks || []).map(check => ({
    track_id: check.victim_id, true_distance_m: check.distance_m, valid: check.valid,
  }));
  const audits = ev.aerial_candidates_audit || [];
  const tbody = document.getElementById('victimTable');
  if (!audits.length && !dels.length) {
    const message = ev.mode === 'rescue' ? '暂无伤员复核或投送记录' :
      `定点闭环导航任务（目标：${ev.goal_label || ev.goal_id || '—'}，终点误差 ${fmt(ev.final_goal_error_m,3,' m')}）`;
    tbody.innerHTML = `<tr><td colspan="6" style="color:var(--muted);text-align:center;">${message}</td></tr>`;
    return;
  }
  const auditRows = audits.map(a => {
    const leg = legs.find(l => l.victim_candidate_id === a.candidate_id) || {};
    const dl = dels.find(x => x.track_id === a.candidate_id);
    const idx = dels.indexOf(dl);
    const locErr = (idx >= 0 && ev.localization_errors_m) ? ev.localization_errors_m[idx] : null;
    return `<tr>
      <td>候选 #${a.candidate_id}（伤员 ${a.nearest_truth_victim_id}）</td>
      <td>${fmt(a.error_m,3,' m')} (${a.views}视角)</td>
      <td>${fmt(locErr,3,' m')}</td>
      <td>${leg.onboard_score ? fmt(leg.onboard_score,3) : '—'}</td>
      <td>${dl ? fmt(dl.true_distance_m,2,' m') : '—'}</td>
      <td>${dl && dl.valid ? '<span class="pill ok">已有效投送</span>' : '<span class="pill warn">空中已定位</span>'}</td>
    </tr>`;
  });
  const deliveryRows = dels.filter(dl => !audits.some(a => a.candidate_id === dl.track_id)).map(dl => `
    <tr><td>候选 #${dl.track_id ?? '—'}</td><td>—</td><td>—</td><td>—</td>
    <td>${fmt(dl.true_distance_m,2,' m')}</td>
    <td>${dl.valid ? '<span class="pill ok">已有效投送</span>' : '<span class="pill warn">投送未通过核验</span>'}</td></tr>`);
  tbody.innerHTML = [...auditRows, ...deliveryRows].join('');
}

function renderAll(d) {
  currentData = d;
  renderKPIs(d);
  drawMap(d);
  renderMediaAndTable(d);
  const tel = d.telemetry || {};
  const ev = d.evaluation || {};
  drawLineChart('chartSpeed', tel.t, [{color:'#38bdf8', data:tel.v}], '');
  drawLineChart('chartAtt', tel.t, [{color:'#fb923c', data:tel.roll}, {color:'#2dd4bf', data:tel.pitch}], '°');
  drawLineChart('chartLoc', tel.t, [{color:'#fbbf24', data:tel.loc_err}], '');
  document.getElementById('vStat').textContent = `均速 ${fmt((d.safety_smoothness||{}).mean_speed_mps,2,' m/s')}`;
  document.getElementById('attStat').textContent = `最大横滚/俯仰 ${fmt(ev.max_abs_roll_deg,1,'°')} / ${fmt(ev.max_abs_pitch_deg,1,'°')}`;
  const pe = ev.rover_pose_error_m || ev.localization_error_m || {};
  document.getElementById('locStat').textContent = `中位 ${fmt(pe.median,3,' m')} · P90 ${fmt(pe.p90,3,' m')}`;
}

async function loadRun(runId) {
  const res = await fetch('/api/run?id=' + encodeURIComponent(runId));
  const data = await res.json();
  renderAll(data);
}

(function init() {
  const sel = document.getElementById('runSelect');
  if (STATIC_PAYLOAD) {
    sel.innerHTML = `<option value="${STATIC_PAYLOAD.run_id}">${STATIC_PAYLOAD.run_id}（离线快照）</option>`;
    sel.disabled = true;
    renderAll(STATIC_PAYLOAD);
    return;
  }
  sel.innerHTML = RUNS_INDEX.map(r =>
    `<option value="${r.run_id}">${r.has_video?'[视频] ':''}${r.status} | ${r.run_id}</option>`
  ).join('');
  sel.addEventListener('change', () => loadRun(sel.value));
  if (RUNS_INDEX.length) loadRun(RUNS_INDEX[0].run_id);
})();
</script>
</body>
</html>
"""


def build_html(static_payload: dict | None = None, runs_index: list[dict] | None = None) -> str:
    html = DASHBOARD_HTML.replace("__STATIC_PAYLOAD__", json.dumps(static_payload, ensure_ascii=False))
    html = html.replace("__RUNS_INDEX__", json.dumps(runs_index or [], ensure_ascii=False))
    return html


def serve_dashboard(port: int, open_browser: bool = False):
    runs_index = discover_runs()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = build_html(None, discover_runs()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/run":
                qs = parse_qs(parsed.query)
                run_id = (qs.get("id") or [""])[0]
                target = (ROOT / "runs" / run_id).resolve()
                if not target.is_dir() or target.parent != (ROOT / "runs").resolve():
                    self.send_error(404, "Run not found")
                    return
                payload = json.dumps(_load_run_payload(target, include_video_url=True), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path.startswith("/media/"):
                rel = parsed.path[len("/media/"):]
                fpath = (ROOT / "runs" / rel).resolve()
                if not fpath.is_file() or (ROOT / "runs").resolve() not in fpath.parents:
                    self.send_error(404, "Media not found")
                    return
                self.path = str(fpath)
                size = fpath.stat().st_size
                ctype = "video/mp4" if fpath.suffix.lower() == ".mp4" else "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                with fpath.open("rb") as f:
                    while True:
                        chunk = f.read(1 << 18)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                return
            self.send_error(404)

        def log_message(self, fmt_str, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"[Dashboard] Serving {len(runs_index)} run(s) at {url} (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open default browser automatically.")
    parser.add_argument("--run", type=Path, default=None, help="Run directory for --export-static.")
    parser.add_argument("--export-static", type=Path, default=None, help="Export a self-contained offline HTML dashboard.")
    args = parser.parse_args()

    if args.export_static is not None:
        run_dir = args.run
        if run_dir is None:
            runs = discover_runs(limit=1)
            if not runs:
                raise RuntimeError("No completed runs found under runs/.")
            run_dir = ROOT / "runs" / runs[0]["run_id"]
        payload = _load_run_payload(run_dir, include_video_url=False)
        out = Path(args.export_static)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_html(static_payload=payload, runs_index=[]), encoding="utf-8")
        print(f"[Dashboard] Exported standalone offline HTML: {out} ({out.stat().st_size / 1024:.1f} KB)", flush=True)
        return

    serve_dashboard(args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
