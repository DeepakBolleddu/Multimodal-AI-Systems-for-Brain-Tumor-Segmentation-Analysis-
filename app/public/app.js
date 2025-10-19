// --- imports (nifti first to avoid circular surprises) ---
import * as nifti from "https://esm.sh/nifti-reader-js@0.6.8";
import * as THREE from "three";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "https://unpkg.com/three@0.160.0/examples/jsm/loaders/OBJLoader.js";

/* === Simple hash router (#/home, #/run/<id>) ============================ */
const routes = { home: renderHome, run: renderRun };
const rootEl  = document.getElementById('route-root');
const tplHome = document.getElementById('tpl-home');
const tplRun  = document.getElementById('tpl-run');

const newRunBtn = document.getElementById('new-run-btn');
if (newRunBtn) newRunBtn.addEventListener('click', () => go('/home'));

function go(path){ if (location.hash !== '#'+path) location.hash = path; else onRoute(); }
window.addEventListener('hashchange', onRoute);

function parseHash(){
  const h = location.hash.replace(/^#\/?/, '');
  const parts = h.split('/');
  const name = parts[0] || 'home';
  const params = parts.slice(1);
  return { name, params };
}
function onRoute(){
  const { name, params } = parseHash();
  if (name in routes) routes[name](...params);
  else renderHome();
}
function mountTemplate(tpl){
  rootEl.innerHTML = '';
  const node = document.importNode(tpl.content, true);
  rootEl.appendChild(node);
}

/* === History (localStorage) ============================================= */
const HISTORY_KEY = 'tumor_runs_v1';

function loadHistory(){ try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; } }
function saveHistory(arr){ localStorage.setItem(HISTORY_KEY, JSON.stringify(arr)); }
function addHistory(entry){
  const list = loadHistory();
  list.unshift(entry);
  saveHistory(list);
  renderHistorySidebar();
}
function renderHistorySidebar(){
  const list = loadHistory();
  const emptyEl = document.getElementById('history-empty');
  const wrap = document.getElementById('history-list');
  if (!wrap) return;
  wrap.innerHTML = '';
  if (!list.length){
    if (emptyEl) emptyEl.style.display = 'block';
    return;
  }
  if (emptyEl) emptyEl.style.display = 'none';
  list.forEach(item => {
    const label = (item.title || ('Run ' + item.run_id));
    const div = document.createElement('div');
    div.className = 'history-item';
    div.innerHTML = `
      <div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"><strong title="${label}">${label}</strong></div>
      <div class="history-meta">
        <span>${new Date(item.ts).toLocaleString()}</span>
        <span>Vol: ${item.summary?.vol_cm3 ?? '—'} cm³</span>
        <span>${item.summary?.urgency ?? ''}</span>
      </div>`;
    div.addEventListener('click', () => go(`/run/${item.run_id}`));
    wrap.appendChild(div);
  });
}

/* === Global state reused across pages =================================== */
let lastUrls = { brain: null, tumor: null, mask: null };
let scene, camera, renderer, controls;
const group = new THREE.Group();

/* NIfTI state */
let bgVolume = null;   // From first uploaded file (grayscale)
let maskVolume = null; // From server mask
let dims = null;
let axialZ = 0;

/* ================= THREE setup ================= */
function initThree(container){
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 5000);
  camera.position.set(0, 0, 300);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.innerHTML = '';
  container.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 1.0);
  hemi.position.set(0, 200, 0);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 0.85);
  dir.position.set(100, 120, 80);
  scene.add(dir);

  scene.add(group);
  animate();
}
function animate(){ requestAnimationFrame(animate); controls?.update(); renderer?.render(scene, camera); }
window.addEventListener('resize', () => {
  const viewerEl = document.getElementById('viewer');
  if (!viewerEl || !renderer) return;
  camera.aspect = viewerEl.clientWidth / viewerEl.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(viewerEl.clientWidth, viewerEl.clientHeight);
});
function clearMeshes(){ while(group.children.length) group.remove(group.children[0]); }
function fitViewToObject(object3D){
  const box = new THREE.Box3().setFromObject(object3D);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI/180);
  let cameraZ = Math.abs(maxDim / (2*Math.tan(fov/2))); cameraZ *= 1.5;
  camera.position.set(center.x, center.y, cameraZ + center.z);
  camera.lookAt(center);
  controls.target.copy(center);
  controls.update();
}

/* ================= Dataset browser (home) ================= */
async function loadDataset(){
  const el = document.getElementById('dataset-status');
  const listEl = document.getElementById('patient-list');
  if (!el || !listEl) return;
  el.textContent = 'Loading /dataset ...';
  try{
    const res = await fetch('/dataset', {cache:'no-store'});
    if(!res.ok) throw new Error(await res.text());
    const data = await res.json();
    el.textContent = `Dataset: ${data.dataset_dir || 'not set'} • Patients: ${data.patient_ids.length}`;

    listEl.innerHTML = '';
    data.patient_ids.forEach(pid => {
      const pInfo = data.patients?.[pid] || {};
      const hasMask = !!pInfo.has_mask;
      const item = document.createElement('div');
      item.className = 'item';
      item.innerHTML = `
        <div>${pid}</div>
        <div>
          ${hasMask ? '<span class="badge">mask</span>' : ''}
          ${Object.entries(pInfo.modalities || {})
             .filter(([,v]) => !!v)
             .map(([k]) => `<span class="badge">${k}</span>`)
             .join(' ')}
        </div>`;
      item.addEventListener('click', () => runInferenceByPatient(pid));
      listEl.appendChild(item);
    });
  }catch(e){
    el.textContent = 'Failed to load dataset: ' + e.message;
  }
}

/* ================= Upload UI (home) ================= */
function wireUploadUI(){
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const runUploadBtn = document.getElementById('run-upload');
  const uploadStatusEl = document.getElementById('upload-status');
  const fileListEl = document.getElementById('file-list');

  function renderFileList(list){
    fileListEl.innerHTML = '';
    if(!list || list.length === 0){ runUploadBtn.disabled = true; return; }
    for(const f of list){
      const row = document.createElement('div');
      row.className = 'item';
      row.innerHTML = `<div>${f.name}</div><div class="badge">${Math.round(f.size/1024)} KB</div>`;
      fileListEl.appendChild(row);
    }
    runUploadBtn.disabled = false;
  }

  fileInput.addEventListener('change', () => renderFileList(fileInput.files));
  ['dragenter','dragover'].forEach(evt => {
    dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.style.borderColor = '#3b82f6'; });
  });
  ['dragleave','drop'].forEach(evt => {
    dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.style.borderColor = '#374151'; });
  });
  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if(files?.length){
      const dt = new DataTransfer();
      for(const f of files) dt.items.add(f);
      fileInput.files = dt.files;
      renderFileList(fileInput.files);
      uploadStatusEl.textContent = `${files.length} file(s) selected.`;
    }
  });

  runUploadBtn.addEventListener('click', async () => {
    if(!fileInput.files || fileInput.files.length === 0){
      uploadStatusEl.textContent = 'Select at least one NIfTI file.'; return;
    }
    const fd = new FormData();
    [...fileInput.files].forEach(f => fd.append('files', f, f.name));
    uploadStatusEl.textContent = 'Uploading & inferring...';
    runUploadBtn.disabled = true;

    try{
      try { bgVolume = await readLocalNiftiAsVolume(fileInput.files[0]); } catch { bgVolume = null; }
      const { run_id, data } = await runInference(fd);

      // ===== Use file names as the history label =====
      const names = [...fileInput.files].map(f => f.name);
      const label = names.length === 1 ? names[0] : names.join(', ');

      addHistory({
        run_id,
        ts: Date.now(),
        title: label,                   // <<— file name(s) shown in Past History
        summary: {
          vol_cm3: data?.metrics?.tumor_volume_mm3 ? Math.round((data.metrics.tumor_volume_mm3/1000)*100)/100 : '—',
          urgency: data?.metrics?.urgency?.urgency_level ?? ''
        }
      });
      go(`/run/${run_id}`);
    } finally {
      runUploadBtn.disabled = false;
      uploadStatusEl.textContent = '';
    }
  });
}

async function runInferenceByPatient(patientId){
  const fd = new FormData();
  const { run_id, data } = await runInference(fd, patientId);
  addHistory({
    run_id,
    ts: Date.now(),
    title: patientId, // keep patient id as label for dataset runs
    summary: {
      vol_cm3: data?.metrics?.tumor_volume_mm3 ? Math.round((data.metrics.tumor_volume_mm3/1000)*100)/100 : '—',
      urgency: data?.metrics?.urgency?.urgency_level ?? ''
    }
  });
  go(`/run/${run_id}`);
}

/* ================= Inference (shared) ================= */
async function runInference(formData, patientId=null){
  const url = patientId ? `/infer?patient_id=${encodeURIComponent(patientId)}` : '/infer';
  const res = await fetch(url, { method:'POST', body: formData, cache:'no-store' });
  if(!res.ok){
    const txt = await res.text();
    throw new Error(txt || res.statusText);
  }
  const data = await res.json();
  lastUrls.brain = data.brain_mesh_url || null;
  lastUrls.tumor = data.tumor_mesh_url || null;
  lastUrls.mask  = data.mask_url || null;
  sessionStorage.setItem(`run:${data.run_id}`, JSON.stringify(data));
  return { run_id: data.run_id, data };
}

/* ================= Metrics (results) ================= */
function urgencyBadge(level){
  const map = {
    "High urgency":   { bg: "#fee2e2", fg: "#991b1b", border: "#fecaca" },
    "Medium urgency": { bg: "#fff7ed", fg: "#9a3412", border: "#ffedd5" },
    "Low urgency":    { bg: "#ecfdf5", fg: "#065f46", border: "#d1fae5" }
  };
  const s = map[level] || { bg: "#eef2ff", fg: "#3730a3", border: "#e0e7ff" };
  return `style="background:${s.bg};color:${s.fg};border:1px solid ${s.border};padding:6px 10px;border-radius:6px;font-weight:600"`;
}
function drawMetrics(m){
  const grid = document.getElementById('metrics-grid');
  if (!grid) return;
  const rows = [
    ['Voxel volume (mm³)', fmt(m.voxel_volume_mm3)],
    ['Tumor voxels', fmt(m.tumor_voxels)],
    ['Tumor volume (mm³)', fmt(m.tumor_volume_mm3)],
    ['Surface area (mm²)', fmt(m.tumor_surface_area_mm2)],
    ['Centroid X (mm)', fmt(m.tumor_centroid_mm?.x)],
    ['Centroid Y (mm)', fmt(m.tumor_centroid_mm?.y)],
    ['Centroid Z (mm)', fmt(m.tumor_centroid_mm?.z)],
    ['BBox X min..max (mm)', rangeFmt(m.tumor_bbox_mm?.x_min, m.tumor_bbox_mm?.x_max)],
    ['BBox Y min..max (mm)', rangeFmt(m.tumor_bbox_mm?.y_min, m.tumor_bbox_mm?.y_max)],
    ['BBox Z min..max (mm)', rangeFmt(m.tumor_bbox_mm?.z_min, m.tumor_bbox_mm?.z_max)],
  ];

  const u = m?.urgency;
  let urgencyHtml = '';
  if (u) {
    const comp = u.components || {};
    const vol_cm3 = comp.volume_cm3 != null ? Number(comp.volume_cm3) : null;
    urgencyHtml = `
      <div class="metric" style="grid-column: 1 / -1; margin-top: 6px; padding: 8px 10px; border-radius: 8px; border: 1px dashed #334155;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
          <span ${urgencyBadge(u.urgency_level)}>${u.urgency_level || 'Urgency'}</span>
          <span style="font-weight:600;">Score:</span>
          <span>${fmt(u.final_score)}</span>
        </div>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:8px;">
          <div><div class="k">Sphericity</div><div class="v">${fmt(comp.sphericity)}</div></div>
          <div><div class="k">Convexity</div><div class="v">${fmt(comp.convexity)}</div></div>
          <div><div class="k">Surface area (mm²)</div><div class="v">${fmt(comp.surface_area_mm2)}</div></div>
          <div><div class="k">Volume (cm³)</div><div class="v">${vol_cm3 != null ? fmt(vol_cm3) : '—'}</div></div>
          <div><div class="k">Volume score</div><div class="v">${fmt(comp.volume_score)}</div></div>
          <div><div class="k">Shape score</div><div class="v">${fmt(comp.shape_score)}</div></div>
        </div>
      </div>`;
  }

  grid.innerHTML = [
    ...rows.map(([k,v]) => `
      <div class="metric">
        <div class="k">${k}</div>
        <div class="v">${v}</div>
      </div>`),
    urgencyHtml
  ].join('');
}
function fmt(v){ return (v===0 || v) ? String(Math.round(v*100)/100) : '—'; }
function rangeFmt(a,b){ if((a===0||a) && (b===0||b)) return `${fmt(a)} .. ${fmt(b)}`; return '—'; }

/* ================= 3D meshes (results) ================= */
async function loadMeshes(brainUrl, tumorUrl){
  const viewerEl = document.getElementById('viewer');
  initThree(viewerEl);
  clearMeshes();
  const loader = new OBJLoader();

  if(brainUrl){
    await new Promise((resolve, reject) => {
      loader.load(brainUrl, (obj) => {
        obj.traverse((child) => {
          if(child.isMesh){
            child.material = new THREE.MeshPhongMaterial({ transparent:true, opacity:0.18 });
          }
        });
        group.add(obj); resolve();
      }, undefined, reject);
    });
  }
  if(tumorUrl){
    await new Promise((resolve, reject) => {
      loader.load(tumorUrl, (obj) => {
        obj.traverse((child) => {
          if(child.isMesh){
            child.material = new THREE.MeshPhongMaterial({ color: 0xff3b3b });
          }
        });
        group.add(obj); resolve();
      }, undefined, reject);
    });
  }
  if(group.children.length){ fitViewToObject(group); }
}
function wireResultsButtons(){
  const btnShowBoth     = document.getElementById('btn-show-both');
  const btnShowBrain    = document.getElementById('btn-show-brain');
  const btnShowTumor    = document.getElementById('btn-show-tumor');
  const btnDownloadMask = document.getElementById('btn-download-mask');
  const inferStatusEl   = document.getElementById('infer-status');

  btnShowBoth.addEventListener('click', async (e) => {
    e.preventDefault();
    if(!lastUrls.brain && !lastUrls.tumor){ inferStatusEl.textContent = 'Run inference first.'; return; }
    inferStatusEl.textContent = 'Displaying brain + tumor...';
    await loadMeshes(lastUrls.brain, lastUrls.tumor);
    setActiveButton(btnShowBoth, [btnShowBoth, btnShowBrain, btnShowTumor]);
    inferStatusEl.textContent = 'Done.';
  });
  btnShowBrain.addEventListener('click', async (e) => {
    e.preventDefault();
    if(!lastUrls.brain){ inferStatusEl.textContent = 'No brain mesh yet.'; return; }
    inferStatusEl.textContent = 'Displaying brain...';
    await loadMeshes(lastUrls.brain, null);
    setActiveButton(btnShowBrain, [btnShowBoth, btnShowBrain, btnShowTumor]);
    inferStatusEl.textContent = 'Done.';
  });
  btnShowTumor.addEventListener('click', async (e) => {
    e.preventDefault();
    if(!lastUrls.tumor){ inferStatusEl.textContent = 'No tumor mesh yet.'; return; }
    inferStatusEl.textContent = 'Displaying tumor...';
    await loadMeshes(null, lastUrls.tumor);
    setActiveButton(btnShowTumor, [btnShowBoth, btnShowBrain, btnShowTumor]);
    inferStatusEl.textContent = 'Done.';
  });
  btnDownloadMask.addEventListener('click', async (e) => {
    e.preventDefault();
    if(!lastUrls.mask){ inferStatusEl.textContent = 'No mask yet.'; return; }
    try{
      const resp = await fetch(lastUrls.mask, {cache:'no-store'});
      if(!resp.ok) throw new Error(await res.text());
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'tumor_mask.nii.gz';
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
      inferStatusEl.textContent = 'Mask downloaded.';
    }catch(err){
      inferStatusEl.textContent = 'Error downloading mask: ' + err.message;
    }
  });
}
function setActiveButton(activeEl, buttons){
  buttons.forEach(b => b.classList.remove('active'));
  if (activeEl) activeEl.classList.add('active');
}

/* ================= Slice viewer (results) ================= */
function setupSliceViewer(){
  const sliceCanvas   = document.getElementById('slice-canvas');
  const sliceIndexEl  = document.getElementById('slice-index');
  const sliceSlider   = document.getElementById('slice-slider');
  const sliceStatusEl = document.getElementById('slice-status');
  const ctx = sliceCanvas.getContext('2d');

  if(!maskVolume && !bgVolume){
    sliceStatusEl.textContent = 'No mask or background available.';
    ctx.clearRect(0,0,sliceCanvas.width,sliceCanvas.height);
    sliceSlider.min = 0; sliceSlider.max = 0; sliceSlider.value = 0;
    sliceIndexEl.textContent = '—';
    return;
  }

  const ref = maskVolume || bgVolume;
  const { X, Y, Z } = ref.dims;

  let bestZ = Math.floor((Z-1)/2);
  if (maskVolume) {
    let bestSum = -1;
    const plane = maskVolume.dims.X * maskVolume.dims.Y;
    for (let z = 0; z < maskVolume.dims.Z; z++) {
      let s = 0, base = z * plane;
      for (let i = 0; i < plane; i++) s += (maskVolume.data[base + i] > 0) ? 1 : 0;
      if (s > bestSum) { bestSum = s; bestZ = z; }
    }
    if (bestSum <= 0) sliceStatusEl.textContent = 'Mask appears empty (no tumor voxels found).';
    else sliceStatusEl.textContent = '';
  } else {
    sliceStatusEl.textContent = 'No mask; showing background only.';
  }

  dims = ref.dims;
  axialZ = Math.min(Math.max(bestZ, 0), Z-1);

  sliceSlider.min = 0;
  sliceSlider.max = Math.max(0, Z - 1);
  sliceSlider.value = axialZ;
  sliceIndexEl.textContent = String(axialZ);
  drawSlice();

  sliceSlider.addEventListener('input', () => {
    axialZ = parseInt(sliceSlider.value, 10) || 0;
    sliceIndexEl.textContent = String(axialZ);
    drawSlice();
  });

  function drawSlice(){
    const renderDims = bgVolume ? bgVolume.dims : (maskVolume ? maskVolume.dims : null);
    if (!renderDims) {
      ctx.clearRect(0,0,sliceCanvas.width,sliceCanvas.height);
      return;
    }
    const { X: RX, Y: RY, Z: RZ } = renderDims;
    const z = Math.min(Math.max(axialZ, 0), RZ-1);

    sliceCanvas.width = RX;
    sliceCanvas.height = RY;

    // Background
    if (bgVolume) {
      const { X: BX, Y: BY } = bgVolume.dims;
      const bz = Math.min(Math.max(axialZ, 0), bgVolume.dims.Z-1);
      const imgDataBg = ctx.createImageData(RX, RY);

      if (BX === RX && BY === RY) {
        let min = Infinity, max = -Infinity;
        const plane = BX * BY, base = bz * plane;
        for (let i = 0; i < plane; i++) {
          const v = bgVolume.data[base + i];
          if (isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
        }
        const rng = (max - min) || 1;
        for (let y = 0; y < BY; y++) {
          const row = y*BX, baseRow = bz*BX*BY;
          for (let x = 0; x < BX; x++) {
            const idx = x + row + baseRow;
            const g = Math.max(0, Math.min(255, Math.round(((bgVolume.data[idx]-min)/rng)*255 )));
            const p = (x + y*RX) * 4;
            imgDataBg.data[p+0] = g; imgDataBg.data[p+1] = g; imgDataBg.data[p+2] = g; imgDataBg.data[p+3] = 255;
          }
        }
      } else {
        const bzBase = bz * BX * BY;
        const cx0 = Math.max(0, Math.floor(BX/2 - 64)), cx1 = Math.min(BX-1, cx0 + 127);
        const cy0 = Math.max(0, Math.floor(BY/2 - 64)), cy1 = Math.min(BY-1, cy0 + 127);
        let min = Infinity, max = -Infinity;
        for (let y = cy0; y <= cy1; y++) {
          for (let x = cx0; x <= cx1; x++) {
            const v = bgVolume.data[bzBase + x + y*BX];
            if (isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
          }
        }
        const rng = (max - min) || 1;
        for (let y = 0; y < RY; y++) {
          const sy = Math.min(BY-1, Math.round(y * (BY-1) / Math.max(1, RY-1)));
          for (let x = 0; x < RX; x++) {
            const sx = Math.min(BX-1, Math.round(x * (BX-1) / Math.max(1, RX-1)));
            const v = bgVolume.data[bzBase + sx + sy*BX];
            const g = Math.max(0, Math.min(255, Math.round(((v - min)/rng)*255 )));
            const p = (x + y*RX) * 4;
            imgDataBg.data[p+0] = g; imgDataBg.data[p+1] = g; imgDataBg.data[p+2] = g; imgDataBg.data[p+3] = 255;
          }
        }
      }
      ctx.putImageData(imgDataBg, 0, 0);
    } else {
      ctx.fillStyle = "#0a0d14";
      ctx.fillRect(0,0,sliceCanvas.width,sliceCanvas.height);
    }

    // Mask overlay
    if (maskVolume) {
      const { X: MX, Y: MY } = maskVolume.dims;
      const mz = Math.min(Math.max(axialZ, 0), maskVolume.dims.Z-1);
      const imgDataMask = ctx.getImageData(0, 0, RX, RY);

      if (MX === RX && MY === RY) {
        const plane = MX * MY, base = mz * plane;
        for (let i = 0; i < plane; i++) {
          if (maskVolume.data[base + i] > 0) {
            const p = i * 4;
            imgDataMask.data[p+0] = 255; imgDataMask.data[p+1] = 70; imgDataMask.data[p+2] = 70; imgDataMask.data[p+3] = 150;
          }
        }
      } else {
        for (let y = 0; y < RY; y++) {
          const sy = Math.min(MY-1, Math.round(y * (MY-1) / Math.max(1, RY-1)));
          for (let x = 0; x < RX; x++) {
            const sx = Math.min(MX-1, Math.round(x * (MX-1) / Math.max(1, RX-1)));
            const mv = maskVolume.data[sx + sy*MX + mz*MX*MY];
            if (mv > 0) {
              const p = (x + y*RX) * 4;
              imgDataMask.data[p+0] = 255; imgDataMask.data[p+1] = 70; imgDataMask.data[p+2] = 70; imgDataMask.data[p+3] = 150;
            }
          }
        }
      }
      ctx.putImageData(imgDataMask, 0, 0);
    }
  }
}

/* ============== Page renderers ========================================== */
function renderHome(){
  mountTemplate(tplHome);
  renderHistorySidebar();
  wireUploadUI();
  loadDataset();
}
async function renderRun(run_id){
  mountTemplate(tplRun);
  renderHistorySidebar();

  const inferStatusEl = document.getElementById('infer-status');
  try{
    let data = null;
    const cached = sessionStorage.getItem(`run:${run_id}`);
    if (cached) {
      data = JSON.parse(cached);
      lastUrls.brain = data.brain_mesh_url || null;
      lastUrls.tumor = data.tumor_mesh_url || null;
      lastUrls.mask  = data.mask_url || null;
    } else {
      inferStatusEl.textContent = 'Reloaded page: run artifacts not cached in this tab. Re-run to restore viewer.';
      return;
    }

    drawMetrics(data.metrics || {});
    await loadMeshes(lastUrls.brain, lastUrls.tumor);
    wireResultsButtons();

    const sliceStatusEl = document.getElementById('slice-status');
    sliceStatusEl.textContent = 'Loading mask for slice viewer...';
    maskVolume = await fetchServerNifti(lastUrls.mask);
    setupSliceViewer();
    sliceStatusEl.textContent = 'Ready.';
    inferStatusEl.textContent = 'Done.';
  }catch(e){
    inferStatusEl.textContent = 'Error: ' + e.message;
  }
}

/* ================= Helpers (NIfTI IO) ================= */
async function readLocalNiftiAsVolume(file){
  const ab = await file.arrayBuffer();
  return parseNiftiVolume(ab);
}
async function fetchServerNifti(url){
  const res = await fetch(url, {cache:'no-store'});
  if(!res.ok) throw new Error(await res.text());
  const ab = await res.arrayBuffer();
  if (ab.byteLength < 352) throw new Error(`Downloaded mask looks truncated (${ab.byteLength} bytes).`);
  return parseNiftiVolume(ab, true);
}
function parseNiftiVolume(arrayBuffer, forceUint8=false){
  let buf = arrayBuffer;
  if (nifti.isCompressed(buf)) buf = nifti.decompress(buf);
  if (!nifti.isNIFTI(buf)) throw new Error("Not a valid NIfTI file.");

  const header = nifti.readHeader(buf);
  if (!header || header.vox_offset > buf.byteLength) throw new Error("Invalid NIfTI header.");
  const image  = nifti.readImage(header, buf);

  const X = header.dims[1], Y = header.dims[2], Z = header.dims[3] || 1;
  const pixDims = [header.pixDims[1], header.pixDims[2], header.pixDims[3]];

  let data;
  const dt = header.datatypeCode;
  if(forceUint8){
    data = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
  }else{
    if(dt === nifti.NIFTI1.TYPE_UINT8)        data = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
    else if(dt === nifti.NIFTI1.TYPE_INT16)   data = new Int16Array(image.buffer, image.byteOffset, image.byteLength/2);
    else if(dt === nifti.NIFTI1.TYPE_INT32)   data = new Int32Array(image.buffer, image.byteOffset, image.byteLength/4);
    else if(dt === nifti.NIFTI1.TYPE_FLOAT32) data = new Float32Array(image.buffer, image.byteOffset, image.byteLength/4);
    else if(dt === nifti.NIFTI1.TYPE_FLOAT64) data = new Float64Array(image.buffer, image.byteOffset, image.byteLength/8);
    else {
      const u8 = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
      data = new Float32Array(u8.length);
      for(let i=0;i<u8.length;i++) data[i] = u8[i];
    }
  }
  return { data, dims: {X,Y,Z}, pixdim: pixDims };
}

/* ================= Boot ================= */
renderHistorySidebar();
if (!location.hash) go('/home'); else onRoute();
