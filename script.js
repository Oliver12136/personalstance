import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

/* =======================
   可调参数（改这里）
   =======================
*/
const PARAMS = {
  interactionThreshold: 200,
  mergeThreshold: 120,           // 距离阈值（越大越容易触发强效果）
  mergeMaxRadius: 60,
  particleCount: 18,
  particleCooldown: 700,
  lineDash: [6,6],
  labelFontSize: 14,
  detectionScoreThreshold: 0.5
};
/* ======================= */

const S = {
  samples: 28,
  hairBase: 6,
  hairMax: 26,
  bandWidthMax: 26,
  waveAmpMax: 38,
  waveFreqBase: 5.5,
  tangentJitterMax: 6,
  hairLineWidth: 0.9,
  glowMax: 70,
  baseLineWidthMax: 16,
  minAlpha: 0.7,               // 所有线条的最低透明度
  maxAlphaBoost: 0.22,         // 线条接近时附加的透明度增量
  glowAlphaBase: 0.7,          // glow 线的基础透明度
  glowAlphaBoost: 0.18         // glow 接近时附加的透明度增量
};

const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const overlay = document.getElementById("overlay");
const enableWebcamButton = document.getElementById("webcamButton");

let objectDetector;
let runningMode = "IMAGE";
let children = [];
let particles = [];
let pairColors = {};   // 每对线固定颜色
let pairParams = {};   // 每对线固定“毛笔参数”（让笔迹稳定）
let lastFrameTime = performance.now();

const COLORS = ["#ff3b30","#0a84ff","#34c759","#ff9500","#bf5af2","#64d2ff","#ffd60a"];

/* ---------- 初始化模型 ---------- */
async function initializeObjectDetector() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );
  objectDetector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite",
      delegate: "GPU"
    },
    scoreThreshold: PARAMS.detectionScoreThreshold,
    runningMode: runningMode
  });
  console.log("Object detector ready");
}
initializeObjectDetector();

/* ---------- overlay 自适应全屏 ---------- */
function resizeOverlay() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  overlay.width = w;
  overlay.height = h;
  overlay.style.width = w + "px";
  overlay.style.height = h + "px";
  liveView.style.width = w + "px";
  liveView.style.height = h + "px";
}
video.addEventListener("loadedmetadata", resizeOverlay);
window.addEventListener("resize", resizeOverlay);

/* ---------- 坐标映射 ---------- */
function mapBBoxToPixels(bbox, displayW, displayH) {
  const ox = bbox.originX;
  const oy = bbox.originY;
  const bw = bbox.width;
  const bh = bbox.height;
  const isNorm = ox <= 1 && oy <= 1 && bw <= 1 && bh <= 1;

  if (isNorm) {
    const left = displayW * (1 - ox - bw);
    const top = displayH * oy;
    const width = displayW * bw;
    const height = displayH * bh;
    return { left, top, width, height };
  } else {
    const scaleX = displayW / video.videoWidth;
    const scaleY = displayH / video.videoHeight;
    const left = displayW - (ox + bw) * scaleX;
    const top = oy * scaleY;
    const width = bw * scaleX;
    const height = bh * scaleY;
    return { left, top, width, height };
  }
}

/* ---------- 启动摄像头 ---------- */
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() not supported");
}
async function enableCam() {
  if (!objectDetector) {
    console.log("Object detector not loaded yet.");
    return;
  }
  enableWebcamButton.style.display = "none";
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  video.addEventListener("loadeddata", () => {
    resizeOverlay();
    predictWebcam();
  });
}

/* ---------- 主循环 ---------- */
let lastVideoTime = -1;
async function predictWebcam() {
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await objectDetector.setOptions({ runningMode: "VIDEO" });
  }

  const now = performance.now();
  lastFrameTime = now;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const detections = objectDetector.detectForVideo(video, now);
    displayVideoDetections(detections);
  }
  window.requestAnimationFrame(predictWebcam);
}

/* ---------- 计算框边缘距离（重叠为负） ---------- */
function edgeDistance(boxA, boxB) {
  const ax1 = boxA.left, ay1 = boxA.top;
  const ax2 = boxA.left + boxA.width, ay2 = boxA.top + boxA.height;
  const bx1 = boxB.left, by1 = boxB.top;
  const bx2 = boxB.left + boxB.width, by2 = boxB.top + boxB.height;

  let dx = 0;
  if (ax2 < bx1) dx = bx1 - ax2;
  else if (bx2 < ax1) dx = ax1 - bx2;
  else dx = -Math.min(ax2 - bx1, bx2 - ax1);

  let dy = 0;
  if (ay2 < by1) dy = by1 - ay2;
  else if (by2 < ay1) dy = ay1 - by2;
  else dy = -Math.min(ay2 - by1, by2 - ay1);

  return Math.hypot(dx, dy);
}

/* ---------- 小工具：稳定随机/噪声 ---------- */
function hash32(str) { // 稳定 hash
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
function mulberry32(seed) { // 稳定 PRNG
  return function() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function getPairParams(pairKey, needCount) {
  if (!pairParams[pairKey]) {
    pairParams[pairKey] = { hairs: [] };
  }
  const bag = pairParams[pairKey].hairs;
  while (bag.length < needCount) {
    const idx = bag.length;
    const rng = mulberry32(hash32(pairKey + ":" + idx));
    bag.push({
      offsetBias: (rng() - 0.5),                 // 每条细线在带宽里的偏置（-0.5..0.5）
      widthScale: 0.8 + rng()*0.6,               // 线宽微差
      alphaScale: 0.7 + rng()*0.6,               // 透明度微差
      ph1: rng()*Math.PI*2, ph2: rng()*Math.PI*2, // 噪声相位
      f1: 3 + rng()*2.5, f2: 6 + rng()*3.5,      // 噪声频率
      tSpeed1: 0.6 + rng()*0.6,                   // 时间速度
      tSpeed2: 0.8 + rng()*0.8
    });
  }
  return pairParams[pairKey];
}
function smoothNoise(t, p) { // 两个正弦的混合，平滑、可动画
  return Math.sin(t * p.f1 + p.ph1) * 0.6 + Math.sin(t * p.f2 + p.ph2) * 0.4;
}
function clamp01(x){ return x<0?0:(x>1?1:x); }

/* ---------- 渲染 ---------- */
function displayVideoDetections(result) {
  for (const el of children) if (el && el.parentNode) el.parentNode.removeChild(el);
  children = [];

  resizeOverlay();
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0,0,overlay.width,overlay.height);

  const vw = overlay.width;
  const vh = overlay.height;
  if (!vw || !vh) return;

  const boxes = [];

  result.detections.forEach((det, idx) => {
    const color = COLORS[idx % COLORS.length];
    const bbox = mapBBoxToPixels(det.boundingBox, vw, vh);
    const label = det.categories[0].categoryName;

    const boxDiv = document.createElement("div");
    boxDiv.className = "highlighter";
    boxDiv.style.left = Math.round(bbox.left) + "px";
    boxDiv.style.top = Math.round(bbox.top) + "px";
    boxDiv.style.width = Math.round(bbox.width) + "px";
    boxDiv.style.height = Math.round(bbox.height) + "px";
    boxDiv.style.border = `2px dashed ${color}`;
    liveView.appendChild(boxDiv);
    children.push(boxDiv);

    const labelP = document.createElement("p");
    labelP.innerText = `${label} ${Math.round(det.categories[0].score*100)}%`;
    labelP.style.left = Math.round(bbox.left) + "px";
    labelP.style.top = Math.max(0, Math.round(bbox.top - 18)) + "px";
    labelP.style.color = color;
    labelP.style.fontSize = PARAMS.labelFontSize + "px";
    liveView.appendChild(labelP);
    children.push(labelP);

    boxes.push({ ...bbox, label, color });
  });

  /* ------------------- 两两连线：细线成束 + 波浪手写 + 距离始终可见 ------------------- */
  ctx.save();
  ctx.lineCap = 'round';

  const time = performance.now() * 0.001;


  const SOFT_RANGE = (PARAMS.softRange ?? PARAMS.mergeThreshold * 5);
  const MIN_VIS    = (PARAMS.minVis ?? 0.12);
  const GAMMA      = (PARAMS.proximityGamma ?? 1.6);

  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const A = boxes[i], B = boxes[j];
      const dist = edgeDistance(A, B);
      const near = clamp01(1 - dist / SOFT_RANGE);
      const p = MIN_VIS + Math.pow(near, GAMMA) * (1 - MIN_VIS);

      const ax = A.left + A.width/2, ay = A.top + A.height/2;
      const bx = B.left + B.width/2, by = B.top + B.height/2;
      const dx = bx - ax, dy = by - ay;
      const len = Math.hypot(dx, dy);
      if (len === 0) continue;

      const ux = dx / len,  uy = dy / len;
      const nx = -uy,       ny = ux;

      const pairKey = `${i}-${j}`;
      if (!pairColors[pairKey]) {
        const hue = Math.floor((Math.random() + j * 0.6180339887) * 360) % 360; // 避免集中
        pairColors[pairKey] = `hsl(${hue}, 92%, 56%)`;
      }
      const baseColor = pairColors[pairKey];

      const hairCount      = Math.round(S.hairBase + p * (S.hairMax - S.hairBase));
      const band           = p * S.bandWidthMax;
      const waveAmp        = (0.25 + 0.75*p) * S.waveAmpMax;
      const tangentJitter  = p * S.tangentJitterMax;

      // Glow 背景线
      ctx.save();
      ctx.beginPath();
      for (let s = 0; s <= S.samples; s++) {
        const t = s / S.samples;
        let cx = ax + dx * t;
        let cy = ay + dy * t;
        const w = Math.sin((t * (S.waveFreqBase + 1.5)) * Math.PI * 2 + time) * waveAmp * 0.25;
        cx += nx * w;
        cy += ny * w;
        if (s === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
      }
      ctx.strokeStyle = baseColor;
      ctx.globalAlpha = S.glowAlphaBase + p * S.glowAlphaBoost;
      ctx.lineWidth   = 2 + p * (S.baseLineWidthMax ?? 16);
      ctx.shadowBlur  = p * S.glowMax;
      ctx.shadowColor = baseColor;
      ctx.stroke();
      ctx.restore();

      // 细线束绘制
      const params = getPairParams(pairKey, hairCount);
      for (let h = 0; h < hairCount; h++) {
        const hp = params.hairs[h];
        const offset = ( (h/(hairCount-1 || 1)) - 0.5 );
        const offsetWithBias = (offset + hp.offsetBias*0.15) * band;
        const alpha = (S.minAlpha + p * S.maxAlphaBoost) * hp.alphaScale;

        ctx.beginPath();
        for (let s = 0; s <= S.samples; s++) {
          const t = s / S.samples;
          let x = ax + dx * t;
          let y = ay + dy * t;
          let px = x + nx * offsetWithBias;
          let py = y + ny * offsetWithBias;

          const wave =
            smoothNoise(t * (S.waveFreqBase + hp.f1*0.2) + time*hp.tSpeed1, hp) * (waveAmp * 0.7) +
            smoothNoise(t * (S.waveFreqBase + hp.f2*0.2) + time*hp.tSpeed2, hp) * (waveAmp * 0.3);

          const tj = smoothNoise(t*8 + time*1.1 + hp.ph2, hp) * tangentJitter;
          const nj = smoothNoise(t*6 + time*0.9 + hp.ph1, hp) * (band * 0.12);

          px += nx * (wave + nj);
          py += ny * (wave + nj);
          px += ux * tj;
          py += uy * tj;

          if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }

        ctx.strokeStyle = baseColor;
        ctx.globalAlpha = alpha;
        ctx.lineWidth   = S.hairLineWidth * hp.widthScale * (1 + p*0.6);
        ctx.shadowBlur  = p * (S.glowMax * 0.2);
        ctx.shadowColor = baseColor;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }
  }

  ctx.restore();
}

// ---- Live controls for S parameters ----
document.getElementById("hairWidth")?.addEventListener("input", (e) => {
  S.hairLineWidth = parseFloat(e.target.value);
});
document.getElementById("glowMax")?.addEventListener("input", (e) => {
  S.glowMax = parseFloat(e.target.value);
});
document.getElementById("waveAmpMax")?.addEventListener("input", (e) => {
  S.waveAmpMax = parseFloat(e.target.value);
});
document.getElementById("baseLineWidthMax")?.addEventListener("input", (e) => {
  S.baseLineWidthMax = parseFloat(e.target.value);
});
document.getElementById("minAlpha")?.addEventListener("input", (e) => {
  S.minAlpha = parseFloat(e.target.value);
});
document.getElementById("maxAlphaBoost")?.addEventListener("input", (e) => {
  S.maxAlphaBoost = parseFloat(e.target.value);
});
document.getElementById("glowAlphaBase")?.addEventListener("input", (e) => {
  S.glowAlphaBase = parseFloat(e.target.value);
});
document.getElementById("glowAlphaBoost")?.addEventListener("input", (e) => {
  S.glowAlphaBoost = parseFloat(e.target.value);
});

/* ---------- 动态修改置信度阈值 ---------- */
async function setDetectionScoreThreshold(v){
  if(!objectDetector) return;
  PARAMS.detectionScoreThreshold = v;
  await objectDetector.setOptions({ scoreThreshold: v });
  console.log("Updated detection scoreThreshold to", v);
}