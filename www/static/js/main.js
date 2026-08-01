// ===========================================================================
// main.js - 3D Pocket-Keeper AI: Touch/Mouse physics + On-Device AI + render loop
// Captures first 15ms flick impulse then feeds Arm-optimized TFLite then 3D keeper dive
// ===========================================================================
import { Stadium } from "./scene.js";
import { Ball, getGoalZone, zoneToTargetXY, ZONE_LABELS, PENALTY_DISTANCE, GOAL_WIDTH, GOAL_HEIGHT } from "./physics.js";
import { TFLiteGoalkeeper } from "./keeper_ai.js";

const app = document.getElementById("app");
const loader = document.getElementById("loader");
const loadMsg = document.getElementById("loadMsg");
const loadErr = document.getElementById("loadErr");
const dbgInf = document.getElementById("dbgInf");
const dbgFps = document.getElementById("dbgFps");
const dbgPred = document.getElementById("dbgPred");
const dbgAct = document.getElementById("dbgAct");
const statusBox = document.getElementById("statusBox");
const probsBox = document.getElementById("probs");

const stadium = new Stadium(app);
const ball = new Ball();
let ai = null;
const keeper = { x: 0, y: 1.05, diving: false, dir: 0 };
const stats = { goals: 0, saves: 0, misses: 0 };

let state = "AIM";
let pred = { zone: -1, latency: 0 };
let actualZone = -1;
let resetTimer = 0;

const touch = { active: false, startX: 0, startY: 0, startTime: 0, points: [] };
const SHOT_POWER = 22;
const TOUCH_WINDOW_MS = 15;

function onDown(e) {
  if (state !== "AIM") return;
  const p = e.touches ? e.touches[0] : e;
  touch.active = true;
  touch.startX = p.clientX; touch.startY = p.clientY;
  touch.startTime = performance.now();
  touch.points = [{ x: p.clientX, y: p.clientY, t: touch.startTime }];
  touch.predicted = false;
}
function onMove(e) {
  if (!touch.active) return;
  const p = e.touches ? e.touches[0] : e;
  touch.points.push({ x: p.clientX, y: p.clientY, t: performance.now() });
}
function onUp(e) {
  if (!touch.active) return;
  touch.active = false;
  if (state !== "AIM") return;
  const elapsed = performance.now() - touch.startTime;
  if (elapsed < 8) return;
  const pts = touch.points;
  if (pts.length < 2) return;
  const last = pts[pts.length - 1];
  const p1 = pts[0];
  
  // 3D ekran izdüşümünü kullanarak topun tam olarak mouse'u bıraktığınız yere gitmesini sağlarız.
  const coords = stadium.getGoalTargetCoords(last.x, last.y);
  const targetX = Math.max(-GOAL_WIDTH/2 * 1.12, Math.min(GOAL_WIDTH/2 * 1.12, coords.x));
  const targetY = Math.max(0.1, Math.min(GOAL_HEIGHT * 1.12, coords.y));
  
  const dx = last.x - p1.x;
  const dy = p1.y - last.y;
  
  const flickSpeed = Math.min(Math.hypot(dx, dy) / Math.max(elapsed, 1) * 1000, 40);
  const vz0 = 24.0 + (flickSpeed * 0.1);
  const t = PENALTY_DISTANCE / vz0;
  
  const pm = pts[Math.floor(pts.length / 2)];
  const curveVal = (last.x - p1.x) * (pm.y - p1.y) - (last.y - p1.y) * (pm.x - p1.x);
  const spin = Math.max(-45.0, Math.min(45.0, curveVal * 0.12));
  
  const isFeint = (Math.abs(dy) > 30 && targetY > 1.8 && flickSpeed < 16);
  const finalSpin = isFeint ? -spin * 1.3 : spin;
  
  // Magnus sapma düzeltmesi
  const airDensity = 1.2, ballMass = 0.43, ballRadius = 0.11, magnusCoeff = 0.15;
  const ballArea = Math.PI * (ballRadius ** 2);
  const fMagX = magnusCoeff * airDensity * ballArea * ballRadius * finalSpin * vz0;
  const axMag = fMagX / ballMass;
  const dxMag = 0.5 * axMag * (t ** 2);
  
  const vx0 = (targetX - dxMag) / t;
  const vy0 = (targetY + 0.5 * 9.81 * (t ** 2)) / t;
  
  ball.launch(vx0, vy0, vz0, finalSpin);
  ball.saved = false;
  
  try {
    const r = ai.predict(vx0, vy0, finalSpin, flickSpeed);
    pred = { zone: r.zone, latency: r.latencyMs };
    dbgInf.textContent = r.latencyMs.toFixed(2) + " ms";
    dbgInf.className = "v" + (r.latencyMs < 1 ? " ok" : (r.latencyMs < 3 ? " warn" : " bad"));
    dbgPred.textContent = ZONE_LABELS[r.zone] + " (z" + r.zone + ")";
    renderProbs(r.probs);
    stadium.highlightZone(r.zone);
    startKeeper(r.zone, isFeint ? 2 : 0, targetX, targetY);
  } catch (err) {
    console.error("AI predict failed:", err);
    statusBox.textContent = "AI HATA: " + err.message;
    statusBox.className = "miss";
  }
  state = "FLY";
}

function startKeeper(zone, feintOffset, targetX, targetY) {
  const actualZone = getGoalZone(targetX, targetY);
  if (zone === actualZone) {
    // Kaleci doğru köşeyi tahmin ettiyse, görsel çarpışmanın tam olması için direkt şutun gittiği hedefe atlar.
    keeper.x = targetX;
    keeper.y = Math.max(targetY, 0.5);
  } else {
    // Yanlış tahmin ettiyse, yanlış tahmin ettiği bölgenin merkezine uçar (ters köşe olur).
    const tg = zoneToTargetXY(zone);
    keeper.x = tg.x + (feintOffset ? (tg.x < 0 ? feintOffset : -feintOffset) : 0);
    keeper.y = Math.max(tg.y, 0.5);
  }
  keeper.diving = true;
  keeper.dir = keeper.x < 0 ? -1 : (keeper.x > 0 ? 1 : 0);
}

const canvas = stadium.renderer.domElement;
canvas.addEventListener("mousedown", onDown); addEventListener("mousemove", onMove); addEventListener("mouseup", onUp);
canvas.addEventListener("touchstart", onDown, { passive: true }); addEventListener("touchmove", onMove, { passive: true }); addEventListener("touchend", onUp, { passive: true });

let fps = 60, frameCount = 0, fpsTimer = performance.now();
function loop() {
  requestAnimationFrame(loop);
  ball.update(1 / 200);
  stadium.updateBall(ball);
  if (ball.z >= PENALTY_DISTANCE - 0.05 && state === "FLY") onResolve();
  if (state === "RESOLVE") { resetTimer++; if (resetTimer > 90) reset(); }
  if (ball.active || state === "AIM") stadium.updateCamera(ball);
  stadium.updateKeeper(keeper.x, keeper.y, keeper.diving, keeper.dir);
  frameCount++;
  const n = performance.now();
  if (n - fpsTimer >= 1000) {
    fps = Math.round(frameCount * 1000 / (n - fpsTimer));
    frameCount = 0; fpsTimer = n;
    dbgFps.textContent = fps + " FPS";
    dbgFps.className = "v" + (fps >= 55 ? " ok" : (fps >= 30 ? " warn" : " bad"));
  }
  stadium.render();
}

function onResolve() {
  const az = getGoalZone(ball.x, ball.y);
  actualZone = az;
  dbgAct.textContent = az === -1 ? "KACTI" : ZONE_LABELS[az];
  let saved = false;
  if (az !== -1) {
    // Uzuv bazlı çarpışma kontrolü (Kapatma alanı: 1.4m genişlik, 1.2m yükseklik)
    const dx = Math.abs(ball.x - keeper.x);
    const dy = Math.abs(ball.y - keeper.y);
    const isTouching = (dx < 1.4 && dy < 1.2);
    
    // Rastgelelik/Hata payı: Kaleci bazen doğru bölgede olsa bile topu elinden kaçırabilir (Professional Denge)
    const skillFactor = Math.random() < 0.85; 
    
    if (isTouching && skillFactor) saved = true; 
  }
  state = "RESOLVE"; resetTimer = 0;
  ball.saved = saved; // Fizik motorunun topu filede tutması veya sektirmesi için flag'i setliyoruz
  
  if (az === -1) { 
    statusBox.textContent = "KACTI (MISS)"; statusBox.className = "miss"; 
    stats.misses++;
  }
  else if (saved) { 
    statusBox.textContent = "KURTARMA (SAVED)"; statusBox.className = "save"; 
    stats.saves++;
    // Görsel temas için topu kalecinin ellerine çekip sahaya doğru geri sektiririz.
    ball.x = keeper.x + (ball.x - keeper.x) * 0.35;
    ball.y = keeper.y + (ball.y - keeper.y) * 0.35;
    ball.vx = -ball.vx * 0.4 + (Math.random() - 0.5) * 4;
    ball.vy = Math.max(2.0, -ball.vy * 0.3 + 3);
    ball.vz = -Math.abs(ball.vz) * 0.45;
    ball.active = true;
  }
  else if (Math.abs(ball.spin) > 0.0001 && Math.abs(ball.x - keeper.x) > 1.5 && pred.zone !== az) { 
    statusBox.textContent = "TERS KOSE (FEINT)"; statusBox.className = "feint"; 
    stats.goals++;
    ball.active = true; // Topun fileye gitmeye devam etmesi için aktif tutuyoruz
  }
  else if (Math.abs(ball.vx) < 2 && Math.abs(ball.x) < 1 && ball.y < 1.2) { 
    statusBox.textContent = "PANENKA"; statusBox.className = "panenka"; 
    stats.goals++;
    ball.active = true; 
  }
  else { 
    statusBox.textContent = "GOL!"; statusBox.className = "goal"; 
    stats.goals++;
    ball.active = true; 
  }
  document.getElementById("dbgScore").textContent = stats.goals + " / " + stats.saves + " / " + stats.misses;
}

function reset() {
  state = "AIM"; ball.reset();
  keeper.x = 0; keeper.y = 1.05; keeper.diving = false; keeper.dir = 0;
  pred = { zone: -1, latency: 0 }; actualZone = -1;
  dbgPred.textContent = "-"; dbgAct.textContent = "-"; dbgInf.textContent = "-";
  statusBox.textContent = "Bekleniyor"; statusBox.className = "";
  probsBox.innerHTML = "";
  stadium.highlightZone(-1);
}

function renderProbs(probs) {
  let html = "";
  for (let i = 0; i < 9; i++) {
    const pct = (probs[i] * 100).toFixed(1);
    html += '<div class="bar"><span class="lbl">' + ZONE_LABELS[i] + '</span><div class="track"><div class="fill" style="width:' + pct + '%"></div></div><span class="pc">' + pct + '%</span></div>';
  }
  probsBox.innerHTML = html;
}

(async () => {
  try {
    loadMsg.textContent = "On-Device TFLite model yukleniyor (NEON SIMD / WASM)...";
    ai = await new TFLiteGoalkeeper("/models/goalkeeper_ai_9zone.tflite").load();
    loadMsg.textContent = "Model hazir - INT8 cikarim aktif";
    loader.style.display = "none";
    loop();
  } catch (err) {
    loadErr.textContent = "Model yuklenemedi: " + err.message + " (On-device icin /models/*.tflite gerekli)";
  }
})();
