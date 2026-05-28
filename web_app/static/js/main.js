/* ============================================================
   BIPEDAL WALKER PPO — Main Application Logic
   WebSocket Client + Humanoid Canvas Renderer + Chart.js
   ============================================================ */

// --- Constants ---
const WS_URL = `ws://${window.location.host}/ws/sim`;
const STEP_INTERVAL_MS = { 1: 60, 2: 30, 5: 12 };

// --- State ---
let ws = null;
let isRunning = false;
let isPaused = false;
let currentSpeed = 1;
let episodeRewards = [];
let currentEpisodeReward = 0;
let currentSteps = 0;
let bestReward = -Infinity;
let params = { noise: 0.0, wind: 0.0, gravity: 1.0 };
let rewardChart = null;
let lastObs = null;
let posX = 0;
let terrainSeed = Math.random() * 1000;
let needsChartReset = true;


// --- DOM refs ---
const canvas = document.getElementById('robot-canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('canvas-overlay');
const fallenOverlay = document.getElementById('fallen-overlay');
const statusBadge = document.getElementById('connection-badge');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

const statReward = document.getElementById('stat-reward');
const statSteps = document.getElementById('stat-steps');
const statBest = document.getElementById('stat-best');
const statEpisodes = document.getElementById('stat-episodes');

const btnRun = document.getElementById('btn-run');
const btnPause = document.getElementById('btn-pause');
const btnReset = document.getElementById('btn-reset');

const sliderNoise = document.getElementById('slider-noise');
const sliderWind = document.getElementById('slider-wind');
const sliderGravity = document.getElementById('slider-gravity');
const valNoise = document.getElementById('val-noise');
const valWind = document.getElementById('val-wind');
const valGravity = document.getElementById('val-gravity');

// ============================================================
// CHART SETUP
// ============================================================
function initChart() {
  const chartCtx = document.getElementById('reward-chart').getContext('2d');
  rewardChart = new Chart(chartCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Episode Reward',
        data: [],
        borderColor: '#0ea5e9',
        backgroundColor: 'rgba(14, 165, 233, 0.1)',
        borderWidth: 3,
        tension: 0.4,
        fill: true,
        pointRadius: 4,
        pointBackgroundColor: '#0ea5e9',
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#475569', font: { family: 'Nunito', size: 10, weight: 'bold' } },
          grid: { color: 'rgba(56, 189, 248, 0.15)' }
        },
        y: {
          ticks: { color: '#475569', font: { family: 'Nunito', size: 10, weight: 'bold' } },
          grid: { color: 'rgba(56, 189, 248, 0.15)' }
        }
      }
    }
  });
}

function pushEpisodeToChart(reward) {
  const ep = episodeRewards.length;
  rewardChart.data.labels.push(`Ep ${ep}`);
  rewardChart.data.datasets[0].data.push(reward.toFixed(1));
  if (rewardChart.data.labels.length > 30) {
    rewardChart.data.labels.shift();
    rewardChart.data.datasets[0].data.shift();
  }
  rewardChart.update('none');
}

function resetChart() {
  if (rewardChart) {
    rewardChart.data.labels = [];
    rewardChart.data.datasets[0].data = [];
    rewardChart.update();
  }
  episodeRewards = [];
  statEpisodes.textContent = '0';
  bestReward = -Infinity;
  statBest.textContent = '-100.0';
}

// ============================================================
// CONNECTION STATUS
// ============================================================
function setStatus(state) {
  // state: 'disconnected' | 'connecting' | 'connected'
  statusBadge.className = `connection-badge ${state}`;
  const labels = {
    disconnected: '⬤  Disconnected',
    connecting:   '⬤  Connecting...',
    connected:    '⬤  Connected'
  };
  statusText.textContent = labels[state] || state;
}

// ============================================================
// WEBSOCKET
// ============================================================
function connectAndRun() {
  if (ws) { ws.close(); ws = null; }

  // Reset biểu đồ điểm số khi chạy hệ số mới hoặc reset
  if (needsChartReset) {
    resetChart();
    needsChartReset = false;
  }

  setStatus('connecting');
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setStatus('connected');
    posX = 0;
    terrainSeed = Math.random() * 1000;
    // Send reset command with current params
    sendMessage({ action: 'reset', ...params });
    isRunning = true;
    isPaused = false;
    updateButtons();
    hideOverlay();
  };

  ws.onmessage = (event) => {
    if (isPaused) return;
    const data = JSON.parse(event.data);
    handleSimState(data);
  };

  ws.onerror = () => setStatus('disconnected');

  ws.onclose = () => {
    setStatus('disconnected');
    isRunning = false;
    updateButtons();
  };
}

function sendMessage(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ============================================================
// SIMULATION STATE HANDLER
// ============================================================
function handleSimState(data) {
  if (!data.observation) return;

  lastObs = data.observation;
  currentEpisodeReward = data.total_reward || 0;
  currentSteps = data.steps || 0;

  // Tích lũy khoảng cách vật lý posX (velX * dt) với dt = 0.05s mỗi step
  const velX = lastObs[2] || 0;
  posX += velX * 0.05;

  // Update stats
  updateStatValue(statReward, currentEpisodeReward.toFixed(1));
  statSteps.textContent = currentSteps;

  // Render humanoid robot
  resizeCanvas();
  drawScene(lastObs);

  if (data.done) {
    episodeRewards.push(currentEpisodeReward);
    pushEpisodeToChart(currentEpisodeReward);
    statEpisodes.textContent = episodeRewards.length;

    if (currentEpisodeReward > bestReward) {
      bestReward = currentEpisodeReward;
      statBest.textContent = bestReward.toFixed(1);
      statBest.parentElement.classList.add('pulse');
      setTimeout(() => statBest.parentElement.classList.remove('pulse'), 800);
    }

    if (currentEpisodeReward < -50) {
      showFallen();
    }

    // Khi Episode kết thúc, đóng WebSocket và yêu cầu người dùng nhấn Launch để chạy lại
    isRunning = false;
    if (ws) {
      ws.close();
      ws = null;
    }
    updateButtons();
    
    // Hiện overlay thông báo hoàn thành episode sau một khoảng trễ ngắn
    setTimeout(() => {
      showOverlay('🔁', 'Episode Finished', 'Simulation paused. Click "Launch Live Simulation" to start a new episode with current settings');
    }, 1000);
  }
}

function updateStatValue(el, value) {
  el.textContent = value;
  const num = parseFloat(value);
  el.className = 'stat-value ' + (num > 0 ? 'positive' : num < -50 ? 'negative' : '');
}

// ============================================================
// HUMANOID ROBOT RENDERER
// ============================================================

// BipedalWalker-v3 observation indices:
//  0: hull_angle           1: hull_angular_vel
//  2: vel_x                3: vel_y
//  4: hip_1_angle          5: hip_1_angular_vel
//  6: knee_1_angle         7: knee_1_angular_vel
//  8: leg_1_ground_contact
//  9: hip_2_angle         10: hip_2_angular_vel
// 11: knee_2_angle        12: knee_2_angular_vel
// 13: leg_2_ground_contact
// 14-23: lidar readings

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  if (canvas.width !== rect.width || canvas.height !== rect.height) {
    canvas.width = rect.width;
    canvas.height = rect.height;
  }
}

function getGroundHeight(x, seed) {
  if (x < 1.5) return 0; // 1.5m đầu tiên phẳng để xuất phát vững vàng
  
  // Chuyển tiếp mượt từ phẳng sang gồ ghề giữa 1.5m và 3.0m
  const transition = Math.min(1, Math.max(0, (x - 1.5) / 1.5));
  
  // Tổng các sóng sine/cosine có tần số khác nhau, lệch pha bằng seed
  const wave1 = Math.sin(x * 0.35 + seed) * 0.45;
  const wave2 = Math.cos(x * 0.85 - seed * 0.7) * 0.20;
  const wave3 = Math.sin(x * 1.85 + seed * 1.3) * 0.08;
  const wave4 = Math.cos(x * 3.5 + seed * 2.1) * 0.03;
  
  return (wave1 + wave2 + wave3 + wave4) * transition;
}

function drawScene(obs) {
  const W = canvas.width;
  const H = canvas.height;

  // --- Background ---
  ctx.clearRect(0, 0, W, H);

  // Sky gradient
  const sky = ctx.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, '#bae6fd'); // Playful bright sky blue
  sky.addColorStop(1, '#e0f2fe');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H);

  // Ground baseline position
  const groundY = H * 0.72;

  // Tọa độ Y hông robot nhấp nhô theo mặt đồi tại vị trí posX hiện tại
  const robotY = groundY - getGroundHeight(posX, terrainSeed) * 60 - 75;

  // Draw lidar beams (obs 14–23)
  drawLidar(obs, W / 2, robotY);

  // Draw moving ground grid / wavy terrain
  drawGround(obs, groundY, W, H);

  // Draw humanoid robot
  const hullAngle = obs[0];
  const contact1  = obs[8] > 0.5;
  const contact2  = obs[13] > 0.5;

  const hip1Angle  = obs[4];
  const knee1Angle = obs[6];
  const hip2Angle  = obs[9];
  const knee2Angle = obs[11];

  drawHumanoid(
    W / 2, robotY,
    hullAngle,
    hip1Angle, knee1Angle, contact1,
    hip2Angle, knee2Angle, contact2
  );
}

function drawGround(obs, groundY, W, H) {
  // Vẽ đa giác kín mặt đất đồi núi gập ghềnh
  ctx.beginPath();
  ctx.moveTo(0, H);
  
  const stepPx = 6;
  for (let px = 0; px <= W + stepPx; px += stepPx) {
    const globalX = posX + (px - W / 2) / 60;
    const y = groundY - getGroundHeight(globalX, terrainSeed) * 60;
    ctx.lineTo(px, y);
  }
  
  ctx.lineTo(W, H);
  ctx.closePath();

  // Gradient màu nền đất (Playful green grass)
  const groundGrad = ctx.createLinearGradient(0, groundY - 40, 0, H);
  groundGrad.addColorStop(0, '#86efac'); // Happy bright green grass
  groundGrad.addColorStop(0.4, '#4ade80');
  groundGrad.addColorStop(1, '#166534'); // Deep friendly garden green
  ctx.fillStyle = groundGrad;
  ctx.fill();

  // Vẽ đường viền neon phát sáng
  ctx.beginPath();
  for (let px = 0; px <= W; px += stepPx) {
    const globalX = posX + (px - W / 2) / 60;
    const y = groundY - getGroundHeight(globalX, terrainSeed) * 60;
    if (px === 0) ctx.moveTo(px, y);
    else ctx.lineTo(px, y);
  }

  const glowGrad = ctx.createLinearGradient(0, 0, W, 0);
  glowGrad.addColorStop(0, 'rgba(34, 197, 94, 0.2)');
  glowGrad.addColorStop(0.3, 'rgba(34, 197, 94, 0.95)');
  glowGrad.addColorStop(0.7, 'rgba(16, 185, 129, 0.95)');
  glowGrad.addColorStop(1, 'rgba(16, 185, 129, 0.2)');

  ctx.strokeStyle = glowGrad;
  ctx.lineWidth = 4;
  ctx.shadowColor = 'rgba(34, 197, 94, 0.6)';
  ctx.shadowBlur = 10;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Vẽ các vạch mốc mét m
  const minX = posX - (W / 2) / 60;
  const maxX = posX + (W / 2) / 60;
  const startM = Math.floor(minX);
  const endM = Math.ceil(maxX);

  ctx.font = '10px monospace';
  ctx.textAlign = 'center';

  for (let m = startM; m <= endM; m++) {
    if (m < 0) continue;
    const px = W / 2 + (m - posX) * 60;
    const groundYAtM = groundY - getGroundHeight(m, terrainSeed) * 60;

    // Vạch đứng nhỏ phát sáng
    ctx.strokeStyle = 'rgba(21, 128, 61, 0.35)'; // dark green markers
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, groundYAtM);
    ctx.lineTo(px, groundYAtM + 12);
    ctx.stroke();

    // Nhãn chữ m
    ctx.fillStyle = '#166534'; // clear readable green text
    ctx.fillText(`${m}m`, px, groundYAtM + 25);
  }
}

function drawLidar(obs, cx, cy) {
  const lidar = obs.slice(14, 24); // 10 readings
  const hullAngle = obs[0] || 0;
  const lidarRangePx = 5.33 * 60; // 320px

  lidar.forEach((dist, i) => {
    // Quét tia lidar xung quanh hướng thẳng đứng, nghiêng theo góc hullAngle
    const angle = hullAngle + Math.PI / 2 + (i - 4.5) * 0.15;
    const len = dist * lidarRangePx;

    const endX = cx + Math.cos(angle) * len;
    const endY = cy + Math.sin(angle) * len;

    const isNear = dist < 0.4;
    const color = isNear ? 'rgba(244, 63, 94, 0.85)' : 'rgba(14, 165, 233, 0.55)';

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(endX, endY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Vẽ điểm va chạm neon sáng đẹp trên mặt đất
    if (dist < 1.0) {
      ctx.shadowColor = isNear ? 'rgba(244, 63, 94, 0.9)' : 'rgba(14, 165, 233, 0.9)';
      ctx.shadowBlur = 8;
      ctx.fillStyle = isNear ? 'rgba(255, 100, 100, 1)' : 'rgba(100, 255, 255, 1)';
      ctx.beginPath();
      ctx.arc(endX, endY, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  });
}

function drawHumanoid(cx, cy, hullAngle, hip1, knee1, contact1, hip2, knee2, contact2) {
  // Scale for drawing
  const THIGH_LEN = 52;
  const SHIN_LEN  = 48;
  const BODY_W    = 22;
  const BODY_H    = 38;
  const HEAD_R    = 14;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(hullAngle);

  // --- BODY (torso) ---
  drawRoundedRect(ctx, -BODY_W / 2, -BODY_H / 2, BODY_W, BODY_H, 8,
    createBodyGrad(ctx, BODY_H));

  // Chest detail line
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(-BODY_W / 2 + 4, -4);
  ctx.lineTo(BODY_W / 2 - 4, -4);
  ctx.stroke();

  // --- HEAD ---
  ctx.save();
  ctx.translate(0, -BODY_H / 2 - HEAD_R - 4);
  ctx.rotate(-hullAngle * 0.3); // slight counter-rotation for natural feel
  
  // Head glow
  ctx.shadowColor = 'rgba(14, 165, 233, 0.4)';
  ctx.shadowBlur = 12;
  
  const headGrad = ctx.createRadialGradient(-3, -3, 2, 0, 0, HEAD_R);
  headGrad.addColorStop(0, '#38bdf8'); // sky blue head
  headGrad.addColorStop(1, '#0284c7');
  ctx.fillStyle = headGrad;
  ctx.beginPath();
  ctx.arc(0, 0, HEAD_R, 0, Math.PI * 2);
  ctx.fill();

  // Visor / eyes
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#ffffff'; // friendly bright white eyes
  ctx.beginPath();
  ctx.roundRect(-7, -4, 6, 5, 2);
  ctx.fill();
  ctx.beginPath();
  ctx.roundRect(1, -4, 6, 5, 2);
  ctx.fill();
  ctx.restore();

  // --- ARMS (swing opposite to legs) ---
  const armSwing = (hip1 + hip2) * 0.3;

  // Left arm
  ctx.save();
  ctx.translate(-BODY_W / 2, -BODY_H / 4);
  ctx.rotate(-armSwing - 0.3);
  ctx.strokeStyle = '#64748b'; // friendly gray arms
  ctx.lineWidth = 7;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(-8, 28);
  ctx.stroke();
  // Forearm
  ctx.save();
  ctx.translate(-8, 28);
  ctx.rotate(0.5);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(-4, 22);
  ctx.stroke();
  ctx.restore();
  ctx.restore();

  // Right arm
  ctx.save();
  ctx.translate(BODY_W / 2, -BODY_H / 4);
  ctx.rotate(armSwing + 0.3);
  ctx.strokeStyle = '#64748b';
  ctx.lineWidth = 7;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(8, 28);
  ctx.stroke();
  ctx.save();
  ctx.translate(8, 28);
  ctx.rotate(-0.5);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(4, 22);
  ctx.stroke();
  ctx.restore();
  ctx.restore();

  // --- LEGS ---
  // Hip pivot at bottom-center of torso
  const hipY = BODY_H / 2;

  // Leg 1 (left) — drawn behind
  drawLeg(ctx, -BODY_W / 4, hipY, hip1, knee1, contact1, THIGH_LEN, SHIN_LEN, false);
  // Leg 2 (right) — drawn in front
  drawLeg(ctx, BODY_W / 4, hipY, hip2, knee2, contact2, THIGH_LEN, SHIN_LEN, true);

  ctx.restore();
}

function drawLeg(ctx, x, y, hipAngle, kneeAngle, contact, thighLen, shinLen, isFront) {
  const alpha = isFront ? 1.0 : 0.65;
  
  ctx.save();
  ctx.translate(x, y);

  // Thigh
  ctx.save();
  ctx.rotate(hipAngle);

  // Thigh segment
  const thighColor = isFront ? `rgba(129, 140, 248, ${alpha})` : `rgba(79, 70, 229, ${alpha})`;
  ctx.strokeStyle = thighColor;
  ctx.lineWidth = isFront ? 12 : 10;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(0, thighLen);
  ctx.stroke();

  // Knee joint circle
  ctx.fillStyle = isFront ? 'rgba(129, 140, 248, 0.9)' : 'rgba(99, 102, 241, 0.7)';
  ctx.beginPath();
  ctx.arc(0, thighLen, 5, 0, Math.PI * 2);
  ctx.fill();

  // Shin
  ctx.translate(0, thighLen);
  ctx.rotate(kneeAngle);

  ctx.strokeStyle = isFront
    ? `rgba(99, 102, 241, ${alpha})`
    : `rgba(67, 56, 202, ${alpha})`;
  ctx.lineWidth = isFront ? 10 : 8;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(0, shinLen);
  ctx.stroke();

  // Foot
  ctx.translate(0, shinLen);

  if (contact) {
    // Ground contact splash / lawn green glow
    ctx.shadowColor = 'rgba(34, 197, 94, 0.9)';
    ctx.shadowBlur = 16;
    ctx.fillStyle = '#22c55e';
  } else {
    ctx.shadowBlur = 0;
    ctx.fillStyle = isFront ? 'rgba(129, 140, 248, 0.8)' : 'rgba(99, 102, 241, 0.6)';
  }

  ctx.beginPath();
  ctx.roundRect(-10, -4, 20, 8, 4);
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.restore();
  ctx.restore();
}

function createBodyGrad(ctx, bodyH) {
  const g = ctx.createLinearGradient(0, -bodyH / 2, 0, bodyH / 2);
  g.addColorStop(0, '#f97316'); // playful orange chest
  g.addColorStop(0.5, '#f97316');
  g.addColorStop(1, '#ea580c');
  return g;
}

function drawRoundedRect(ctx, x, y, w, h, r, fillStyle) {
  ctx.fillStyle = fillStyle;
  ctx.shadowColor = 'rgba(249, 115, 22, 0.25)';
  ctx.shadowBlur = 20;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Border
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.stroke();
}

// ============================================================
// IDLE / FALLEN CANVAS STATES
// ============================================================
function drawIdleScreen() {
  resizeCanvas();
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, '#bae6fd');
  bg.addColorStop(1, '#e0f2fe');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Ground
  ctx.strokeStyle = '#4ade80';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(0, H * 0.72);
  ctx.lineTo(W, H * 0.72);
  ctx.stroke();

  // Draw static humanoid silhouette in idle pose
  drawHumanoid(W / 2, H * 0.72 - 80, 0, 0.1, 0.2, false, -0.1, 0.2, false);
}

function showFallen() {
  fallenOverlay.classList.add('show');
}

function hideFallen() {
  fallenOverlay.classList.remove('show');
}

function hideOverlay() {
  overlay.classList.add('hidden');
}

function showOverlay(icon, text, sub) {
  overlay.classList.remove('hidden');
  overlay.querySelector('.overlay-icon').textContent = icon;
  overlay.querySelector('.overlay-text').textContent = text;
  overlay.querySelector('.overlay-sub').textContent = sub;
}

// ============================================================
// SLIDERS
// ============================================================
function initSliders() {
  const onSliderChange = () => {
    hideFallen();
    if (isRunning || ws) {
      if (ws) {
        ws.close();
        ws = null;
      }
      isRunning = false;
      isPaused = false;
      updateButtons();
      drawIdleScreen();
    }
    needsChartReset = true;
    showOverlay('⚙️', 'Parameters Changed', 'Please click "Launch Live Simulation" to apply the new settings.');
  };

  sliderNoise.addEventListener('input', () => {
    params.noise = parseFloat(sliderNoise.value);
    valNoise.textContent = params.noise.toFixed(2);
    onSliderChange();
  });

  sliderWind.addEventListener('input', () => {
    params.wind = parseFloat(sliderWind.value);
    valWind.textContent = (params.wind >= 0 ? '+' : '') + params.wind.toFixed(2);
    onSliderChange();
  });

  sliderGravity.addEventListener('input', () => {
    params.gravity = parseFloat(sliderGravity.value);
    valGravity.textContent = params.gravity.toFixed(1) + 'g';
    onSliderChange();
  });
}

// ============================================================
// BUTTONS
// ============================================================
function updateButtons() {
  btnRun.disabled = isRunning;
  btnPause.disabled = !isRunning;
  btnReset.disabled = false;
}

function initButtons() {
  btnRun.addEventListener('click', () => {
    connectAndRun();
  });

  btnPause.addEventListener('click', () => {
    isPaused = !isPaused;
    btnPause.textContent = isPaused ? '▶  Resume' : '⏸  Pause';
  });

  btnReset.addEventListener('click', () => {
    hideFallen();
    if (ws) {
      ws.close();
      ws = null;
    }
    isRunning = false;
    isPaused = false;
    currentEpisodeReward = 0;
    currentSteps = 0;
    posX = 0;
    terrainSeed = Math.random() * 1000; // Đổi đồi ngẫu nhiên mới
    statReward.textContent = '0.0';
    statSteps.textContent = '0';
    
    needsChartReset = true;
    
    updateButtons();
    drawIdleScreen(); // Đưa robot về idle silhouette
    showOverlay('🤖', 'Simulation Reset', 'Terrain randomized. Click "Launch Live Simulation" to start');
  });

  // Speed buttons
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentSpeed = parseInt(btn.dataset.speed);
      sendMessage({ action: 'set_speed', speed: currentSpeed });
    });
  });
}

function loadDiagnostics() {
  fetch('/api/config')
    .then(res => res.json())
    .then(config => {
      const modelTargetEl = document.querySelector('.info-val.highlight');
      if (modelTargetEl && config.EPISODE !== undefined) {
        modelTargetEl.textContent = `best_model.pth (Ep ${config.EPISODE})`;
      }
    })
    .catch(err => console.error('Failed to load diagnostics:', err));
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load', () => {
  initChart();
  initSliders();
  initButtons();
  drawIdleScreen();
  setStatus('disconnected');
  updateButtons();
  btnPause.disabled = true;
  loadDiagnostics();
});

window.addEventListener('resize', () => {
  if (lastObs) drawScene(lastObs);
  else drawIdleScreen();
});
