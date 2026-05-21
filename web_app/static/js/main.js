// Global variables for training and evaluation polling
let trainPollInterval = null;
let evalPollInterval = null;
let isInteractiveSimRunning = false;
let simInterval = null;
let scrollOffset = 0;
let cumulativeSimReward = 0.0;

// Chart.js Instance
let performanceChart = null;

// Canvas details
const canvas = document.getElementById("telemetryCanvas");
const ctx = canvas.getContext("2d");

// Joint dimensions for visualizer
const L1 = 45; // Thigh length
const L2 = 40; // Shin length
const HULL_W = 60;
const HULL_H = 26;

// Color palette
const COLORS = {
    bg: "#07080d",
    hullOutline: "#8b5cf6",
    hullFill: "rgba(139, 92, 246, 0.2)",
    leg1: "#00e5ff",
    leg2: "#ec4899",
    lidarGreen: "rgba(16, 185, 129, 0.25)",
    lidarRed: "rgba(239, 68, 68, 0.7)",
    ground: "#1f2937",
    grid: "rgba(255, 255, 255, 0.03)"
};

// --- INITIALIZE GRAPH ---
function initChart() {
    const chartCtx = document.getElementById("performanceChart").getContext("2d");
    performanceChart = new Chart(chartCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Episode Reward',
                    data: [],
                    borderColor: '#00e5ff',
                    backgroundColor: 'rgba(0, 229, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: 'y'
                },
                {
                    label: 'Mean Reward (10)',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.05)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    tension: 0.3,
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#9ca3af',
                        font: { family: 'Outfit' }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { family: 'Outfit' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { family: 'Outfit' } },
                    position: 'left'
                }
            }
        }
    });
}

// --- TRAINING SUBPROCESS CONTROL ---

function startTraining() {
    fetch("/api/train/start", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            logToTerminal("[SYSTEM] " + data.message, "success");
            document.getElementById("btn-train-start").disabled = true;
            document.getElementById("btn-train-stop").disabled = false;
            
            // Bắt đầu vòng lặp thăm dò trạng thái huấn luyện
            if (trainPollInterval) clearInterval(trainPollInterval);
            trainPollInterval = setInterval(pollTrainStatus, 1000);
        })
        .catch(err => {
            logToTerminal("[ERROR] Failed to start training: " + err, "error");
        });
}

function stopTraining() {
    fetch("/api/train/stop", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            logToTerminal("[SYSTEM] " + data.message, "error");
            document.getElementById("btn-train-start").disabled = false;
            document.getElementById("btn-train-stop").disabled = true;
            
            if (trainPollInterval) {
                clearInterval(trainPollInterval);
                trainPollInterval = null;
            }
            setTimeout(pollTrainStatus, 1000); // Poll một lần cuối cùng
        })
        .catch(err => {
            logToTerminal("[ERROR] Failed to stop training: " + err, "error");
        });
}

function pollTrainStatus() {
    fetch("/api/train/status")
        .then(res => res.json())
        .then(data => {
            const status = data.status;
            const history = data.history;
            const logs = data.logs;
            
            // Cập nhật giao diện Trạng thái
            const dot = document.getElementById("train-status-dot");
            const text = document.getElementById("train-status-text");
            
            if (status.is_running) {
                dot.className = "status-dot active";
                text.innerText = "Training Active";
                document.getElementById("btn-train-start").disabled = true;
                document.getElementById("btn-train-stop").disabled = false;
            } else {
                dot.className = "status-dot";
                text.innerText = "Idle";
                document.getElementById("btn-train-start").disabled = false;
                document.getElementById("btn-train-stop").disabled = true;
                if (trainPollInterval) {
                    clearInterval(trainPollInterval);
                    trainPollInterval = null;
                }
            }
            
            // Thống kê số liệu
            document.getElementById("stat-steps").innerText = status.current_step.toLocaleString();
            document.getElementById("stat-episodes").innerText = status.current_episode;
            document.getElementById("stat-reward").innerText = status.latest_reward.toFixed(2);
            document.getElementById("stat-mean").innerText = status.mean_reward_10.toFixed(2);
            
            // Dòng Log từ Subprocess
            logs.forEach(line => {
                // Kiểm tra xem dòng log đã có trong terminal chưa để tránh in trùng
                if (!isLineInTerminal(line)) {
                    logToTerminal(line);
                }
            });
            
            // Vẽ biểu đồ
            if (history && history.length > 0) {
                const labels = history.map(h => "Ep " + h.episode);
                const rewards = history.map(h => h.reward);
                const means = history.map(h => h.mean_reward_10);
                
                performanceChart.data.labels = labels;
                performanceChart.data.datasets[0].data = rewards;
                performanceChart.data.datasets[1].data = means;
                performanceChart.update();
            }
        })
        .catch(err => {
            console.error("Error polling training status:", err);
        });
}

// --- EVALUATION & SIMULATION RENDER ---

function runEvaluation() {
    document.getElementById("btn-eval").disabled = true;
    const progressBlock = document.getElementById("eval-progress-bar");
    const progressText = document.getElementById("eval-progress-text");
    
    progressBlock.style.display = "block";
    progressText.innerText = "Initializing record environment...";
    logToTerminal("[EVAL] Launching high-quality physics video rendering...", "success");
    
    fetch("/api/evaluate", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            if (evalPollInterval) clearInterval(evalPollInterval);
            evalPollInterval = setInterval(pollEvalStatus, 1000);
        })
        .catch(err => {
            logToTerminal("[EVAL ERROR] " + err, "error");
            document.getElementById("btn-eval").disabled = false;
        });
}

function pollEvalStatus() {
    fetch("/api/evaluate/status")
        .then(res => res.json())
        .then(data => {
            const progressText = document.getElementById("eval-progress-text");
            progressText.innerText = data.progress;
            
            if (!data.is_running) {
                clearInterval(evalPollInterval);
                evalPollInterval = null;
                document.getElementById("btn-eval").disabled = false;
                
                if (data.error) {
                    logToTerminal("[EVAL ERROR] " + data.error, "error");
                } else {
                    logToTerminal("[EVAL] Physics rendering finished successfully! Video saved.", "success");
                    // Reload video player to bypass browser cache
                    const videoContainer = document.querySelector(".video-container");
                    videoContainer.innerHTML = `
                        <video id="demoVideo" controls autoplay loop>
                            <source src="/static/demo.mp4?t=${Date.now()}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    `;
                }
            }
        })
        .catch(err => {
            console.error("Error polling eval status:", err);
        });
}

// --- INTERACTIVE HEADLESS SIMULATION & CANVAS DRAWING ---

function toggleInteractiveSim() {
    if (isInteractiveSimRunning) {
        stopInteractiveSim();
    } else {
        startInteractiveSim();
    }
}

function startInteractiveSim() {
    logToTerminal("[SIM] Loading neural policy and initializing headless simulator...", "success");
    
    fetch("/api/sim/reset", { method: "POST" })
        .then(res => {
            if (!res.ok) throw new Error("Check model checkpoint exists. Train PPO first!");
            return res.json();
        })
        .then(data => {
            isInteractiveSimRunning = true;
            cumulativeSimReward = 0.0;
            
            // Thiết lập badge
            document.getElementById("sim-status-dot").className = "status-dot simulating";
            document.getElementById("sim-status-text").innerText = "Simulating Walk";
            document.getElementById("btn-sim-toggle").innerHTML = `
                <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M15.75 5.25v13.5m-7.5-13.5v13.5"></path></svg>
                Stop Live Scan
            `;
            document.getElementById("btn-sim-reset").disabled = false;
            
            // Nhận quan sát đầu tiên và vẽ
            drawSimulationFrame(data.observation);
            
            // Bắt đầu vòng lặp physics step (25 FPS ~ 40ms)
            if (simInterval) clearInterval(simInterval);
            simInterval = setInterval(stepInteractiveSim, 40);
        })
        .catch(err => {
            logToTerminal("[SIM ERROR] " + err.message, "error");
        });
}

function stopInteractiveSim() {
    isInteractiveSimRunning = false;
    if (simInterval) {
        clearInterval(simInterval);
        simInterval = null;
    }
    
    document.getElementById("sim-status-dot").className = "status-dot";
    document.getElementById("sim-status-text").innerText = "Inactive";
    document.getElementById("btn-sim-toggle").innerHTML = `
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z"></path></svg>
        Start Live Scan
    `;
    document.getElementById("btn-sim-reset").disabled = true;
    
    fetch("/api/sim/close", { method: "POST" })
        .then(() => logToTerminal("[SIM] Interactive simulator closed.", "error"));
}

function resetInteractiveSim() {
    fetch("/api/sim/reset", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            cumulativeSimReward = 0.0;
            logToTerminal("[SIM] Environment reset triggered.");
            drawSimulationFrame(data.observation);
        });
}

function stepInteractiveSim() {
    if (!isInteractiveSimRunning) return;
    
    fetch("/api/sim/step", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            cumulativeSimReward = data.total_reward;
            
            // Cập nhật số liệu telemetry lên màn hình overlay
            document.getElementById("telemetry-reward").innerText = data.total_reward.toFixed(2);
            document.getElementById("readout-steps").innerText = data.steps;
            
            // Vẽ frame
            drawSimulationFrame(data.observation);
            
            // Tự động loop nếu done (robot ngã hoặc hết thời gian)
            if (data.done) {
                logToTerminal("[SIM] Robot fell or completed epoch. Auto-resetting...", "error");
                resetInteractiveSim();
            }
        })
        .catch(err => {
            console.error("Step sim error:", err);
            stopInteractiveSim();
        });
}

// --- TELEMETRY PHYSICS CANVAS RENDER ENGINE ---

function drawSimulationFrame(obs) {
    if (!obs || obs.length < 24) return;
    
    // Parse quan sát
    const hullAngle = obs[0];
    const hullVx = obs[2];
    const hullVy = obs[3];
    const hip1 = obs[4];
    const knee1 = obs[6];
    const contact1 = obs[8];
    const hip2 = obs[9];
    const knee2 = obs[11];
    const contact2 = obs[13];
    
    // Cập nhật thông số text
    document.getElementById("telemetry-vx").innerText = hullVx.toFixed(2);
    document.getElementById("telemetry-vy").innerText = hullVy.toFixed(2);
    document.getElementById("telemetry-angle").innerText = (hullAngle * 180 / Math.PI).toFixed(1);
    
    document.getElementById("readout-hip1").innerText = hip1.toFixed(2) + " rad";
    document.getElementById("readout-knee1").innerText = knee1.toFixed(2) + " rad";
    document.getElementById("readout-hip2").innerText = hip2.toFixed(2) + " rad";
    document.getElementById("readout-knee2").innerText = knee2.toFixed(2) + " rad";
    
    document.getElementById("readout-contact1").innerText = contact1 > 0.5 ? "ON" : "OFF";
    document.getElementById("readout-contact1").className = contact1 > 0.5 ? "sensor-value green" : "sensor-value text-muted";
    document.getElementById("readout-contact2").innerText = contact2 > 0.5 ? "ON" : "OFF";
    document.getElementById("readout-contact2").className = contact2 > 0.5 ? "sensor-value pink" : "sensor-value text-muted";
    
    document.getElementById("readout-tilt").innerText = hullAngle.toFixed(2) + " rad";
    
    // Xóa màn hình vẽ
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Tính toán camera scrolling bằng cách tích lũy dịch chuyển theo vận tốc ngang
    // Giúp mặt đất và robot chuyển động mượt mà tiến lên phía trước
    scrollOffset += hullVx * 4;
    
    // 1. Vẽ lưới tọa độ chuyển động (Scrolling Grid)
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    const gridSpacing = 40;
    const startX = -(scrollOffset % gridSpacing);
    for (let x = startX; x < canvas.width; x += gridSpacing) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    
    // 2. Vẽ mặt đất tĩnh (Ground Baseline)
    const groundY = canvas.height - 80;
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(canvas.width, groundY);
    ctx.stroke();
    
    // Điểm mốc trọng tâm robot trong Canvas (Đặt tại trung tâm)
    const cx = canvas.width / 2;
    const cy = groundY - 95 + hullVy * 20; // Dao động nhẹ trục Y theo vận tốc đứng
    
    // 3. Vẽ 10 Lidar Beams (Laser Scanner)
    ctx.lineWidth = 1;
    for (let i = 0; i < 10; i++) {
        const lidarVal = obs[14 + i];
        // Tính toán góc quét tuyệt đối từ -30 đến +30 độ xung quanh trục đứng của robot
        const lidarAngleLocal = -Math.PI / 6 + i * (Math.PI / 3) / 9;
        const absAngle = hullAngle + lidarAngleLocal + Math.PI / 2;
        
        // Quét khoảng cách tối đa 160 pixel
        const maxRange = 160;
        const beamLen = lidarVal * maxRange;
        
        const beamEndX = cx + beamLen * Math.sin(absAngle);
        const beamEndY = cy + beamLen * Math.cos(absAngle);
        
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(beamEndX, beamEndY);
        
        // Đổi màu neon đỏ nếu khoảng cách phát hiện vật cản/mặt đất ngắn
        if (lidarVal < 0.95) {
            ctx.strokeStyle = "rgba(239, 68, 68, 0.2)";
            ctx.stroke();
            
            // Vẽ điểm đỏ laser phản xạ
            ctx.fillStyle = "#ef4444";
            ctx.beginPath();
            ctx.arc(beamEndX, beamEndY, 2.5, 0, Math.PI * 2);
            ctx.fill();
        } else {
            ctx.strokeStyle = COLORS.lidarGreen;
            ctx.stroke();
        }
    }
    
    // 4. Vẽ Chân Robot (Legs) - Leg 2 vẽ trước (ở phía sau) rồi tới Leg 1 (ở phía trước)
    drawLeg(cx, cy, hullAngle, hip2, knee2, contact2, COLORS.leg2);
    drawLeg(cx, cy, hullAngle, hip1, knee1, contact1, COLORS.leg1);
    
    // 5. Vẽ Thân Robot (Hull Chassis)
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(hullAngle);
    
    // Vẽ khối hộp thân (Rounded Chassis)
    ctx.fillStyle = COLORS.hullFill;
    ctx.strokeStyle = COLORS.hullOutline;
    ctx.lineWidth = 3.5;
    ctx.shadowBlur = 12;
    ctx.shadowColor = COLORS.hullOutline;
    
    ctx.beginPath();
    ctx.roundRect(-HULL_W / 2, -HULL_H / 2, HULL_W, HULL_H, 6);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0; // Tắt shadow cho các nét tiếp theo
    
    // Vẽ buồng điều khiển trung tâm (Futuristic windshield)
    ctx.fillStyle = "rgba(0, 229, 255, 0.4)";
    ctx.beginPath();
    ctx.moveTo(10, -HULL_H / 2 + 3);
    ctx.lineTo(HULL_W / 2 - 4, -HULL_H / 2 + 3);
    ctx.lineTo(HULL_W / 2 - 12, 2);
    ctx.lineTo(10, 2);
    ctx.closePath();
    ctx.fill();
    
    ctx.restore();
}

function drawLeg(cx, cy, hullAngle, hipAngle, kneeAngle, contact, color) {
    // Vị trí khớp hông tuyệt đối (Có lệch nhẹ so với tâm Hull)
    const hipXLocal = -8; // Đặt khớp hông lệch sau một chút
    const hipYLocal = 8;
    
    const cosHA = Math.cos(hullAngle);
    const sinHA = Math.sin(hullAngle);
    const hipX = cx + hipXLocal * cosHA - hipYLocal * sinHA;
    const hipY = cy + hipXLocal * sinHA + hipYLocal * cosHA;
    
    // Góc Thigh (Đùi) tuyệt đối
    const thighAngle = hullAngle + hipAngle + Math.PI / 2;
    const kneeX = hipX + L1 * Math.sin(thighAngle);
    const kneeY = hipY + L1 * Math.cos(thighAngle);
    
    // Góc Shin (Cẳng chân) tuyệt đối
    const shinAngle = thighAngle + kneeAngle;
    const footX = kneeX + L2 * Math.sin(shinAngle);
    const footY = kneeY + L2 * Math.cos(shinAngle);
    
    // Vẽ đùi (Thigh)
    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(hipX, hipY);
    ctx.lineTo(kneeX, kneeY);
    ctx.stroke();
    
    // Vẽ cẳng chân (Shin)
    ctx.beginPath();
    ctx.moveTo(kneeX, kneeY);
    ctx.lineTo(footX, footY);
    ctx.stroke();
    
    // Vẽ bàn chân (Foot/Sole)
    ctx.lineWidth = 3;
    const footW = 14;
    ctx.beginPath();
    ctx.moveTo(footX - footW / 2, footY);
    ctx.lineTo(footX + footW / 2, footY);
    ctx.stroke();
    
    // Khớp gối & khớp hông (Neon Joints)
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(kneeX, kneeY, 4, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(hipX, hipY, 5, 0, Math.PI * 2);
    ctx.fill();
    
    // Vẽ vòng hào quang tiếp đất nếu chân chạm đất (Contact Glow)
    if (contact > 0.5) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.shadowBlur = 10;
        ctx.shadowColor = color;
        ctx.beginPath();
        ctx.arc(footX, footY, 8, 0, Math.PI, true);
        ctx.stroke();
        ctx.shadowBlur = 0;
    }
}

// --- CORE UTILITIES ---

function logToTerminal(text, type = "normal") {
    const term = document.getElementById("terminalWindow");
    if (!term) return;
    
    const line = document.createElement("div");
    line.className = "terminal-line " + type;
    
    // Thêm timestamp ngắn gọn
    const timeStr = new Date().toLocaleTimeString();
    line.innerText = `[${timeStr}] ${text}`;
    
    term.appendChild(line);
    term.scrollTop = term.scrollHeight; // Tự động cuộn xuống dưới
}

function isLineInTerminal(lineText) {
    const term = document.getElementById("terminalWindow");
    const lines = term.getElementsByClassName("terminal-line");
    // Kiểm tra trong 15 dòng cuối
    const startIdx = Math.max(0, lines.length - 15);
    for (let i = startIdx; i < lines.length; i++) {
        if (lines[i].innerText.includes(lineText)) return true;
    }
    return false;
}

// --- INIT PAGE ---
window.addEventListener("DOMContentLoaded", () => {
    initChart();
    
    // Vẽ khung canvas trống ban đầu đẹp mắt
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(255, 255, 255, 0.15)";
    ctx.font = "14px Outfit";
    ctx.textAlign = "center";
    ctx.fillText("Interactive Radar Visualizer Off • Click Start Live Scan", canvas.width / 2, canvas.height / 2);
    
    // Thăm dò trạng thái lúc tải trang
    pollTrainStatus();
    
    // Nếu huấn luyện đang hoạt động, thiết lập polling liên tục
    setTimeout(() => {
        const text = document.getElementById("train-status-text").innerText;
        if (text === "Training Active") {
            trainPollInterval = setInterval(pollTrainStatus, 1000);
        }
    }, 1200);
});
