import os
import sys
import json
import time
import subprocess
import threading
import signal
import numpy as np
import torch
import gymnasium as gym
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Thêm thư mục dự án vào sys.path để import cấu hình và mạng
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.agent import PPOAgent
from src.utils import RunningMeanStd, load_checkpoint

app = FastAPI(title="Bipedal Walker PPO Premium Dashboard")

# Mount thư mục static cho CSS, JS và video demo
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# Cài đặt Jinja2 template
templates = Jinja2Templates(directory="web_app/templates")

# Global states để kiểm soát tiến trình huấn luyện và đánh giá
train_process = None
train_thread = None
training_logs = []
training_history = []  # Danh sách dict chứa: step, episode, reward, length, mean_reward_10
training_status = {
    "is_running": False,
    "current_step": 0,
    "current_episode": 0,
    "latest_reward": 0.0,
    "mean_reward_10": 0.0,
    "sps": 0,
    "error": None
}

eval_process = None
eval_status = {
    "is_running": False,
    "progress": "",
    "output_video": None,
    "error": None
}

# Đơn vị lưu trữ mô phỏng trực tiếp (Interactive Live Scanner)
class LiveSimulator:
    def __init__(self):
        self.env = None
        self.agent = None
        self.running_ms = None
        self.state = None
        self.total_reward = 0.0
        self.steps = 0
        self.is_active = False

    def initialize(self):
        model_path = "models/best_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Best model checkpoint not found. Please train the agent first!")
        
        # Tạo môi trường headless
        self.env = gym.make(Config.ENV_NAME, render_mode="rgb_array")
        
        # Khởi tạo Agent
        self.agent = PPOAgent(Config.STATE_DIM, Config.ACTION_DIM)
        self.running_ms = RunningMeanStd(shape=(Config.STATE_DIM,)) if Config.NORMALIZE_OBS else None
        
        # Tải mô hình
        load_checkpoint(self.agent, self.running_ms, model_path, Config.DEVICE)
        self.agent.network.eval()
        
        self.is_active = True

    def reset(self):
        if not self.is_active:
            self.initialize()
        
        self.state, info = self.env.reset()
        self.total_reward = 0.0
        self.steps = 0
        return self.state.tolist()

    def step(self):
        if not self.is_active or self.state is None:
            raise HTTPException(status_code=400, detail="Simulator not initialized. Run reset first.")
        
        # 1. Chuẩn hóa trạng thái (Evaluation Mode: static normalization)
        if Config.NORMALIZE_OBS and self.running_ms:
            std = np.sqrt(self.running_ms.var + 1e-8)
            state_norm = (self.state - self.running_ms.mean) / std
            state_norm = np.clip(state_norm, -Config.CLIP_OBS, Config.CLIP_OBS)
        else:
            state_norm = self.state

        # 2. Lấy hành động tối ưu (Deterministic Actor mean)
        state_t = torch.as_tensor(state_norm, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            action_mean = self.agent.network.actor_mean(state_t)
            action = action_mean.squeeze(0).cpu().numpy()
            
        clipped_action = np.clip(action, -1.0, 1.0)
        
        # 3. Tương tác môi trường
        next_state, reward, terminated, truncated, info = self.env.step(clipped_action)
        done = terminated or truncated
        
        self.total_reward += reward
        self.steps += 1
        self.state = next_state
        
        return {
            "observation": next_state.tolist(),
            "reward": float(reward),
            "done": bool(done),
            "total_reward": float(self.total_reward),
            "steps": int(self.steps)
        }

    def close(self):
        if self.env:
            self.env.close()
        self.env = None
        self.is_active = False

simulator = LiveSimulator()


# --- BACKGROUND LOG READING THREAD ---
def read_subprocess_output(process):
    global training_logs, training_status, training_history
    
    # Đọc từng dòng log xuất ra từ train.py
    for line in iter(process.stdout.readline, ""):
        clean_line = line.strip()
        if not clean_line:
            continue
            
        # Lưu trữ tối đa 100 dòng log gần nhất để hiển thị
        training_logs.append(clean_line)
        if len(training_logs) > 100:
            training_logs.pop(0)
            
        # Parse thông tin tiến trình huấn luyện bằng regex hoặc split thủ công
        # Ví dụ dòng log: [Step: 0000100] Episode 0001: Reward = -119.30 | Len =  100 | Mean(10) = -119.30
        if "[Step:" in clean_line and "Episode" in clean_line:
            try:
                parts = clean_line.split("|")
                # Parse step và episode từ block đầu tiên
                first_part = parts[0]  # "[Step: 0000100] Episode 0001: Reward = -119.30 "
                step_str = first_part.split("Step:")[1].split("]")[0].strip()
                ep_str = first_part.split("Episode")[1].split(":")[0].strip()
                rew_str = first_part.split("Reward =")[1].strip()
                
                # Parse length từ block 2
                len_str = parts[1].split("=")[1].strip()
                
                # Parse mean từ block 3
                mean_str = parts[2].split("=")[1].strip()
                
                step = int(step_str)
                episode = int(ep_str)
                reward = float(rew_str)
                length = int(len_str)
                mean_reward_10 = float(mean_str)
                
                # Cập nhật status
                training_status["current_step"] = step
                training_status["current_episode"] = episode
                training_status["latest_reward"] = reward
                training_status["mean_reward_10"] = mean_reward_10
                
                # Lưu vào lịch sử vẽ đồ thị
                training_history.append({
                    "step": step,
                    "episode": episode,
                    "reward": reward,
                    "length": length,
                    "mean_reward_10": mean_reward_10
                })
            except Exception as e:
                # Bỏ qua nếu có lỗi parse (log không khớp định dạng mong muốn)
                pass

    process.wait()
    training_status["is_running"] = False


# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # Kiểm tra xem mô hình best_model.pth đã tồn tại chưa
    model_exists = os.path.exists("models/best_model.pth")
    # Kiểm tra xem video demo đã tồn tại chưa
    video_exists = os.path.exists("web_app/static/demo.mp4")
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "model_exists": model_exists,
            "video_exists": video_exists,
            "config": {
                "env_name": Config.ENV_NAME,
                "learning_rate": Config.LEARNING_RATE,
                "gamma": Config.GAMMA,
                "gae_lambda": Config.GAE_LAMBDA,
                "total_timesteps": Config.TOTAL_TIMESTEPS,
                "num_steps": Config.NUM_STEPS,
                "device": str(Config.DEVICE)
            }
        }
    )


@app.post("/api/train/start")
def start_training():
    global train_process, train_thread, training_logs, training_status, training_history
    
    if training_status["is_running"]:
        return JSONResponse(status_code=400, content={"message": "Training is already running."})
        
    training_logs = []
    training_history = []
    training_status = {
        "is_running": True,
        "current_step": 0,
        "current_episode": 0,
        "latest_reward": 0.0,
        "mean_reward_10": 0.0,
        "sps": 0,
        "error": None
    }
    
    try:
        # Chạy train.py như một subprocess để độc lập GIL và bộ nhớ
        # Sử dụng python kèm đường dẫn tương đối
        train_process = subprocess.Popen(
            [sys.executable, "src/train.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Bắt đầu luồng đọc output không chặn
        train_thread = threading.Thread(target=read_subprocess_output, args=(train_process,), daemon=True)
        train_thread.start()
        
        return {"message": "Training started successfully."}
    except Exception as e:
        training_status["is_running"] = False
        training_status["error"] = str(e)
        return JSONResponse(status_code=500, content={"message": f"Failed to start training: {e}"})


@app.post("/api/train/stop")
def stop_training():
    global train_process, training_status
    
    if not training_status["is_running"] or train_process is None:
        return JSONResponse(status_code=400, content={"message": "Training is not running."})
        
    try:
        # Gửi tín hiệu ngắt (Terminate) tới tiến trình huấn luyện
        train_process.terminate()
        train_process.wait(timeout=5)
        training_status["is_running"] = False
        return {"message": "Training stopped successfully."}
    except Exception as e:
        # Nếu terminate thất bại, cưỡng chế kill
        try:
            train_process.kill()
            training_status["is_running"] = False
            return {"message": "Training forcefully terminated."}
        except Exception as kill_err:
            return JSONResponse(status_code=500, content={"message": f"Failed to stop training: {kill_err}"})


@app.get("/api/train/status")
def get_train_status():
    global training_status, training_logs, training_history
    return {
        "status": training_status,
        "history": training_history,
        "logs": training_logs[-30:]  # Gửi 30 dòng log mới nhất
    }


@app.post("/api/evaluate")
def run_evaluation(background_tasks: BackgroundTasks):
    global eval_status
    
    if eval_status["is_running"]:
        return JSONResponse(status_code=400, content={"message": "Evaluation session is already in progress."})
        
    eval_status["is_running"] = True
    eval_status["progress"] = "Starting evaluation..."
    eval_status["error"] = None
    
    # Chạy quy trình đánh giá trong background task của FastAPI để tránh chặn request chính
    def task_eval():
        global eval_process, eval_status
        try:
            # Ghi hình 1 episode chất lượng cao nhất để xuất video sang web
            eval_process = subprocess.Popen(
                [sys.executable, "evaluate.py", "--episodes", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(eval_process.stdout.readline, ""):
                clean_line = line.strip()
                if clean_line:
                    eval_status["progress"] = clean_line
                    
            eval_process.wait()
            eval_status["is_running"] = False
            eval_status["output_video"] = "/static/demo.mp4"
            eval_status["progress"] = "Evaluation completed successfully! Simulation video is ready."
        except Exception as e:
            eval_status["is_running"] = False
            eval_status["error"] = str(e)
            eval_status["progress"] = f"Evaluation failed: {e}"
            
    background_tasks.add_task(task_eval)
    return {"message": "Evaluation started in background."}


@app.get("/api/evaluate/status")
def get_evaluate_status():
    global eval_status
    return eval_status


# --- INTERACTIVE SIMULATION ENDPOINTS ---

@app.post("/api/sim/reset")
def sim_reset():
    try:
        obs = simulator.reset()
        return {"observation": obs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/api/sim/step")
def sim_step():
    try:
        step_result = simulator.step()
        return step_result
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/api/sim/close")
def sim_close():
    simulator.close()
    return {"message": "Simulator closed successfully."}


@app.get("/api/config")
def get_config():
    return {
        "ENV_NAME": Config.ENV_NAME,
        "STATE_DIM": Config.STATE_DIM,
        "ACTION_DIM": Config.ACTION_DIM,
        "DEVICE": str(Config.DEVICE),
        "NORMALIZE_OBS": Config.NORMALIZE_OBS,
        "CLIP_OBS": Config.CLIP_OBS,
        "LEARNING_RATE": Config.LEARNING_RATE,
        "GAMMA": Config.GAMMA,
        "GAE_LAMBDA": Config.GAE_LAMBDA,
        "TOTAL_TIMESTEPS": Config.TOTAL_TIMESTEPS,
        "NUM_STEPS": Config.NUM_STEPS,
        "MINIBATCH_SIZE": Config.MINIBATCH_SIZE,
        "UPDATE_EPOCHS": Config.UPDATE_EPOCHS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
