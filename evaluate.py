import os
import sys
import glob
import shutil
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Thêm đường dẫn dự án vào sys.path để import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.agent import PPOAgent
from src.utils import RunningMeanStd, load_checkpoint

def evaluate(model_path="models/best_model.pth", num_episodes=3, record=True, output_dir="web_app/static"):
    print("=== STARTING PPO ROBOT MODEL EVALUATION ===")
    
    # 1. Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Initialize environment
    # Use rgb_array for recording, or human to open interactive pygame window
    render_mode = "rgb_array" if record else "human"
    
    try:
        env = gym.make(Config.ENV_NAME, render_mode=render_mode)
    except Exception as e:
        print(f"[!] Cannot initialize environment with render_mode='{render_mode}': {e}")
        print("[!] Tip: Ensure pygame and Box2D are installed correctly. Try: pip install pygame")
        return
    
    if record:
        print(f"[*] Activating RecordVideo. Videos will be exported to: {output_dir}")
        env = RecordVideo(
            env, 
            video_folder=output_dir,
            episode_trigger=lambda x: True,
            name_prefix="eval_session"
        )
        
    # 3. Khởi tạo Agent và RunningMeanStd
    agent = PPOAgent(Config.STATE_DIM, Config.ACTION_DIM)
    running_ms = RunningMeanStd(shape=(Config.STATE_DIM,)) if Config.NORMALIZE_OBS else None
    
    # 4. Tải checkpoint trọng số và trạng thái chuẩn hóa
    try:
        load_checkpoint(agent, running_ms, model_path, Config.DEVICE)
    except FileNotFoundError:
        print(f"\n[!] ERROR: Checkpoint not found at: '{model_path}'")
        print("[!] You need to run training first with train.py to save checkpoints in models/ folder!")
        env.close()
        return
        
    # Chuyển mạng sang chế độ đánh giá (Evaluation Mode) để tắt dropout hoặc tối ưu tính toán
    agent.network.eval()
    
    episode_rewards = []
    
    # 5. Khởi chạy chu kỳ đánh giá
    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        
        while not done:
            # 5.1. Chuẩn hóa trạng thái dựa trên các tham số động RunningMeanStd thu được lúc huấn luyện
            # Lưu ý: Lúc đánh giá, ta chỉ chuẩn hóa tĩnh (không update các tham số mean/var của running_ms)
            if Config.NORMALIZE_OBS:
                std = np.sqrt(running_ms.var + 1e-8)
                state_norm = (state - running_ms.mean) / std
                state_norm = np.clip(state_norm, -Config.CLIP_OBS, Config.CLIP_OBS)
            else:
                state_norm = state
                
            # 5.2. Lựa chọn hành động tối ưu (Deterministic Action)
            # Thay vì lấy mẫu ngẫu nhiên từ phân phối Gaussian, ta lấy trực tiếp giá trị Mean (trung bình)
            # Việc này giúp robot loại bỏ các nhiễu động ngẫu nhiên, giúp đi bộ ổn định và mượt mà nhất.
            state_t = torch.as_tensor(state_norm, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
            with torch.no_grad():
                action_mean = agent.network.actor_mean(state_t)
                action = action_mean.squeeze(0).cpu().numpy()
                
            clipped_action = np.clip(action, -1.0, 1.0)
            
            # 5.3. Tương tác với môi trường
            next_state, reward, terminated, truncated, info = env.step(clipped_action)
            done = terminated or truncated
            
            ep_reward += reward
            state = next_state
            steps += 1
            
        episode_rewards.append(ep_reward)
        print(f" -> Episode {ep + 1}: Reward = {ep_reward:7.2f} | Steps = {steps:4d}")
        
    env.close()
    
    mean_reward = np.mean(episode_rewards)
    print(f"\n[+] Evaluation completed. Mean reward: {mean_reward:.2f}")
    
    # 6. Find simulation video and copy to demo.mp4 for Web App
    if record:
        video_files = glob.glob(os.path.join(output_dir, "eval_session-episode-*.mp4"))
        if video_files:
            video_files.sort(key=os.path.getmtime)
            latest_video = video_files[-1]
            dest_path = os.path.join(output_dir, "demo.mp4")
            shutil.copy2(latest_video, dest_path)
            print(f"[SUCCESS] Exported simulation video successfully to web demo: {dest_path}")
        else:
            print("[!] Could not find output video from RecordVideo!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation script for PPO Agent on BipedalWalker-v3")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Path to the model .pth file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to evaluate")
    parser.add_argument("--no-record", action="store_true", help="If set, open Pygame window to watch live instead of recording video")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        record=not args.no_record
    )
