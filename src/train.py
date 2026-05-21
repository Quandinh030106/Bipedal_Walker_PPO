import os
import sys
import time
import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

# Đảm bảo import được các module từ thư mục src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.buffer import RolloutBuffer
from src.agent import PPOAgent
from src.utils import RunningMeanStd, save_checkpoint, RewardNormalizer

def train():
    # 1. Khởi tạo TensorBoard Writer
    run_name = f"runs/{Config.ENV_NAME}_PPO_{int(time.time())}"
    writer = SummaryWriter(run_name)
    print(f"[*] Starting training. TensorBoard logs are recorded at: {run_name}")
    
    # 2. Thiết lập hạt giống ngẫu nhiên (Reproducibility)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
        
    # 3. Khởi tạo môi trường huấn luyện
    # BipedalWalker không cần render khi train để đạt tốc độ tối đa
    env = gym.make(Config.ENV_NAME)
    
    # 4. Khởi tạo Agent, Buffer, bộ chuẩn hóa Obs và bộ chuẩn hóa Reward
    agent = PPOAgent(Config.STATE_DIM, Config.ACTION_DIM)
    buffer = RolloutBuffer(
        buffer_size=Config.NUM_STEPS,
        state_dim=Config.STATE_DIM,
        action_dim=Config.ACTION_DIM,
        device=Config.DEVICE
    )
    running_ms = RunningMeanStd(shape=(Config.STATE_DIM,)) if Config.NORMALIZE_OBS else None
    reward_normalizer = RewardNormalizer(gamma=Config.GAMMA) if Config.NORMALIZE_REWARDS else None
    
    # 5. Khởi tạo biến trạng thái huấn luyện
    global_step = 0
    update_idx = 0
    episode_idx = 0
    best_mean_reward = -100.0  # Điểm kỷ lục ban đầu của robot
    
    # Bộ đệm để theo dõi trung bình phần thưởng của 10 episodes gần nhất
    recent_rewards = []
    
    # Bắt đầu reset môi trường ban đầu
    state, info = env.reset(seed=Config.SEED)
    if Config.NORMALIZE_OBS:
        state_norm = running_ms.normalize(state, clip_val=Config.CLIP_OBS)
    else:
        state_norm = state
        
    episode_reward = 0.0
    episode_len = 0
    
    start_time = time.time()
    
    # 6. Vòng lặp huấn luyện chính
    try:
        while global_step < Config.TOTAL_TIMESTEPS:
            # Thu thập các rollouts tương tác từ môi trường
            for t in range(Config.NUM_STEPS):
                global_step += 1
                episode_len += 1
                
                # Chọn hành động từ Agent
                action, log_prob, _, value = agent.select_action(state_norm)
                
                # Giới hạn lực mô-men khớp động cơ trong biên độ [-1.0, 1.0] của Gymnasium
                clipped_action = np.clip(action, -1.0, 1.0)
                
                # Tương tác với môi trường vật lý
                next_state, reward, terminated, truncated, info = env.step(clipped_action)
                done = terminated or truncated
                
                # Chuẩn hóa phần thưởng nếu cấu hình bật NORMALIZE_REWARDS
                if Config.NORMALIZE_REWARDS:
                    reward_norm = reward_normalizer.normalize(reward, done)
                else:
                    reward_norm = reward
                
                # Lưu trữ vào bộ đệm dữ liệu Rollout (sử dụng reward đã chuẩn hóa)
                buffer.store(state_norm, action, log_prob, reward_norm, done, value[0])
                
                episode_reward += reward
                state = next_state
                
                # Chuẩn hóa trạng thái mới thu được
                if Config.NORMALIZE_OBS:
                    state_norm = running_ms.normalize(state, clip_val=Config.CLIP_OBS)
                else:
                    state_norm = state
                    
                # Xử lý khi kết thúc một Episode (Robot ngã hoặc về đích)
                if done:
                    episode_idx += 1
                    recent_rewards.append(episode_reward)
                    if len(recent_rewards) > 10:
                        recent_rewards.pop(0)
                        
                    mean_reward_10 = np.mean(recent_rewards)
                    
                    # Ghi log ra màn hình console
                    print(
                        f"[Step: {global_step:07d}] "
                        f"Episode {episode_idx:04d}: Reward = {episode_reward:7.2f} | "
                        f"Len = {episode_len:4d} | "
                        f"Mean(10) = {mean_reward_10:7.2f}"
                    )
                    
                    # Ghi log ra TensorBoard
                    writer.add_scalar("charts/episodic_reward", episode_reward, global_step)
                    writer.add_scalar("charts/episodic_length", episode_len, global_step)
                    writer.add_scalar("charts/mean_reward_10", mean_reward_10, global_step)
                    
                    # Đặt lại các biến theo dõi cho episode tiếp theo
                    state, info = env.reset()
                    if Config.NORMALIZE_OBS:
                        state_norm = running_ms.normalize(state, clip_val=Config.CLIP_OBS)
                    else:
                        state_norm = state
                    episode_reward = 0.0
                    episode_len = 0
                    
            # 7. Tính toán GAE và Returns mục tiêu sau khi đầy Rollout Buffer
            # Dự đoán giá trị Critic của trạng thái tiếp theo để làm gốc tính toán GAE
            _, _, _, next_val = agent.select_action(state_norm)
            buffer.compute_returns_and_advantages(
                next_value=next_val[0],
                next_done=done,
                gamma=Config.GAMMA,
                gae_lambda=Config.GAE_LAMBDA
            )
            
            # 8. Suy giảm tuyến tính tốc độ học (Learning Rate Annealing)
            lr_now = Config.LEARNING_RATE
            if Config.ANNEAL_LR:
                frac = 1.0 - (global_step - 1.0) / Config.TOTAL_TIMESTEPS
                frac = max(0.0, frac)
                lr_now = frac * Config.LEARNING_RATE
                
            # 9. Thực hiện tối ưu hóa cập nhật mạng Neural
            losses = agent.update(buffer, lr_now=lr_now if Config.ANNEAL_LR else None)
            update_idx += 1
            
            # Ghi nhận các tổn thất vào TensorBoard để theo dõi hội tụ mạng
            writer.add_scalar("losses/policy_loss", losses['policy_loss'], global_step)
            writer.add_scalar("losses/value_loss", losses['value_loss'], global_step)
            writer.add_scalar("losses/entropy_loss", losses['entropy_loss'], global_step)
            writer.add_scalar("losses/total_loss", losses['total_loss'], global_step)
            writer.add_scalar("charts/learning_rate", lr_now, global_step)
            
            # In tốc độ xử lý (SPS - Steps Per Second)
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            
            # 10. Auto-save checkpoints
            if len(recent_rewards) > 0:
                current_mean = np.mean(recent_rewards)
                # Save new best model if mean reward improved
                if current_mean > best_mean_reward and len(recent_rewards) >= 1:
                    best_mean_reward = current_mean
                    save_checkpoint(agent, running_ms, episode_idx, "models/best_model.pth", reward_normalizer=reward_normalizer)
                    
                if update_idx % 20 == 0:
                    save_checkpoint(agent, running_ms, episode_idx, "models/latest_model.pth", reward_normalizer=reward_normalizer)
                    
            # Clear buffer for the next rollout
            buffer.clear()
            
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user (Ctrl+C). Saving progress...")
    finally:
        # Save the final model at the end of training
        save_checkpoint(agent, running_ms, episode_idx, "models/latest_model.pth", reward_normalizer=reward_normalizer)
        # If best_model does not exist yet (e.g. training ended before any episodes were completed), save it too
        if not os.path.exists("models/best_model.pth"):
            save_checkpoint(agent, running_ms, episode_idx, "models/best_model.pth", reward_normalizer=reward_normalizer)
            
        # Close environment and TensorBoard
        env.close()
        writer.close()
        print("[SUCCESS] Clean exit completed! Model weights have been saved in models/ directory.")

if __name__ == "__main__":
    train()
