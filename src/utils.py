import torch
import numpy as np
import os

class RunningMeanStd:
    """
    Theo dõi trung bình và phương sai động để chuẩn hóa (normalize) states theo thời gian thực.
    Được triển khai dựa trên giải thuật tính toán song song Welford để tối ưu số học.
    """
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        x = np.array(x, dtype=np.float32)
        # Nếu x là scalar (0-D array), đưa về 1-D array (batch size = 1) trước khi tính mean/var
        if x.ndim == 0:
            x = x.reshape(1)
        # Nếu x là 1-D vector và self.mean là vector nhiều chiều (chuẩn hóa quan sát), đưa về batch (1, N)
        elif x.ndim == 1 and self.mean.ndim >= 1:
            x = np.expand_dims(x, axis=0)
            
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # Cập nhật mean động
        self.mean = self.mean + delta * batch_count / tot_count
        
        # Cập nhật phương sai động (Welford's algorithm)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, obs, clip_val=10.0):
        """Chuẩn hóa quan sát và cắt các giá trị ngoại lai (outliers)"""
        obs = np.array(obs, dtype=np.float32)
        self.update(obs)
        std = np.sqrt(self.var + 1e-8)
        normalized_obs = (obs - self.mean) / std
        return np.clip(normalized_obs, -clip_val, clip_val)


class RewardNormalizer:
    """
    Chuẩn hóa phần thưởng (rewards) trực tuyến bằng phương sai động của returns tích lũy.
    Giúp giảm bước nhảy giá trị lớn (như hình phạt ngã -100) của BipedalWalker.
    """
    def __init__(self, gamma=0.99):
        self.return_rms = RunningMeanStd(shape=())
        self.gamma = gamma
        self.returns = 0.0

    def normalize(self, reward, done):
        self.returns = self.returns * self.gamma + reward
        # Đưa returns về dạng numpy array 0-D để update
        self.return_rms.update(np.array(self.returns, dtype=np.float32))
        std = np.sqrt(self.return_rms.var + 1e-8)
        normalized_reward = reward / std
        if done:
            self.returns = 0.0
        return normalized_reward


def save_checkpoint(agent, running_ms, episode, path, reward_normalizer=None):
    """Lưu trữ trọng số mạng neural và trạng thái chuẩn hóa"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'actor_critic_state_dict': agent.network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'running_ms_mean': running_ms.mean if running_ms else None,
        'running_ms_var': running_ms.var if running_ms else None,
        'running_ms_count': running_ms.count if running_ms else None,
        'reward_rms_mean': reward_normalizer.return_rms.mean if reward_normalizer else None,
        'reward_rms_var': reward_normalizer.return_rms.var if reward_normalizer else None,
        'reward_rms_count': reward_normalizer.return_rms.count if reward_normalizer else None,
        'episode': episode
    }
    torch.save(checkpoint, path)
    print(f"==> Saved checkpoint to: {path}")


def load_checkpoint(agent, running_ms, path, device, reward_normalizer=None):
    """Tải trọng số mạng neural và khôi phục trạng thái chuẩn hóa"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
        
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    agent.network.load_state_dict(checkpoint['actor_critic_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if running_ms and checkpoint.get('running_ms_mean') is not None:
        running_ms.mean = checkpoint['running_ms_mean']
        running_ms.var = checkpoint['running_ms_var']
        running_ms.count = checkpoint['running_ms_count']
        
    if reward_normalizer and checkpoint.get('reward_rms_mean') is not None:
        reward_normalizer.return_rms.mean = checkpoint['reward_rms_mean']
        reward_normalizer.return_rms.var = checkpoint['reward_rms_var']
        reward_normalizer.return_rms.count = checkpoint['reward_rms_count']
        
    print(f"==> Restored checkpoint from: {path} (Episode: {checkpoint['episode']})")
    return checkpoint['episode']
