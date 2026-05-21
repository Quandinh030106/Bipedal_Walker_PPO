import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Khởi tạo các tensors lưu trữ
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        
        # Các mảng tính toán trung gian sau khi kết thúc một chu kỳ tương tác
        self.advantages = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        
        self.ptr = 0
        
    def store(self, state, action, logprob, reward, done, value):
        """Lưu trữ một bước tương tác vào buffer"""
        if self.ptr >= self.buffer_size:
            raise IndexError("Rollout buffer đầy! Hãy huấn luyện để cập nhật mạng trước khi lưu thêm.")
            
        # Chuyển đổi và lưu trữ dưới dạng PyTorch Tensors để đồng bộ GPU nếu có
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.logprobs[self.ptr] = torch.as_tensor(logprob, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        
        self.ptr += 1
        
    def clear(self):
        """Đặt lại bộ đệm sau khi đã cập nhật xong mạng Neural"""
        self.ptr = 0
        
    def compute_returns_and_advantages(self, next_value, next_done, gamma, gae_lambda):
        """
        Tính toán GAE (Generalized Advantage Estimation) và Returns mục tiêu.
        Đây là thuật toán tối ưu hóa ước lượng lợi thế cực kỳ quan trọng giúp giảm phương sai 
        trong các bài toán điều khiển liên tục (Continuous Control).
        """
        # Đảm bảo next_done và next_value là Tensors kiểu float32 dạng 0-D để tránh xung đột chiều (shape mismatch)
        next_value = torch.as_tensor(next_value, dtype=torch.float32, device=self.device)
        next_done = torch.as_tensor(next_done, dtype=torch.float32, device=self.device)
        
        last_gae_lam = 0.0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - next_done
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = self.values[t + 1]
            
            # delta_t = r_t + gamma * V(s_t+1) * (1 - d_t) - V(s_t)
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            
            # GAE: A_t = delta_t + gamma * lambda * (1 - d_t) * A_{t+1}
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
            
        # Target Return = Advantage + Value
        self.returns = self.advantages + self.values
        
    def get_generator(self, minibatch_size):
        """
        Tạo generator trả về ngẫu nhiên các minibatch của dữ liệu đã được làm phẳng (flattened).
        Giúp mạng học không bị thiên kiến (bias) do trật tự bước đi liên tục.
        """
        indices = np.random.permutation(self.buffer_size)
        
        for start_idx in range(0, self.buffer_size, minibatch_size):
            end_idx = start_idx + minibatch_size
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                self.states[batch_indices],
                self.actions[batch_indices],
                self.logprobs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
                self.values[batch_indices]
            )
