# FILE NÀY LÀ ĐỂ DỰNG KHUNG XƯƠNG VÀ KHỚP CHÂN 
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from src.config import Config

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=None):
        super(ActorCritic, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = Config.HIDDEN_DIM
        
        # 1. Định nghĩa mạng Critic: Đầu vào là State (24), Đầu ra là 1 giá trị vô hướng V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 2. Định nghĩa mạng Actor: Đầu vào là State (24), Đầu ra là Mean (4) của hành động
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 3. Standard Deviation cho hành động liên tục.
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # 4. Khởi tạo trọng số mạng bằng thuật toán Orthogonal Initialization (Quan trọng cho PPO ổn định)
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số trực giao với hệ số scale phù hợp"""
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
                
        for layer in self.actor_mean:
            if isinstance(layer, nn.Linear):
                # Actor lớp cuối cùng khởi tạo rất nhỏ (gain=0.01) để hành động ban đầu gần mức 0
                gain = 0.01 if layer == self.actor_mean[-1] else 1.0
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0.0)

    def get_value(self, state):
        """Trả về giá trị ước lượng trạng thái V(s) của mạng Critic"""
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        """
        Hàm cốt lõi tính toán hành động và giá trị V(s) của trạng thái hiện tại.
        Được gọi trong quá trình Agent chạy thử hoặc trong quá trình học.
        """
        # Lấy giá trị trung bình từ mạng Actor
        action_mean = self.actor_mean(state)
        
        # Tính độ lệch chuẩn std = exp(log_std)
        action_std = torch.exp(self.actor_logstd)
        
        # Tạo phân phối chuẩn Gaussian từ mean và std
        dist = Normal(action_mean, action_std)
        
        # Nếu chưa có hành động truyền vào (lúc Agent đang chạy thử và tương tác)
        if action is None:
            action = dist.sample()
            
        # Tính Log Probability (Xác suất log của hành động đó trong phân phối)
        # Đối với BipedalWalker (4 khớp), chúng ta sum giá trị log_prob ở chiều cuối cùng (dim=-1)
        log_prob = dist.log_prob(action).sum(axis=-1)
        
        # Tính Entropy (độ đa dạng của phân phối hành động)
        entropy = dist.entropy().sum(axis=-1)
        
        # Lấy giá trị V(s) từ Critic
        value = self.get_value(state)
        
        return action, log_prob, entropy, value