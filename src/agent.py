import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.config import Config
from src.actor_critic import ActorCritic

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = Config.DEVICE
        
        # Khởi tạo mạng Actor-Critic
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        
        # Sử dụng Adam optimizer với tham số eps giúp tối ưu hóa số học ổn định hơn
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=Config.LEARNING_RATE, 
            eps=1e-5
        )

    def select_action(self, state):
        """
        Lấy hành động từ mạng Neural Actor dựa trên trạng thái hiện tại.
        Hỗ trợ cả trạng thái đơn lẻ (1D) và trạng thái theo lô (batched 2D).
        """
        state = np.array(state, dtype=np.float32)
        has_batch = len(state.shape) > 1
        
        # Thêm chiều batch nếu là trạng thái đơn lẻ (1D)
        if not has_batch:
            state_input = np.expand_dims(state, axis=0)
        else:
            state_input = state
            
        state_t = torch.as_tensor(state_input, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(state_t)
            
        if not has_batch:
            # Loại bỏ chiều batch để trả về mảng 1D cho môi trường Gymnasium
            return (
                action.squeeze(0).cpu().numpy(),
                log_prob.squeeze(0).cpu().numpy(),
                entropy.squeeze(0).cpu().numpy(),
                value.squeeze(0).cpu().numpy()
            )
        else:
            return (
                action.cpu().numpy(),
                log_prob.cpu().numpy(),
                entropy.cpu().numpy(),
                value.cpu().numpy()
            )

    def update(self, buffer, lr_now=None):
        """
        Cập nhật trọng số mạng Neural PPO sử dụng dữ liệu từ RolloutBuffer.
        Hỗ trợ giảm tốc độ học động (lr_now) nếu cấu hình bật ANNEAL_LR.
        """
        # 1. Cập nhật tốc độ học nếu có truyền vào lr_now
        if lr_now is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_now

        # Lưu lại các chỉ số thống kê tổn thất (Loss) để vẽ biểu đồ
        epoch_pg_loss = 0.0
        epoch_v_loss = 0.0
        epoch_ent_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        # 2. Học lại qua nhiều epochs trên cùng dữ liệu rollout thu thập được
        for epoch in range(Config.UPDATE_EPOCHS):
            # Lấy các minibatch dữ liệu đã được xáo trộn ngẫu nhiên từ buffer
            batch_generator = buffer.get_generator(Config.MINIBATCH_SIZE)
            
            for mb_states, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values in batch_generator:
                # 3. Tính toán lại logprobs, entropy và value của hành động cũ dựa trên mạng mới hiện tại
                _, new_logprob, entropy, new_value = self.network.get_action_and_value(mb_states, mb_actions)
                new_value = new_value.view(-1)
                
                # 4. Tính toán tỷ lệ chính sách (Ratio): r_t(theta) = pi_new(a|s) / pi_old(a|s)
                logratio = new_logprob - mb_logprobs
                ratio = torch.exp(logratio)
                
                # 5. Chuẩn hóa Advantage (Ưu thế) trong minibatch để huấn luyện mượt mà hơn
                advantages = mb_advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 6. Tính toán Clipped Policy Loss (Tổn thất chính sách có cắt)
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - Config.CLIP_COEF, 1.0 + Config.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # 7. Tính toán Value Loss (Critic Loss) có hoặc không có clipping
                # Tắt clipping (Config.CLIP_VALUE_LOSS = False) giúp Critic học nhanh các phần thưởng đột ngột (phạt ngã -100)
                if Config.CLIP_VALUE_LOSS:
                    v_loss_unclipped = (new_value - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(new_value - mb_values, -Config.CLIP_COEF, Config.CLIP_COEF)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    value_loss = 0.5 * v_loss_max.mean()
                else:
                    value_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                
                # 8. Tính toán Entropy Loss (Độ hỗn loạn để tăng khám phá địa hình mới)
                entropy_loss = entropy.mean()
                
                # 9. Tổng tổn thất (Total Loss)
                loss = pg_loss + Config.VF_COEF * value_loss - Config.ENT_COEF * entropy_loss
                
                # 10. Tối ưu hóa và lan truyền ngược
                self.optimizer.zero_grad()
                loss.backward()
                # Giới hạn Gradient (Gradient Clipping) để tránh bùng nổ gradient
                nn.utils.clip_grad_norm_(self.network.parameters(), Config.MAX_GRAD_NORM)
                self.optimizer.step()
                
                # Cộng dồn thống kê
                epoch_pg_loss += pg_loss.item()
                epoch_v_loss += value_loss.item()
                epoch_ent_loss += entropy_loss.item()
                epoch_total_loss += loss.item()
                num_batches += 1
                
        # Trả về trung bình các giá trị tổn thất của đợt cập nhật này
        return {
            'policy_loss': epoch_pg_loss / num_batches,
            'value_loss': epoch_v_loss / num_batches,
            'entropy_loss': epoch_ent_loss / num_batches,
            'total_loss': epoch_total_loss / num_batches
        }
