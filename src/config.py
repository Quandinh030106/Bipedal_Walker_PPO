#FILE NÀY LÀ ĐỂ THIẾT LẬP MÔI TRƯỜNG CHO BIPEDAL_WALKER_v3 VÀ CÓ CÁC THAM SỐ
import torch

class Config:
    # Môi trường
    ENV_NAME = "BipedalWalker-v3"
    
    # Kiến trúc mạng
    STATE_DIM = 24       # 24 cảm biến đầu vào của robot
    ACTION_DIM = 4       # 4 khớp động cơ điều khiển chân
    HIDDEN_DIM = 256     # Kích thước lớp ẩn trong mạng Neural (Tăng từ 64 để tránh underfitting)
    
    # PPO Hyperparameters
    SEED = 42
    TOTAL_TIMESTEPS = 2_000_000  # Chạy 2 triệu bước (BipedalWalker cần khoảng 1.5M - 2M để đi vững)
    LEARNING_RATE = 3e-4
    NUM_STEPS = 2048             # Chiều dài của rollout buffer thu thập trước khi update
    MINIBATCH_SIZE = 64          # Kích thước một minibatch để cập nhật mạng
    UPDATE_EPOCHS = 10           # Số lần học lại trên cùng một tập rollout
    
    # Các kỹ thuật tối ưu hóa độ ổn định
    NORMALIZE_OBS = True         # Chuẩn hóa quan sát đầu vào (Quan trọng đối với BipedalWalker)
    NORMALIZE_REWARDS = True     # Chuẩn hóa phần thưởng
    ANNEAL_LR = True             # Giảm dần tốc độ học tuyến tính về 0
    CLIP_OBS = 10.0              # Giới hạn giá trị state sau khi chuẩn hóa
    
    # Các hệ số toán học trong PPO
    GAMMA = 0.99                 # Hệ số chiết khấu phần thưởng tương lai
    GAE_LAMBDA = 0.95            # Hệ số Lamda trong GAE
    CLIP_COEF = 0.2              # Hệ số epsilon để cắt (clipping) tỷ lệ ratio
    ENT_COEF = 0.001             # Hệ số Entropy khuyến khích robot thử nghiệm thế đi mới
    VF_COEF = 0.5                # Hệ số Value Loss (cự ly sai lệch mạng Critic)
    MAX_GRAD_NORM = 0.5          # Giới hạn Gradient để tránh bùng nổ gradient
    CLIP_VALUE_LOSS = False      # Tắt việc giới hạn Value Loss của Critic (mặc định đặt là False giúp học nhanh phạt ngã -100)
    
    
    # Thiết bị chạy
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")