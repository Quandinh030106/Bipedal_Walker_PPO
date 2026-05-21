# 🤖 Bipedal Walker PPO

Nghiên cứu và triển khai thuật toán **Proximal Policy Optimization (PPO)** từ đầu (from scratch) bằng PyTorch để huấn luyện robot hai chân tự học đi bộ trên môi trường vật lý `BipedalWalker-v3` (Gymnasium Box2D), kết hợp Web Dashboard trực quan bằng FastAPI.

---

## Pipeline tổng quan
```
[Môi trường BipedalWalker-v3]
        ↓ state (24 sensors)
[Actor-Critic Network]  ←─── PPO Update (mỗi 2048 steps)
        ↓ action (4 joints)          ↑
[Rollout Buffer + GAE] ──────────────┘
        ↓
[TensorBoard Logs] + [Checkpoint .pth]
        ↓
[Evaluate & Record Video]
        ↓
[FastAPI Web Dashboard]
```

**Luồng huấn luyện:**
1. Robot quan sát môi trường → Actor xuất hành động theo phân phối Gaussian
2. Tương tác với môi trường → nhận phần thưởng → lưu vào Rollout Buffer
3. Sau 2048 bước: tính GAE advantages → cập nhật mạng PPO (10 epochs × minibatch)
4. Lặp lại đến 2M steps, lưu checkpoint tốt nhất tự động

---

## Kết quả
|
 Chỉ số 
|
 Giá trị 
|
|
---
|
---
|
|
 Mean Reward (evaluate 3 episodes) 
|
**
~251 / 300
**
|
|
 Độ ổn định 
|
 ±7 (243 / 253 / 257) 
|
|
 Benchmark "Solved" chính thức 
|
 Mean ≥ 300 
|

---

## Cài đặt & Chạy
```bash
pip install -r requirements.txt
python src/train.py
```
---

## Cấu trúc thư mục
```
├── src/
│   ├── config.py          # Hyperparameters
│   ├── actor_critic.py    # Mạng Neural Actor-Critic
│   ├── buffer.py          # Rollout Buffer & GAE
│   ├── agent.py           # Logic cập nhật PPO
│   ├── utils.py           # Normalization & Checkpoint
│   └── train.py           # Vòng lặp huấn luyện
├── evaluate.py            # Đánh giá & xuất video
├── web_app/               # FastAPI Dashboard
├── models/                # Trọng số mô hình (.pth)
└── runs/                  # TensorBoard logs
```

---

## Tài liệu tham khảo
- Schulman et al. (2017) — [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347)
- Schulman et al. (2015) — [*High-Dimensional Continuous Control Using GAE*](https://arxiv.org/abs/1506.02438)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — PPO reference implementation
- Gymnasium BipedalWalker — [https://gymnasium.farama.org/environments/box2d/bipedal_walker](https://gymnasium.farama.org/environments/box2d/bipedal_walker)