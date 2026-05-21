"""
强化学习 — 智能体通过"试错+奖励"学会做决策
=============================================
Q-Learning: 学一张 Q表，告诉智能体每种状态下各动作的"价值"

场景: 迷宫寻宝
  状态: 0,1,2,3,4,5（6个位置）
  动作: 0=左, 1=右
  奖励: 到达位置5(宝藏)=+1, 其他=0
  目标: 学会一直往右走去找宝藏
"""
import torch
import torch.nn as nn
import random

# ===== 环境: 6格的线性迷宫 =====
#   [0] [1] [2] [3] [4] [5=宝藏]
#   智能体随机起点，终点=位置5，走到+1分
NUM_STATES = 6
NUM_ACTIONS = 2                              # 0=左, 1=右

# ===== Q表: 6×2 的矩阵，Q[s][a] = 在状态s做动作a的"价值" =====
Q = torch.zeros(NUM_STATES, NUM_ACTIONS)     # 初始全0

# ===== Q-Learning =====
EPSILON = 0.2                                # 20%随机探索，80%选最优
GAMMA = 0.9                                  # 未来奖励折扣
LR = 0.1                                     # 学习率

print("Q表初始值:")
print(Q)

for episode in range(100):
    state = random.randint(0, 4)              # 随机起点（不是宝藏位置）
    done = False

    while not done:
        # ---- ε-greedy 选动作: 80%选最好, 20%随机 ----
        if random.random() < EPSILON:
            action = random.randint(0, 1)     # 探索
        else:
            action = Q[state].argmax().item() # 利用（选当前价值最高的）

        # ---- 执行动作, 观察结果 ----
        if action == 0:                       # 左
            next_state = max(0, state - 1)
        else:                                 # 右
            next_state = min(5, state + 1)

        reward = 1.0 if next_state == 5 else 0.0
        done = (next_state == 5)

        # ---- Q-Learning 核心公式 ----
        # Q[s][a] = Q[s][a] + lr * (奖励 + γ*未来最优 - 当前)
        best_future = Q[next_state].max()
        Q[state, action] += LR * (reward + GAMMA * best_future - Q[state, action])

        state = next_state

    if episode % 25 == 0 and episode > 0:
        print(f"第 {episode} 轮 Q表:\n{Q}")

print(f"\n最终 Q表 (值越大=越应该往那个方向走):")
print(Q)
print(f"\n策略: 位置0→{'右' if Q[0,1]>Q[0,0] else '左'}, "
      f"位置3→{'右' if Q[3,1]>Q[3,0] else '左'}, "
      f"位置4→{'右' if Q[4,1]>Q[4,0] else '左'}")
print("特点: 没标签,靠奖励信号试错,学的是'做什么动作能拿奖励'")
