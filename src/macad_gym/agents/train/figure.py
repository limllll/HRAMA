import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取数据
episodes = []
rewards = []

with open('Tjunction_ppo_LSTM.csv', 'r') as file:
    for line in file:
        parts = line.split(',')
        if len(parts) >= 2:
            # 提取轮次和奖励值，忽略多余的数字
            episode = parts[0].strip()
            reward = parts[1].strip()
            # 尝试将字符串转换为浮点数
            try:
                episode_num = float(episode)
                reward_num = float(reward)
                episodes.append(episode_num)
                rewards.append(reward_num)
            except ValueError:
                continue  # 跳过无法转换的行

# 转换为numpy数组
episodes = np.array(episodes)
rewards = np.array(rewards)

# 计算移动平均值（窗口大小为10）
window_size = 10
moving_averages = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

# 绘制原始数据和移动平均值
plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, alpha=0.5, label='Original Rewards')
plt.plot(episodes[:len(moving_averages)], moving_averages, label='Moving Average (window=10)', color='red')
plt.title('PPO Algorithm Episode Rewards with Moving Average')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.grid(True)
# plt.yticks(np.arange(-100, 220, 20))
plt.show()