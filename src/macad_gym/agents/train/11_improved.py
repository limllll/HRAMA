import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 读取数据文件
def read_rewards_data(filename):
    episodes = []
    rewards = []

    with open(filename, 'r') as file:
        for line in file:
            # 跳过空行和非数据行
            if line.strip() and not line.startswith('LaneChange') and not line.startswith('<'):
                parts = line.strip().split(",")
                # 每行应该有两个数字：回合数和奖励值
                if len(parts) == 2:
                    episode = int(parts[0])
                    reward = float(parts[1])
                    episodes.append(episode)
                    rewards.append(reward)

    return episodes, rewards


# 平滑处理函数
def smooth_data(rewards, window_size=50):
    """使用移动平均平滑数据"""
    return pd.Series(rewards).rolling(window=window_size, center=True).mean()


# 绘制包含原始数据和平滑曲线的图表
def plot_reward_curve_with_original(episodes, rewards, smooth_window=50):
    plt.figure(figsize=(12, 6))

    # 原始奖励曲线
    plt.plot(episodes, rewards, 'b-', alpha=0.3, label='origin reward')

    # 平滑后的曲线
    smoothed_rewards = smooth_data(rewards, smooth_window)
    plt.plot(episodes, smoothed_rewards, 'r-', linewidth=2, label=f'smooth (window={smooth_window})')

    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('PPO Train Reward (with Original Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 绘制只包含平滑曲线的图表
def plot_smoothed_reward_curve_only(episodes, rewards, smooth_window=50):
    plt.figure(figsize=(12, 6))

    # 只绘制平滑后的曲线
    smoothed_rewards = smooth_data(rewards, smooth_window)
    plt.plot(episodes, smoothed_rewards, 'r-', linewidth=2, label=f'smoothed reward (window={smooth_window})')
    plt.ylim(0,120)
    plt.yticks([0,20,40,60,80,100,120])

    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('PPO Train Reward (Smoothed Only)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 读取数据
    episodes, rewards = read_rewards_data('./LaneChange_PPO_Rewards_inference_full.csv')

    print(f"数据点数: {len(episodes)}")
    print(f"回合范围: {min(episodes)} - {max(episodes)}")

    # 绘制两张图表
    # 1. 原始图表（包含原始数据和平滑曲线）
    plot_reward_curve_with_original(episodes, rewards, smooth_window=100)

    # 2. 新图表（只包含平滑后的曲线）
    plot_smoothed_reward_curve_only(episodes, rewards, smooth_window=200)