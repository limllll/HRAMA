import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def read_rewards_data(filename: str):
    episodes = []
    rewards = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip() and not line.startswith("LaneChange") and not line.startswith("<"):
                parts = line.strip().split(",")
                if len(parts) == 2:
                    episodes.append(int(parts[0]))
                    rewards.append(float(parts[1]))

    return episodes, rewards


def smooth_data(rewards, window_size=50):
    # 使用中心窗口，并允许窗口未满时也计算（两端仍有平滑值）
    return (
        pd.Series(rewards)
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
    )


def truncate_by_max_episode(episodes, rewards, max_episode: int):
    if max_episode is None:
        return episodes, rewards
    truncated = [(e, r) for e, r in zip(episodes, rewards) if e <= max_episode]
    if not truncated:
        return [], []
    eps, rws = zip(*truncated)
    return list(eps), list(rws)


def plot_four_reward_curves(
    series,
    smooth_window=200,
    show_original=False,
    ylim=None,
    title="",
):
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # 背景颜色与示例图类似的浅灰蓝色
    ax.set_facecolor("#f2f2f7")
    # 去掉所有边界黑线（四条 spine）
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)

    # 不同线型，仿照示例图：点线、虚线、实线等
    linestyles = [":", "--", "-.", "-"]
    markers = ["", "", "", ""]

    for idx, item in enumerate(series):
        episodes, rewards = item["episodes"], item["rewards"]
        label = item["label"]
        color = item.get("color", None)
        ls = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]

        if show_original:
            plt.plot(
                episodes,
                rewards,
                alpha=0.15,
                linewidth=1,
                color=color,
                linestyle=ls,
                marker=marker,
                label=f"{label} (origin)",
            )

        smoothed_rewards = smooth_data(rewards, smooth_window)
        plt.plot(
            episodes,
            smoothed_rewards,
            linewidth=2,
            color=color,
            linestyle=ls,
            marker=marker,
            label=label,
        )

    if ylim is not None:
        plt.ylim(*ylim)

    # 固定 x 轴到 [0, max_episode]，防止自动缩放不显示 1500
    # 这里假设所有 series 的最大 episode 相同或接近
    all_eps = [e for item in series for e in item["episodes"]]
    if all_eps:
        xmax = max(all_eps)
        # 数据只到 xmax（例如 1500），但坐标轴右侧留白 20
        plt.xlim(0, xmax + 20)
        # 每 250 作为一个刻度，确保右端刻度“刚好”到 xmax（如 1500）
        step = 250
        plt.xticks(range(0, int(xmax) + 1, step))

    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    # 不显示图标题（按需可以在调用时传入）
    if title:
        plt.title(title)
    plt.legend()
    # 使用较浅的网格颜色，风格接近你给的示例图
    plt.grid(True, color="#d0d0e0", alpha=0.7, linestyle="-", linewidth=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    max_episode = 1500
    files = [
        # 控制绘制顺序 = 图例顺序
        (here / "LaneChange_PPO_Rewards_inference_image_only.csv", "image", "#2ca02c"),
        (here / "LaneChange_PPO_Rewards_inference_image_state.csv", "image+state", "#1f77b4"),
        (here / "LaneChange_PPO_Rewards_inference_image_state_lidar.csv", "image+state+lidar", "#ff7f0e"),
        (here / "LaneChange_PPO_Rewards_inference_full.csv", "HRAMA", "#d62728"),
    ]

    series = []
    for path, label, color in files:
        episodes, rewards = read_rewards_data(str(path))
        episodes, rewards = truncate_by_max_episode(episodes, rewards, max_episode=max_episode)
        print(
            f"{label}: points={len(episodes)}, "
            f"episode_range={(min(episodes) if episodes else 'n/a')}-{(max(episodes) if episodes else 'n/a')}"
        )
        series.append({"episodes": episodes, "rewards": rewards, "label": label, "color": color})

    plot_four_reward_curves(
        series,
        smooth_window=150,
        show_original=False,
        ylim=(0, 120),
        title="",
    )
