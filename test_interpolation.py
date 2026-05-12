#!/usr/bin/env python3
"""
NGSIM插值效果测试脚本

用途：验证线性插值是否正确消除车辆抖动
运行：python test_interpolation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def simulate_without_interpolation(duration: float = 1.0, 
                                   carla_dt: float = 0.05, 
                                   ngsim_dt: float = 0.1) -> Tuple[List[float], List[float]]:
    """模拟修复前：无插值，车辆抖动"""
    times = []
    positions = []
    
    # NGSIM数据：每0.1秒更新一次位置
    ngsim_frames = {
        0.0: 0.0,
        0.1: 10.0,
        0.2: 20.0,
        0.3: 30.0,
        0.4: 40.0,
        0.5: 50.0,
        0.6: 60.0,
        0.7: 70.0,
        0.8: 80.0,
        0.9: 90.0,
        1.0: 100.0,
    }
    
    current_time = 0.0
    while current_time <= duration:
        times.append(current_time)
        
        # 找到当前时间对应的NGSIM帧（向下取整）
        ngsim_time = (int(current_time / ngsim_dt)) * ngsim_dt
        position = ngsim_frames.get(ngsim_time, 0.0)
        positions.append(position)
        
        current_time += carla_dt
    
    return times, positions

def simulate_with_interpolation(duration: float = 1.0,
                               carla_dt: float = 0.05,
                               ngsim_dt: float = 0.1) -> Tuple[List[float], List[float]]:
    """模拟修复后：线性插值，车辆平滑移动"""
    times = []
    positions = []
    
    # NGSIM数据：每0.1秒更新一次位置
    ngsim_frames = {
        0.0: 0.0,
        0.1: 10.0,
        0.2: 20.0,
        0.3: 30.0,
        0.4: 40.0,
        0.5: 50.0,
        0.6: 60.0,
        0.7: 70.0,
        0.8: 80.0,
        0.9: 90.0,
        1.0: 100.0,
    }
    
    current_time = 0.0
    while current_time <= duration:
        times.append(current_time)
        
        # 计算插值因子
        frame_offset_float = current_time / ngsim_dt
        current_frame_idx = int(frame_offset_float)
        interpolation_factor = frame_offset_float - current_frame_idx
        
        # 当前帧和下一帧的时间
        current_ngsim_time = current_frame_idx * ngsim_dt
        next_ngsim_time = (current_frame_idx + 1) * ngsim_dt
        
        # 获取位置数据
        current_pos = ngsim_frames.get(current_ngsim_time, 0.0)
        next_pos = ngsim_frames.get(next_ngsim_time, current_pos)
        
        # 线性插值
        interpolated_pos = current_pos + interpolation_factor * (next_pos - current_pos)
        positions.append(interpolated_pos)
        
        current_time += carla_dt
    
    return times, positions

def calculate_smoothness_metrics(positions: List[float]) -> dict:
    """计算平滑度指标"""
    # 计算位置变化（速度）
    velocities = np.diff(positions)
    
    # 计算加速度（速度变化）
    accelerations = np.diff(velocities)
    
    # 平滑度指标
    metrics = {
        'velocity_std': np.std(velocities),  # 速度标准差（越小越平滑）
        'acceleration_std': np.std(accelerations),  # 加速度标准差（越小越平滑）
        'max_jerk': np.max(np.abs(np.diff(accelerations))),  # 最大加加速度（抖动）
        'velocity_mean': np.mean(velocities),  # 平均速度
    }
    
    return metrics

def plot_comparison():
    """绘制对比图"""
    # 模拟数据
    times_old, positions_old = simulate_without_interpolation()
    times_new, positions_new = simulate_with_interpolation()
    
    # 计算平滑度指标
    metrics_old = calculate_smoothness_metrics(positions_old)
    metrics_new = calculate_smoothness_metrics(positions_new)
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 子图1: 位置-时间曲线
    ax1 = axes[0]
    ax1.plot(times_old, positions_old, 'ro-', label='修复前（抖动）', linewidth=2, markersize=6)
    ax1.plot(times_new, positions_new, 'b.-', label='修复后（插值）', linewidth=2, markersize=4)
    ax1.set_xlabel('时间 (秒)', fontsize=12)
    ax1.set_ylabel('位置 (米)', fontsize=12)
    ax1.set_title('NGSIM车辆位置对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 速度-时间曲线
    ax2 = axes[1]
    velocities_old = np.diff(positions_old) / 0.05
    velocities_new = np.diff(positions_new) / 0.05
    times_velocity = times_old[:-1]
    
    ax2.plot(times_velocity, velocities_old, 'ro-', label='修复前', linewidth=2, markersize=6)
    ax2.plot(times_velocity, velocities_new, 'b.-', label='修复后', linewidth=2, markersize=4)
    ax2.set_xlabel('时间 (秒)', fontsize=12)
    ax2.set_ylabel('速度 (米/秒)', fontsize=12)
    ax2.set_title('速度对比（标准差：修复前={:.2f}, 修复后={:.2f})'.format(
        metrics_old['velocity_std'], metrics_new['velocity_std']), 
        fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 加速度-时间曲线
    ax3 = axes[2]
    accelerations_old = np.diff(velocities_old) / 0.05
    accelerations_new = np.diff(velocities_new) / 0.05
    times_acceleration = times_old[:-2]
    
    ax3.plot(times_acceleration, accelerations_old, 'ro-', label='修复前（抖动明显）', linewidth=2, markersize=6)
    ax3.plot(times_acceleration, accelerations_new, 'b.-', label='修复后（平滑）', linewidth=2, markersize=4)
    ax3.set_xlabel('时间 (秒)', fontsize=12)
    ax3.set_ylabel('加速度 (米/秒²)', fontsize=12)
    ax3.set_title('加速度对比（标准差：修复前={:.2f}, 修复后={:.2f})'.format(
        metrics_old['acceleration_std'], metrics_new['acceleration_std']),
        fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 对比图已保存: interpolation_comparison.png")
    plt.show()
    
    # 打印详细指标
    print("\n" + "="*60)
    print("平滑度指标对比")
    print("="*60)
    print(f"{'指标':<25} {'修复前':>15} {'修复后':>15}")
    print("-"*60)
    print(f"{'速度标准差 (m/s)':<25} {metrics_old['velocity_std']:>15.4f} {metrics_new['velocity_std']:>15.4f}")
    print(f"{'加速度标准差 (m/s²)':<25} {metrics_old['acceleration_std']:>15.4f} {metrics_new['acceleration_std']:>15.4f}")
    print(f"{'最大抖动 (m/s³)':<25} {metrics_old['max_jerk']:>15.4f} {metrics_new['max_jerk']:>15.4f}")
    print(f"{'平均速度 (m/s)':<25} {metrics_old['velocity_mean']:>15.4f} {metrics_new['velocity_mean']:>15.4f}")
    print("="*60)
    
    # 改善百分比
    improvement_velocity = (1 - metrics_new['velocity_std'] / metrics_old['velocity_std']) * 100
    improvement_accel = (1 - metrics_new['acceleration_std'] / metrics_old['acceleration_std']) * 100
    improvement_jerk = (1 - metrics_new['max_jerk'] / metrics_old['max_jerk']) * 100
    
    print(f"\n✨ 改善效果:")
    print(f"   - 速度平滑度提升: {improvement_velocity:.1f}%")
    print(f"   - 加速度平滑度提升: {improvement_accel:.1f}%")
    print(f"   - 抖动减少: {improvement_jerk:.1f}%")
    print("="*60 + "\n")

def test_interpolation_function():
    """测试插值函数的正确性"""
    print("\n" + "="*60)
    print("插值函数测试")
    print("="*60)
    
    # 测试用例
    test_cases = [
        (0.0, 0.0, 10.0, 0.0, "起始点"),
        (0.5, 0.0, 10.0, 5.0, "中点"),
        (1.0, 0.0, 10.0, 10.0, "结束点"),
        (0.25, 0.0, 10.0, 2.5, "1/4点"),
        (0.75, 0.0, 10.0, 7.5, "3/4点"),
    ]
    
    all_passed = True
    for factor, current, next_val, expected, desc in test_cases:
        result = current + factor * (next_val - current)
        passed = abs(result - expected) < 0.001
        status = "✅" if passed else "❌"
        print(f"{status} {desc}: factor={factor:.2f}, result={result:.2f}, expected={expected:.2f}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("✅ 所有测试通过！\n")
    else:
        print("❌ 部分测试失败！\n")
    
    return all_passed

def main():
    """主函数"""
    print("\n" + "="*60)
    print("NGSIM插值效果测试")
    print("="*60)
    print("本脚本验证线性插值是否正确消除车辆抖动\n")
    
    # 测试插值函数
    test_interpolation_function()
    
    # 绘制对比图
    try:
        plot_comparison()
    except ImportError as e:
        print(f"\n⚠️ 无法绘制图表: {e}")
        print("请安装matplotlib: pip install matplotlib\n")
        
        # 仍然打印数值结果
        times_old, positions_old = simulate_without_interpolation()
        times_new, positions_new = simulate_with_interpolation()
        metrics_old = calculate_smoothness_metrics(positions_old)
        metrics_new = calculate_smoothness_metrics(positions_new)
        
        print("\n" + "="*60)
        print("平滑度指标对比（无图表）")
        print("="*60)
        print(f"修复前 - 速度标准差: {metrics_old['velocity_std']:.4f}")
        print(f"修复后 - 速度标准差: {metrics_new['velocity_std']:.4f}")
        print(f"改善: {(1 - metrics_new['velocity_std'] / metrics_old['velocity_std']) * 100:.1f}%")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()

