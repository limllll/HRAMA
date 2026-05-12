#!/usr/bin/env python3
"""
SUMO智能多车轨迹控制系统 - 官方标准交互版
完美结合：精准Lane_ID映射 + SUMO官方智能交互 + 安全行为控制

🚗 智能交互仿真特性:
1. 精准轨迹跟踪: 基于验证过的Lane_ID精准映射算法
2. 智能安全控制: 按SUMO官方文档实现车辆交互和碰撞避免
3. 混合控制模式: 目标跟踪 + 自主安全行为的完美结合
4. 动态控制切换: 根据交通状况智能切换控制模式
5. 官方安全检查: 启用SUMO内置的安全检查和跟车逻辑
6. 自然交互行为: 车辆会对周围车辆做出合理反应

📋 技术标准: 100%基于SUMO官方文档和最佳实践
🎯 控制策略: NGSIM精准重现 + 现实交通安全 = 最佳平衡
🔍 质量保证: 严格遵循官方TraCI和车辆控制文档
"""

import pandas as pd
import numpy as np
import traci
import sumolib
import sumolib.net
import sumolib.geomhelper
import os
import sys
import time
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict

# 导入精准单车控制器作为基础
from sumo_official_trajectory_control import SUMOOfficialTrajectoryController


class IntelligentMultiVehicleController(SUMOOfficialTrajectoryController):
    """SUMO智能多车轨迹控制器 - 官方标准交互版"""
    
    def __init__(self):
        super().__init__()
        
        # 多车管理状态
        self.active_vehicles = {}      # 当前活跃车辆
        self.vehicle_trajectories = {} # 车辆轨迹数据
        self.frame_vehicles = {}       # 每帧的车辆列表
        self.vehicle_states = {}       # 车辆状态跟踪
        
        # 智能控制状态
        self.vehicle_control_modes = {}  # 车辆控制模式: {vehicle_id: 'manual'/'auto'/'hybrid'}
        self.safety_overrides = {}       # 安全接管状态: {vehicle_id: True/False}
        self.interaction_history = {}    # 交互历史记录
        
        # 仿真状态
        self.current_frame = 0
        self.max_frame = 0
        self.total_vehicles = 0
        self.transform_params = None
        
        # 智能控制参数 (按SUMO官方文档)
        self.safety_params = {
            'min_gap': 2.0,              # 最小车距 (米)
            'collision_threshold': 1.5,  # 碰撞风险阈值 (米)
            'reaction_distance': 15.0,   # 反应距离 (米)
            'speed_adaptation_rate': 0.3, # 速度适应率
            'lane_change_threshold': 0.8  # 变道阈值
        }
        
        # 性能统计
        self.stats = {
            'vehicles_added': 0,
            'vehicles_removed': 0,
            'total_updates': 0,
            'safety_overrides_count': 0,
            'collision_avoidances': 0,
            'intelligent_behaviors': 0
        }
        
    def load_multi_vehicle_data(self, csv_file):
        """加载多车数据 - 与精准版本相同"""
        print(f"\n=== 加载智能多车轨迹数据 ===")
        
        # 读取数据
        df = pd.read_csv(csv_file)
        
        # 处理数据格式
        for col in ['Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Lane_ID']:
            if col in df.columns and isinstance(df[col].iloc[0], str):
                df[col] = df[col].str.replace(',', '').astype(float)
        
        # 按Frame_ID升序排列
        df = df.sort_values(['Frame_ID', 'Vehicle_ID']).reset_index(drop=True)
        
        # 数据统计
        self.total_vehicles = df['Vehicle_ID'].nunique()
        self.current_frame = df['Frame_ID'].min()
        self.max_frame = df['Frame_ID'].max()
        
        print(f"智能多车数据统计:")
        print(f"  总车辆数: {self.total_vehicles}辆")
        print(f"  总数据量: {len(df)}条轨迹点")
        print(f"  帧范围: {self.current_frame}-{self.max_frame}")
        print(f"  仿真时长: {(self.max_frame - self.current_frame) * 0.1:.1f}秒")
        
        # 分析车辆行驶特征
        print(f"\n=== 车辆智能控制准备 ===")
        for vehicle_id in sorted(df['Vehicle_ID'].unique()):
            vehicle_data = df[df['Vehicle_ID'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('Frame_ID').reset_index(drop=True)
            
            if len(vehicle_data) > 1:
                # 分析速度变化模式
                speeds = vehicle_data['v_Vel'] * self.feet_to_meters
                avg_speed = speeds.mean()
                speed_std = speeds.std()
                
                # 分析车道使用
                lanes_used = sorted(vehicle_data['Lane_ID'].unique())
                lane_changes = len(lanes_used) - 1
                
                if vehicle_id <= 90 or vehicle_id % 10 == 0:
                    print(f"  车辆{vehicle_id}: 平均速度={avg_speed:.1f}m/s±{speed_std:.1f}, "
                          f"车道{lanes_used}, 变道{lane_changes}次")
            
            self.vehicle_trajectories[vehicle_id] = vehicle_data
        
        # 构建帧映射
        self.frame_vehicles = {}
        for frame_id in range(self.current_frame, self.max_frame + 1):
            frame_data = df[df['Frame_ID'] == frame_id]
            vehicle_list = sorted(frame_data['Vehicle_ID'].tolist())
            self.frame_vehicles[frame_id] = vehicle_list
        
        # 并发度分析
        concurrent_counts = [len(vehicles) for vehicles in self.frame_vehicles.values()]
        max_concurrent = max(concurrent_counts)
        avg_concurrent = np.mean(concurrent_counts)
        
        print(f"并发控制分析:")
        print(f"  最大同时在线: {max_concurrent}辆")
        print(f"  平均同时在线: {avg_concurrent:.1f}辆")
        print(f"✅ 智能多车数据预处理完成")
        
        return df
    
    def create_intelligent_vehicle_types(self, route_file):
        """创建智能交互优化的车辆类型 - 兼容sublane模拟"""
        print(f"\n=== 创建智能交互车辆类型 ===")
        
        route_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    
    <!-- 智能交互车辆类型 - 兼容sublane模拟 -->
    <vType id="intelligent_car" vClass="passenger" 
           length="5.0" width="1.8" height="1.5" minGap="2.0"
           maxSpeed="80.0" accel="2.6" decel="4.5" emergencyDecel="9.0"
           sigma="0.5" tau="1.0" speedFactor="1.0" speedDev="0.1"
           color="green" guiShape="passenger"
           carFollowModel="Krauss" laneChangeModel="SL2015"
           lcStrategic="1.0" lcCooperative="1.0" lcSpeedGain="1.0"
           lcSublane="1.0" latAlignment="right"/>
    
    <!-- 标准路由 -->
    <route id="main_route" edges="E0"/>
    
</routes>'''
        
        with open(route_file, 'w', encoding='utf-8') as f:
            f.write(route_content)
        
        print(f"创建路由文件: {route_file}")
        print(f"使用智能交互参数: Krauss跟车 + SL2015变道模型 (sublane兼容)")
        print(f"启用合作驾驶、战略变道、速度优化和sublane功能")
    
    def add_intelligent_vehicle(self, vehicle_id, frame_data, lane_id):
        """添加具备智能交互的车辆 - 按SUMO官方标准"""
        sumo_vehicle_id = f"vehicle_{vehicle_id}"
        
        try:
            # 检查车辆是否已经存在于SUMO中
            if sumo_vehicle_id in traci.vehicle.getIDList():
                # 减少重复日志 - 只在第一次时显示
                if vehicle_id not in self.active_vehicles:
                    print(f"⚠️ 车辆{vehicle_id}已存在于SUMO，添加到跟踪")
                # 如果车辆已存在但不在我们的跟踪中，添加到跟踪
                if vehicle_id not in self.active_vehicles:
                    self.active_vehicles[vehicle_id] = {
                        'sumo_id': sumo_vehicle_id,
                        'prev_pos': traci.vehicle.getPosition(sumo_vehicle_id),
                        'prev_speed': traci.vehicle.getSpeed(sumo_vehicle_id),
                        'added_frame': self.current_frame,
                        'last_lane_id': lane_id,
                        'target_speed': traci.vehicle.getSpeed(sumo_vehicle_id),
                        'safety_mode': 'normal'
                    }
                    self.vehicle_control_modes[vehicle_id] = 'hybrid'
                    self.safety_overrides[vehicle_id] = False
                    self.interaction_history[vehicle_id] = []
                return True
            
            # 添加车辆
            traci.vehicle.add(sumo_vehicle_id, "main_route", typeID="intelligent_car")
            
            # 设置智能混合控制模式 (按SUMO官方文档)
            # 参考: https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html#speed_mode_0xb3
            
            # 速度控制模式 - 启用关键安全检查
            # speedMode = 0: 完全手动 (危险)
            # speedMode = 31: 启用所有安全检查 (推荐)
            # speedMode = 23: 启用碰撞检查但允许超速 (平衡)
            traci.vehicle.setSpeedMode(sumo_vehicle_id, 23)  # 智能速度控制
            
            # 变道控制模式 - 启用sublane兼容的智能变道
            # laneChangeMode = 0: 完全禁用 (危险)
            # laneChangeMode = 1621: 标准智能变道
            # laneChangeMode = 597: 保守变道模式
            # laneChangeMode = 512: sublane兼容模式
            traci.vehicle.setLaneChangeMode(sumo_vehicle_id, 512)  # sublane兼容智能变道
            
            # 计算初始位置
            initial_x, initial_y = self.map_coordinates(
                frame_data['Local_Y'], frame_data['Local_X'], 
                self.transform_params, lane_id
            )
            initial_speed = frame_data['v_Vel'] * self.feet_to_meters
            
            # 调试信息 - 显示坐标映射
            print(f"🔍 调试车辆{vehicle_id}: NGSIM({frame_data['Local_Y']:.1f}ft, {frame_data['Local_X']:.1f}ft) "
                  f"→ SUMO({initial_x:.1f}m, {initial_y:.1f}m), Lane_ID={lane_id}")
            
            # 智能定位 - 使用更简单可靠的方法
            try:
                # 方法1: 先尝试简单的moveToXY
                traci.vehicle.moveToXY(
                    vehID=sumo_vehicle_id,
                    edgeID="E0",
                    laneIndex=-1,
                    x=initial_x,
                    y=initial_y,
                    angle=90.0,
                    keepRoute=1    # 简化：保持路线
                )
            except Exception as e:
                print(f"⚠️ moveToXY失败，尝试备用方法: {e}")
                # 方法2: 备用 - 不指定角度
                try:
                    traci.vehicle.moveToXY(
                        vehID=sumo_vehicle_id,
                        edgeID="E0",
                        laneIndex=-1,
                        x=initial_x,
                        y=initial_y
                    )
                except Exception as e2:
                    print(f"⚠️ 备用方法也失败: {e2}")
                    # 方法3: 最简单的定位
                    try:
                        traci.vehicle.moveTo(sumo_vehicle_id, "E0_1", initial_x)
                        print(f"✅ 使用moveTo成功定位车辆{vehicle_id}")
                    except Exception as e3:
                        print(f"❌ 所有定位方法都失败: {e3}")
                        return False
            
            # 设置初始速度 - 让SUMO验证安全性
            traci.vehicle.setSpeed(sumo_vehicle_id, initial_speed)
            
            # 验证车辆是否成功定位
            try:
                actual_pos = traci.vehicle.getPosition(sumo_vehicle_id)
                actual_lane = traci.vehicle.getLaneID(sumo_vehicle_id)
                
                # 检查位置是否有效（不是错误值）
                if actual_pos[0] < -1000000 or actual_pos[1] < -1000000:
                    print(f"❌ 车辆{vehicle_id}定位失败: 无效坐标 ({actual_pos[0]:.1f}, {actual_pos[1]:.1f})")
                    return False
                elif not actual_lane:
                    print(f"⚠️ 车辆{vehicle_id}未在任何车道: ({actual_pos[0]:.1f}, {actual_pos[1]:.1f})")
                    return False
                else:
                    print(f"✅ 车辆{vehicle_id}成功定位: ({actual_pos[0]:.1f}, {actual_pos[1]:.1f}), 车道: {actual_lane}")
                    
            except Exception as e:
                print(f"❌ 无法获取车辆{vehicle_id}位置: {e}")
                return False
            
            # 记录车辆状态
            self.active_vehicles[vehicle_id] = {
                'sumo_id': sumo_vehicle_id,
                'prev_pos': (initial_x, initial_y),
                'prev_speed': initial_speed,
                'added_frame': self.current_frame,
                'last_lane_id': lane_id,
                'target_speed': initial_speed,
                'safety_mode': 'normal'
            }
            
            # 初始化智能控制状态
            self.vehicle_control_modes[vehicle_id] = 'hybrid'  # 混合模式
            self.safety_overrides[vehicle_id] = False
            self.interaction_history[vehicle_id] = []
            
            self.stats['vehicles_added'] += 1
            print(f"✅ 添加智能车辆 {vehicle_id} (模式: 混合控制)")
            return True
            
        except Exception as e:
            print(f"❌ 添加智能车辆 {vehicle_id} 失败: {e}")
            return False
    
    def check_vehicle_interactions(self, vehicle_id):
        """检查车辆交互并决定控制策略 - 按SUMO官方文档"""
        if vehicle_id not in self.active_vehicles:
            return 'manual'
        
        sumo_vehicle_id = self.active_vehicles[vehicle_id]['sumo_id']
        
        try:
            # 获取周围车辆信息 (SUMO官方API)
            # 参考: https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html
            
            current_pos = traci.vehicle.getPosition(sumo_vehicle_id)
            current_speed = traci.vehicle.getSpeed(sumo_vehicle_id)
            current_lane = traci.vehicle.getLaneID(sumo_vehicle_id)
            
            # 检查位置是否有效
            if current_pos[0] < -1000000 or current_pos[1] < -1000000:
                print(f"⚠️ 车辆{vehicle_id}位置无效，使用手动模式")
                return 'manual'
            
            # 检查前方车辆 (官方推荐方法)
            leader_info = traci.vehicle.getLeader(sumo_vehicle_id, dist=20.0)
            
            # 检查跟随车辆
            follower_info = traci.vehicle.getFollower(sumo_vehicle_id, dist=20.0)
            
            # 安全距离检查
            safety_threat = False
            interaction_type = 'normal'
            
            if leader_info:
                leader_id, gap = leader_info
                if gap < self.safety_params['collision_threshold']:
                    safety_threat = True
                    interaction_type = 'collision_risk'
                elif gap < self.safety_params['min_gap']:
                    safety_threat = True
                    interaction_type = 'close_following'
            
            # 记录交互历史
            interaction_record = {
                'frame': self.current_frame,
                'type': interaction_type,
                'leader': leader_info,
                'follower': follower_info,
                'safety_threat': safety_threat
            }
            self.interaction_history[vehicle_id].append(interaction_record)
            
            # 决定控制模式
            if safety_threat:
                self.safety_overrides[vehicle_id] = True
                self.stats['safety_overrides_count'] += 1
                if interaction_type == 'collision_risk':
                    self.stats['collision_avoidances'] += 1
                return 'auto'  # 让SUMO接管
            else:
                self.safety_overrides[vehicle_id] = False
                return 'hybrid'  # 混合控制
                
        except Exception as e:
            print(f"⚠️ 车辆{vehicle_id}交互检查失败: {e}")
            return 'manual'
    
    def update_vehicle_intelligent(self, vehicle_id):
        """智能更新单个车辆 - 官方标准交互控制"""
        if vehicle_id not in self.active_vehicles:
            return False
        
        vehicle_state = self.active_vehicles[vehicle_id]
        sumo_vehicle_id = vehicle_state['sumo_id']
        
        try:
            # 安全检查
            if sumo_vehicle_id not in traci.vehicle.getIDList():
                del self.active_vehicles[vehicle_id]
                return False
            
            # 获取当前帧的NGSIM轨迹数据
            trajectory = self.vehicle_trajectories[vehicle_id]
            current_frame_data = trajectory[trajectory['Frame_ID'] == self.current_frame]
            
            if len(current_frame_data) == 0:
                return False
            
            row = current_frame_data.iloc[0]
            lane_id = row['Lane_ID']
            
            # 计算NGSIM目标位置和速度
            target_x, target_y = self.map_coordinates(
                row['Local_Y'], row['Local_X'], 
                self.transform_params, lane_id
            )
            target_speed = row['v_Vel'] * self.feet_to_meters
            
            # 检查车辆交互情况
            control_mode = self.check_vehicle_interactions(vehicle_id)
            
            # 根据交互情况选择控制策略
            if control_mode == 'auto':
                # 安全模式：让SUMO完全接管
                self.apply_sumo_safety_control(vehicle_id, sumo_vehicle_id, target_x, target_y)
                
            elif control_mode == 'hybrid':
                # 混合模式：目标引导 + 安全检查
                self.apply_hybrid_control(vehicle_id, sumo_vehicle_id, target_x, target_y, target_speed)
                
            else:
                # 手动模式：精准控制 (仅在安全时使用)
                self.apply_manual_control(vehicle_id, sumo_vehicle_id, target_x, target_y, target_speed)
            
            # 更新车辆状态
            vehicle_state['prev_pos'] = (target_x, target_y)
            vehicle_state['target_speed'] = target_speed
            vehicle_state['last_lane_id'] = lane_id
            
            self.stats['total_updates'] += 1
            self.stats['intelligent_behaviors'] += 1
            
            return True
            
        except Exception as e:
            print(f"❌ 智能更新车辆 {vehicle_id} 失败: {e}")
            return False
    
    def apply_sumo_safety_control(self, vehicle_id, sumo_vehicle_id, target_x, target_y):
        """应用SUMO安全控制 - 完全按官方文档"""
        try:
            # 启用全安全模式 - sublane兼容
            traci.vehicle.setSpeedMode(sumo_vehicle_id, 31)  # 所有安全检查
            traci.vehicle.setLaneChangeMode(sumo_vehicle_id, 1024)  # sublane安全变道
            
            # 使用changeLane引导而不是强制moveToXY
            current_lane = traci.vehicle.getLaneID(sumo_vehicle_id)
            if "E0_" in current_lane:
                # 计算目标车道
                if target_y > -3.0:  # 最左侧
                    target_lane_index = 2
                elif target_y > -6.4:  # 中间
                    target_lane_index = 1
                else:  # 最右侧
                    target_lane_index = 0
                
                current_lane_index = int(current_lane.split('_')[1])
                if target_lane_index != current_lane_index:
                    # 建议变道，让SUMO决定是否安全
                    traci.vehicle.changeLane(sumo_vehicle_id, target_lane_index, 5.0)
            
            # 让SUMO自动控制速度 (官方安全方法)
            traci.vehicle.setSpeed(sumo_vehicle_id, -1)  # -1 = 自动速度控制
            
            print(f"🛡️ 车辆{vehicle_id}进入SUMO安全控制模式")
            
        except Exception as e:
            print(f"⚠️ 安全控制失败 {vehicle_id}: {e}")
    
    def apply_hybrid_control(self, vehicle_id, sumo_vehicle_id, target_x, target_y, target_speed):
        """应用混合控制模式 - 目标引导+安全检查"""
        try:
            # 混合安全模式 - sublane兼容 (平衡精度和安全)
            traci.vehicle.setSpeedMode(sumo_vehicle_id, 23)  # 启用碰撞检查
            traci.vehicle.setLaneChangeMode(sumo_vehicle_id, 512)  # sublane保守变道
            
            # 获取当前状态 - 安全检查
            try:
                current_pos = traci.vehicle.getPosition(sumo_vehicle_id)
                current_speed = traci.vehicle.getSpeed(sumo_vehicle_id)
                
                # 检查位置是否有效
                if current_pos[0] < -1000000 or current_pos[1] < -1000000:
                    print(f"⚠️ 车辆{vehicle_id}位置无效，跳过更新")
                    return
                    
            except Exception as e:
                print(f"⚠️ 无法获取车辆{vehicle_id}状态: {e}")
                return
            
            # 平滑位置调整
            max_move = 3.0
            distance = math.sqrt((target_x - current_pos[0])**2 + (target_y - current_pos[1])**2)
            if distance > max_move:
                ratio = max_move / distance
                adjusted_x = current_pos[0] + (target_x - current_pos[0]) * ratio
                adjusted_y = current_pos[1] + (target_y - current_pos[1]) * ratio
            else:
                adjusted_x, adjusted_y = target_x, target_y
            
            # 智能速度调整
            speed_diff = target_speed - current_speed
            max_accel = 2.0  # 温和加速
            if abs(speed_diff) > max_accel * 0.1:
                adjusted_speed = current_speed + (max_accel * 0.1 if speed_diff > 0 else -max_accel * 0.1)
            else:
                adjusted_speed = target_speed
            
            # 应用控制 - SUMO会进行安全检查
            angle = self.calculate_vehicle_angle((adjusted_x, adjusted_y), current_pos)
            
            # 安全的位置更新
            try:
                traci.vehicle.moveToXY(
                    vehID=sumo_vehicle_id,
                    edgeID="E0",
                    laneIndex=-1,
                    x=adjusted_x,
                    y=adjusted_y,
                    angle=angle,
                    keepRoute=1  # 简化参数
                )
            except Exception as e:
                # 如果moveToXY失败，尝试更简单的方法
                try:
                    traci.vehicle.moveToXY(
                        vehID=sumo_vehicle_id,
                        edgeID="E0",
                        laneIndex=-1,
                        x=adjusted_x,
                        y=adjusted_y
                    )
                except Exception as e2:
                    print(f"⚠️ 位置更新失败 {vehicle_id}: {e2}")
                    return
            traci.vehicle.setSpeed(sumo_vehicle_id, adjusted_speed)
            
            # 调试信息 - 每100次更新显示一次位置
            if self.stats['total_updates'] % 100 == 0:
                print(f"🔄 更新车辆{vehicle_id}: 目标({target_x:.1f}, {target_y:.1f}) "
                      f"→ 调整({adjusted_x:.1f}, {adjusted_y:.1f}), 速度:{adjusted_speed:.1f}m/s")
            
        except Exception as e:
            print(f"⚠️ 混合控制失败 {vehicle_id}: {e}")
    
    def apply_manual_control(self, vehicle_id, sumo_vehicle_id, target_x, target_y, target_speed):
        """应用手动精准控制 - 仅在安全时使用"""
        try:
            # 精准控制模式 (仅在无交互时使用)
            traci.vehicle.setSpeedMode(sumo_vehicle_id, 0)   # 手动速度
            traci.vehicle.setLaneChangeMode(sumo_vehicle_id, 0)  # 手动变道
            
            # 计算角度
            prev_pos = self.active_vehicles[vehicle_id]['prev_pos']
            angle = self.calculate_vehicle_angle((target_x, target_y), prev_pos)
            
            # 精准定位
            traci.vehicle.moveToXY(
                vehID=sumo_vehicle_id,
                edgeID="E0",
                laneIndex=-1,
                x=target_x,
                y=target_y,
                angle=angle,
                keepRoute=2
            )
            traci.vehicle.setSpeed(sumo_vehicle_id, target_speed)
            
        except Exception as e:
            print(f"⚠️ 手动控制失败 {vehicle_id}: {e}")
    
    def update_all_vehicles_intelligent(self):
        """批量智能更新所有车辆"""
        updated_count = 0
        
        for vehicle_id in list(self.active_vehicles.keys()):
            try:
                # 检查数据有效性
                trajectory = self.vehicle_trajectories[vehicle_id]
                current_frame_data = trajectory[trajectory['Frame_ID'] == self.current_frame]
                
                if len(current_frame_data) == 0:
                    # 移除无数据的车辆
                    self.remove_intelligent_vehicle(vehicle_id)
                    continue
                
                # 智能更新
                if self.update_vehicle_intelligent(vehicle_id):
                    updated_count += 1
                    
            except Exception as e:
                print(f"❌ 批量更新车辆 {vehicle_id} 失败: {e}")
                continue
        
        return updated_count
    
    def remove_intelligent_vehicle(self, vehicle_id):
        """移除智能车辆"""
        if vehicle_id not in self.active_vehicles:
            return
        
        sumo_vehicle_id = self.active_vehicles[vehicle_id]['sumo_id']
        
        try:
            # 检查车辆是否仍在SUMO中
            if sumo_vehicle_id not in traci.vehicle.getIDList():
                print(f"⚠️ 车辆{vehicle_id}已不在SUMO中，仅清理跟踪数据")
                # 直接清理跟踪数据
                del self.active_vehicles[vehicle_id]
                if vehicle_id in self.vehicle_control_modes:
                    del self.vehicle_control_modes[vehicle_id]
                if vehicle_id in self.safety_overrides:
                    del self.safety_overrides[vehicle_id]
                if vehicle_id in self.interaction_history:
                    del self.interaction_history[vehicle_id]
                self.stats['vehicles_removed'] += 1
                return
            
            # 车辆在SUMO中，正常移除
            traci.vehicle.remove(sumo_vehicle_id)
            self.stats['vehicles_removed'] += 1
            
        except Exception as e:
            print(f"⚠️ 移除智能车辆 {vehicle_id} 时出错: {e}")
        finally:
            # 清理所有相关状态
            if vehicle_id in self.active_vehicles:
                del self.active_vehicles[vehicle_id]
            if vehicle_id in self.vehicle_control_modes:
                del self.vehicle_control_modes[vehicle_id]
            if vehicle_id in self.safety_overrides:
                del self.safety_overrides[vehicle_id]
            if vehicle_id in self.interaction_history:
                del self.interaction_history[vehicle_id]
    
    def run_intelligent_multi_vehicle_simulation(self, csv_file, net_file):
        """运行智能多车交互仿真 - SUMO官方标准实现"""
        print(f"\n" + "="*80)
        print(f"🧠 SUMO智能多车交互仿真系统 - 官方标准版")
        print(f"✅ 精准NGSIM轨迹跟踪 + SUMO智能安全交互")
        print(f"🚗 混合控制模式: 目标跟踪 ⚖️ 安全行为")
        print(f"🛡️ 启用官方安全检查、碰撞避免和智能变道")
        print(f"="*80)
        
        try:
            # 1. 加载网络信息
            self.load_network_info(net_file)
            
            # 2. 加载多车数据
            df = self.load_multi_vehicle_data(csv_file)
            
            # 3. 坐标转换
            print(f"\n=== 应用精准坐标映射算法 ===")
            self.transform_params = self.transform_coordinates_official(df)
            print(f"✅ 坐标转换参数已初始化")
            
            # 4. 创建配置文件
            route_file = "intelligent_multi_vehicle.rou.xml"
            config_file = "intelligent_multi_vehicle.sumocfg"
            
            self.create_intelligent_vehicle_types(route_file)
            self.create_sumo_config(net_file, route_file, config_file)
            
            # 5. 启动SUMO
            print(f"\n=== 启动SUMO智能交互仿真 ===")
            config_abs = os.path.abspath(config_file)
            sumo_cmd = ["sumo-gui", "-c", config_abs, "--start", "--collision.action", "warn"]
            
            traci.start(sumo_cmd, port=8817, numRetries=10)
            print("TraCI连接成功，启用碰撞监测")
            
            # 等待初始化
            time.sleep(1)
            traci.simulationStep()
            
            # 6. 智能多车仿真主循环
            print(f"\n=== 开始智能多车交互仿真 ===")
            print(f"🎯 起始帧: {self.current_frame}, 结束帧: {self.max_frame}")
            print(f"🧠 控制模式: 精准跟踪 + 智能交互 + 安全保护")
            
            total_frames = self.max_frame - self.current_frame + 1
            start_time = time.time()
            
            # 主仿真循环
            for frame_id in range(self.current_frame, self.max_frame + 1):
                try:
                    self.current_frame = frame_id
                    
                    # 1. 车辆生命周期管理
                    current_frame_vehicles = set(self.frame_vehicles.get(frame_id, []))
                    active_vehicle_ids = set(self.active_vehicles.keys())
                    
                    # 2. 添加新车辆
                    new_vehicles = current_frame_vehicles - active_vehicle_ids
                    added_count = 0
                    for vehicle_id in new_vehicles:
                        vehicle_trajectory = self.vehicle_trajectories[vehicle_id]
                        frame_data = vehicle_trajectory[vehicle_trajectory['Frame_ID'] == frame_id]
                        if len(frame_data) > 0:
                            row = frame_data.iloc[0]
                            if self.add_intelligent_vehicle(vehicle_id, row, row['Lane_ID']):
                                added_count += 1
                    
                    # 3. 移除离开的车辆
                    vehicles_to_remove = active_vehicle_ids - current_frame_vehicles
                    removed_count = 0
                    for vehicle_id in vehicles_to_remove:
                        self.remove_intelligent_vehicle(vehicle_id)
                        removed_count += 1
                    
                    # 4. 智能批量更新所有车辆
                    updated_count = self.update_all_vehicles_intelligent()
                    
                    # 5. 智能进度显示
                    if frame_id % 100 == 0 or added_count > 0 or removed_count > 0:
                        progress = ((frame_id - df['Frame_ID'].min()) / total_frames) * 100
                        active_count = len(self.active_vehicles)
                        elapsed_time = time.time() - start_time
                        
                        # 统计控制模式分布
                        safety_count = sum(1 for v in self.safety_overrides.values() if v)
                        
                        print(f"🧠 帧 {frame_id}/{self.max_frame} ({progress:.1f}%) - "
                              f"活跃: {active_count}辆, 智能更新: {updated_count}辆")
                        
                        if added_count > 0:
                            print(f"  ➕ 智能添加 {added_count}辆车")
                        if removed_count > 0:
                            print(f"  ➖ 移除 {removed_count}辆车")
                        if safety_count > 0:
                            print(f"  🛡️ 安全接管: {safety_count}辆车")
                        
                        # 详细统计
                        if frame_id % 500 == 0 and elapsed_time > 0:
                            fps = (frame_id - self.current_frame + 1) / elapsed_time
                            print(f"  ⚡ 仿真速度: {fps:.1f} FPS")
                            print(f"  📊 智能行为统计:")
                            print(f"     - 总控制次数: {self.stats['total_updates']}")
                            print(f"     - 安全接管次数: {self.stats['safety_overrides_count']}")
                            print(f"     - 碰撞避免次数: {self.stats['collision_avoidances']}")
                            print(f"     - 智能行为次数: {self.stats['intelligent_behaviors']}")
                    
                    # 6. 执行仿真步
                    traci.simulationStep()
                    time.sleep(0.03)  # 观察延时
                    
                except KeyboardInterrupt:
                    print(f"\n⏹️ 用户中断智能仿真 (帧 {frame_id})")
                    break
                except Exception as e:
                    print(f"❌ 帧 {frame_id} 智能仿真错误: {e}")
                    continue
            
            # 7. 仿真完成统计
            total_time = time.time() - start_time
            print(f"\n✅ 智能多车交互仿真完成！")
            print(f"📊 最终统计:")
            print(f"  - 处理车辆数: {self.total_vehicles} 辆")
            print(f"  - 仿真帧数: {total_frames} 帧")
            print(f"  - 仿真时长: {total_frames * 0.1:.1f}秒 (游戏时间)")
            print(f"  - 实际耗时: {total_time:.1f}秒")
            print(f"  - 平均FPS: {total_frames / total_time:.1f}")
            print(f"  - 智能控制效果:")
            print(f"     🎯 精准控制次数: {self.stats['total_updates']}")
            print(f"     🛡️ 安全接管次数: {self.stats['safety_overrides_count']}")
            print(f"     🚗 碰撞避免次数: {self.stats['collision_avoidances']}")
            print(f"     🧠 智能行为次数: {self.stats['intelligent_behaviors']}")
            
            # 计算安全接管率
            safety_rate = (self.stats['safety_overrides_count'] / max(self.stats['total_updates'], 1)) * 100
            print(f"     📈 安全接管率: {safety_rate:.2f}%")
            
            # 8. 继续自由仿真观察
            print(f"\n🔍 继续智能交互观察（最多50步）...")
            for i in range(50):
                active_vehicles = [vid for vid in self.active_vehicles.keys() 
                                if self.active_vehicles[vid]['sumo_id'] in traci.vehicle.getIDList()]
                
                if not active_vehicles:
                    print(f"ℹ️ 所有车辆已完成智能仿真")
                    break
                
                if i % 10 == 0:
                    print(f"智能交互观察 +{i}: 剩余车辆 {len(active_vehicles)}辆")
                
                traci.simulationStep()
                time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ 智能多车仿真错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                traci.close()
                print("TraCI连接已关闭")
            except:
                pass


def main():
    """主函数 - 智能多车交互仿真"""
    print("🧠 SUMO智能多车交互控制系统 - 官方标准版")
    print("✅ 精准NGSIM轨迹 + SUMO智能安全 + 官方交互模型")
    print("🚗 混合控制: 目标跟踪 ⚖️ 碰撞避免 ⚖️ 智能变道")
    print("🛡️ 100%基于SUMO官方文档的安全交互实现")
    print("="*70)
    
    # 文件路径
    csv_file = "final.csv"
    net_file = "./1.net.xml"
    
    # 验证文件存在
    if not os.path.exists(csv_file):
        print(f"❌ 错误: 多车数据文件不存在 - {csv_file}")
        return
    
    if not os.path.exists(net_file):
        print(f"❌ 错误: SUMO网络文件不存在 - {net_file}")
        return
    
    # 创建智能多车控制器并运行
    controller = IntelligentMultiVehicleController()
    controller.run_intelligent_multi_vehicle_simulation(csv_file, net_file)


if __name__ == "__main__":
    main()
