#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🔍 SUMO车辆可见性测试工具

用于诊断CARLA控制的车辆在SUMO中是否被其他车辆检测到

使用方法:
1. 启动训练程序
2. 在另一个终端运行此脚本
3. 观察输出，检查CARLA车辆是否在getIDList()中，以及是否被其他车辆检测
"""

import os
import sys
import time

def setup_sumo():
    """设置SUMO环境"""
    # 尝试找到SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        candidates = [
            "/home/lim/sumo-1.22.0",
            "/usr/share/sumo",
            "/usr/local/share/sumo",
            "C:\\Program Files (x86)\\Eclipse\\Sumo",
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                sumo_home = cand
                os.environ['SUMO_HOME'] = sumo_home
                break
    
    if not sumo_home:
        print("❌ 找不到SUMO_HOME，请设置环境变量")
        sys.exit(1)
    
    tools_dir = os.path.join(sumo_home, 'tools')
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)
    
    print(f"✅ SUMO_HOME: {sumo_home}")
    return sumo_home

def test_visibility():
    """测试车辆可见性"""
    setup_sumo()
    
    import traci
    
    # 连接到正在运行的SUMO实例
    try:
        # 尝试连接到默认端口
        traci.init(8813)
        print("✅ 成功连接到SUMO (端口 8813)")
    except Exception as e:
        print(f"❌ 无法连接到SUMO: {e}")
        print("   请确保训练程序正在运行")
        return
    
    print("\n" + "="*80)
    print("🔍 开始车辆可见性测试")
    print("="*80)
    
    try:
        # 获取所有车辆
        all_vehicles = traci.vehicle.getIDList()
        print(f"\n📊 SUMO中总共有 {len(all_vehicles)} 辆车")
        
        # 分类车辆
        carla_vehicles = [v for v in all_vehicles if v.startswith('carla_')]
        ngsim_vehicles = [v for v in all_vehicles if v.startswith('ngsim_')]
        sumo_vehicles = [v for v in all_vehicles if not v.startswith('carla_') and not v.startswith('ngsim_')]
        
        print(f"   - CARLA控制车辆: {len(carla_vehicles)}")
        print(f"   - NGSIM背景车辆: {len(ngsim_vehicles)}")
        print(f"   - SUMO原生车辆: {len(sumo_vehicles)}")
        
        if not carla_vehicles:
            print("\n⚠️ 没有找到CARLA车辆！")
            return
        
        # 显示CARLA车辆信息
        print("\n" + "-"*80)
        print("🚗 CARLA车辆详情:")
        print("-"*80)
        for veh_id in carla_vehicles[:3]:  # 只显示前3个
            try:
                pos = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                lane = traci.vehicle.getLaneID(veh_id)
                speed_mode = traci.vehicle.getSpeedMode(veh_id)
                lc_mode = traci.vehicle.getLaneChangeMode(veh_id)
                
                print(f"\n车辆: {veh_id}")
                print(f"  位置: ({pos[0]:.2f}, {pos[1]:.2f})")
                print(f"  速度: {speed:.2f} m/s")
                print(f"  车道: {lane}")
                print(f"  SpeedMode: {speed_mode}")
                print(f"  LaneChangeMode: {lc_mode}")
                
            except Exception as e:
                print(f"  ❌ 无法获取信息: {e}")
        
        # 测试SUMO/NGSIM车辆能否看到CARLA车辆
        print("\n" + "-"*80)
        print("👁️ 可见性测试（其他车辆能否检测到CARLA车辆）:")
        print("-"*80)
        
        test_vehicles = (ngsim_vehicles + sumo_vehicles)[:10]  # 测试前10个
        detection_count = 0
        
        for veh_id in test_vehicles:
            try:
                pos = traci.vehicle.getPosition(veh_id)
                lane = traci.vehicle.getLaneID(veh_id)
                
                # 检测前方车辆
                leader_data = traci.vehicle.getLeader(veh_id, 100.0)
                
                veh_type = "NGSIM" if veh_id.startswith('ngsim_') else "SUMO"
                print(f"\n{veh_type}车辆: {veh_id}")
                print(f"  位置: ({pos[0]:.2f}, {pos[1]:.2f})")
                print(f"  车道: {lane}")
                
                if leader_data:
                    leader_id, distance = leader_data
                    is_carla = leader_id.startswith('carla_')
                    
                    if is_carla:
                        print(f"  ✅ 检测到前车: {leader_id} (CARLA车辆!) 距离: {distance:.2f}m")
                        detection_count += 1
                    else:
                        print(f"  🔵 检测到前车: {leader_id} (非CARLA) 距离: {distance:.2f}m")
                else:
                    # 计算与最近CARLA车辆的距离
                    min_dist = float('inf')
                    nearest_carla = None
                    
                    for carla_id in carla_vehicles:
                        carla_pos = traci.vehicle.getPosition(carla_id)
                        dist = ((pos[0] - carla_pos[0])**2 + (pos[1] - carla_pos[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            nearest_carla = carla_id
                    
                    if min_dist < 50:
                        print(f"  ⚠️ 未检测到前车，但附近有CARLA车辆!")
                        print(f"     最近: {nearest_carla}, 距离: {min_dist:.2f}m")
                    else:
                        print(f"  ⭕ 未检测到前车（附近无车辆）")
                        
            except Exception as e:
                print(f"  ❌ 检测失败: {e}")
        
        # 总结
        print("\n" + "="*80)
        print(f"📊 测试结果:")
        print(f"   - 测试了 {len(test_vehicles)} 个背景车辆")
        print(f"   - 其中 {detection_count} 个能检测到CARLA车辆")
        
        if detection_count == 0:
            print("\n⚠️ 警告：没有背景车辆能检测到CARLA车辆!")
            print("   可能原因:")
            print("   1. CARLA车辆的SpeedMode/LaneChangeMode设置不当")
            print("   2. CARLA车辆不在道路网络上")
            print("   3. 车辆距离太远")
        else:
            print(f"\n✅ 成功！{detection_count}个背景车辆能看到CARLA车辆")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    print("="*80)
    print("🔍 SUMO车辆可见性诊断工具")
    print("="*80)
    print("\n请确保:")
    print("1. 训练程序正在运行")
    print("2. SUMO已启动（GUI或headless模式）")
    print("3. 有CARLA控制的车辆在场景中")
    print("\n按Enter开始测试...")
    input()
    
    test_visibility()

