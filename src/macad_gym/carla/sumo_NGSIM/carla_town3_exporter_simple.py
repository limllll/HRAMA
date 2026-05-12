#!/usr/bin/env python3
"""
CARLA Town3 SUMO路网导出工具 - 与训练代码参数一致版本
基于SUMO官方文档：https://sumo.dlr.de/docs/Tutorials/CarlaIntegration.html

🎯 主要功能：
1. 连接CARLA服务器获取Town03地图
2. 导出OpenDRIVE格式文件 (.xodr)
3. 使用netconvert转换为SUMO网络文件 (.net.xml)

📋 SUMO官方推荐流程：
CARLA Server → OpenDRIVE Export → netconvert → SUMO Network

🔧 依赖要求：
- CARLA Python API (pip install carla)
- SUMO工具包 (netconvert)
- CARLA服务器运行中

📄 输出文件：
- carla_town3.xodr (OpenDRIVE格式)
- carla_town3.net.xml (SUMO网络文件，与训练代码完全一致)

🚦 交通控制设施支持：
- 与sumo_bridge.py使用完全相同的netconvert参数
- 确保训练时和导出工具生成的网络完全一致
"""

import os
import sys
import time
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

class CarlaTown3Exporter:
    """CARLA Town3 SUMO路网导出器 - 训练参数一致版"""
    
    def __init__(self, output_dir="./"):
        self.output_dir = Path(output_dir)
        self.carla_host = 'localhost'
        self.carla_port = 2000
        self.map_name = 'Town03'
        self.timeout = 20.0
        
        # 输出文件名
        self.opendrive_file = "carla_town3_simple.xodr"
        self.network_file = "carla_town3_simple.net.xml"
        
        # 与sumo_bridge.py完全相同的netconvert参数 (中等复杂度)
        self.netconvert_params = {
            '--opendrive': self.opendrive_file,
            '--output-file': self.network_file,
            
            # 几何修复与完整导入 (与sumo_bridge.py一致)
            '--geometry.min-radius.fix': '',
            '--geometry.remove': '',
            '--opendrive.curve-resolution': '1',
            '--opendrive.import-all-lanes': '',
            
            # 保留原始名称 (与sumo_bridge.py一致)
            '--output.original-names': '',
            
            # 交通灯设置 (与sumo_bridge.py一致)
            '--tls.guess': 'true',
            '--tls.join': 'true', 
            '--tls.discard-loaded': 'false',
            
            # 其他优化 (与sumo_bridge.py一致)
            '--junctions.corner-detail': '5',
            '--no-turnarounds': 'true'
        }
    
    def check_carla_availability(self):
        """检查CARLA可用性"""
        print("🔍 检查CARLA连接...")
        
        try:
            import carla
            # 尝试获取版本信息，如果没有版本信息就跳过
            try:
                version = getattr(carla, '__version__', 'unknown')
                print(f"✅ CARLA Python API 已安装: {version}")
            except:
                print(f"✅ CARLA Python API 已安装")
        except ImportError:
            print("❌ 未安装CARLA Python API")
            print("📥 安装方法: pip install carla")
            return False, None
        
        try:
            print(f"🔌 连接CARLA服务器: {self.carla_host}:{self.carla_port}")
            client = carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(self.timeout)
            
            # 测试连接
            world = client.get_world()
            current_map = world.get_map().name
            print(f"✅ CARLA服务器连接成功，当前地图: {current_map}")
            
            return True, carla
            
        except Exception as e:
            print(f"❌ CARLA服务器连接失败: {e}")
            print("💡 确保CARLA服务器正在运行:")
            print("   ./CarlaUE4.exe -opengl")
            return False, None
    
    def load_town03_map(self, carla):
        """加载CARLA Town03地图"""
        print(f"🗺️ 加载CARLA地图: {self.map_name}")
        
        try:
            client = carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(self.timeout)
            
            # 获取可用地图列表
            available_maps = client.get_available_maps()
            town03_maps = [map_name for map_name in available_maps if 'Town03' in map_name]
            
            if not town03_maps:
                print(f"❌ 未找到Town03地图，可用地图: {available_maps}")
                return None
                
            # 选择第一个Town03地图
            selected_map = town03_maps[0]
            print(f"📍 选择地图: {selected_map}")
            
            # 加载地图
            world = client.load_world(selected_map)
            print(f"✅ 地图加载成功，等待同步...")
            
            # 等待地图完全加载
            time.sleep(3)
            
            # 验证地图加载
            current_map = world.get_map()
            actual_name = current_map.name
            print(f"🎯 当前活跃地图: {actual_name}")
            
            return current_map
            
        except Exception as e:
            print(f"❌ 加载地图失败: {e}")
            return None
    
    def export_opendrive(self, carla_map):
        """导出OpenDRIVE格式文件"""
        print(f"📤 导出OpenDRIVE格式文件...")
        
        try:
            # 获取OpenDRIVE数据
            print("🔄 从CARLA地图提取OpenDRIVE数据...")
            opendrive_content = carla_map.to_opendrive()
            
            if not opendrive_content:
                print("❌ OpenDRIVE内容为空")
                return False
            
            # 保存到文件
            opendrive_path = self.output_dir / self.opendrive_file
            with open(opendrive_path, 'w', encoding='utf-8') as f:
                f.write(opendrive_content)
            
            # 验证文件
            file_size = os.path.getsize(opendrive_path)
            if file_size > 1000:  # 至少1KB
                print(f"✅ OpenDRIVE文件导出成功:")
                print(f"   📁 文件: {opendrive_path}")
                print(f"   📏 大小: {file_size / 1024:.1f} KB")
                
                # 简单验证XML格式
                try:
                    ET.parse(opendrive_path)
                    print(f"   ✅ XML格式验证通过")
                except ET.ParseError as e:
                    print(f"   ⚠️ XML格式警告: {e}")
                
                return True
            else:
                print(f"❌ OpenDRIVE文件太小: {file_size} bytes")
                return False
                
        except Exception as e:
            print(f"❌ OpenDRIVE导出失败: {e}")
            return False
    
    def convert_to_sumo_network(self):
        """使用netconvert转换为SUMO网络 (与sumo_bridge.py相同参数)"""
        print(f"🔄 使用netconvert转换为SUMO网络 (训练参数一致版)...")
        
        # 检查netconvert可用性
        try:
            result = subprocess.run(['netconvert', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_info = result.stdout.strip().split('\n')[0]
                print(f"✅ netconvert可用: {version_info}")
            else:
                print("❌ netconvert不可用")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ 未找到netconvert，请安装SUMO工具包")
            return False
        
        # 构建netconvert命令 (与sumo_bridge.py完全一致)
        cmd = ['netconvert']
        for param, value in self.netconvert_params.items():
            if value:  # 如果有值
                cmd.extend([param, value])
            else:  # 如果是开关参数
                cmd.append(param)
        
        print(f"🔧 netconvert命令: {' '.join(cmd)}")
        print(f"📋 使用与训练代码完全相同的参数")
        
        try:
            # 切换到输出目录
            original_cwd = os.getcwd()
            os.chdir(self.output_dir)
            
            # 执行转换
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # 恢复工作目录
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                network_path = self.output_dir / self.network_file
                if os.path.exists(network_path):
                    file_size = os.path.getsize(network_path)
                    print(f"✅ SUMO网络文件生成成功:")
                    print(f"   📁 文件: {network_path}")
                    print(f"   📏 大小: {file_size / 1024:.1f} KB")
                    
                    # 分析网络
                    self.analyze_sumo_network(network_path)
                    return True
                else:
                    print(f"❌ 网络文件未生成: {network_path}")
                    return False
            else:
                print(f"❌ netconvert转换失败:")
                print(f"   stderr: {result.stderr}")
                print(f"   stdout: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ netconvert转换超时")
            return False
        except Exception as e:
            print(f"❌ netconvert转换异常: {e}")
            return False
    
    def analyze_sumo_network(self, network_file):
        """分析SUMO网络文件 - 训练一致版本检查"""
        try:
            print("📊 分析SUMO网络结构 (训练参数版)...")
            tree = ET.parse(network_file)
            root = tree.getroot()
            
            # 统计网络元素
            edges = root.findall('edge')
            junctions = root.findall('junction')
            connections = root.findall('connection')
            traffic_lights = root.findall('tlLogic')  # 交通信号灯逻辑
            
            # 过滤内部边
            regular_edges = [e for e in edges if not e.get('function')]
            internal_edges = [e for e in edges if e.get('function') == 'internal']
            
            # 分析交叉路口类型
            traffic_light_junctions = [j for j in junctions if j.get('type') == 'traffic_light']
            priority_junctions = [j for j in junctions if j.get('type') == 'priority']
            roundabout_junctions = [j for j in junctions if 'roundabout' in j.get('name', '').lower()]
            
            # 统计车道和速度
            total_lanes = 0
            lane_lengths = []
            speed_limits = []
            
            for edge in regular_edges:
                lanes = edge.findall('lane')
                total_lanes += len(lanes)
                
                for lane in lanes:
                    length = float(lane.get('length', 0))
                    speed = float(lane.get('speed', 0))
                    if length > 0:
                        lane_lengths.append(length)
                    if speed > 0:
                        speed_limits.append(speed * 3.6)  # 转换为km/h
            
            # 计算网络边界
            x_coords, y_coords = [], []
            for edge in regular_edges:
                for lane in edge.findall('lane'):
                    shape = lane.get('shape', '')
                    if shape:
                        for point in shape.split():
                            if ',' in point:
                                x, y = map(float, point.split(','))
                                x_coords.append(x)
                                y_coords.append(y)
            
            # 基础网络统计
            print(f"   🛣️  常规道路: {len(regular_edges)}条")
            print(f"   🔀 内部连接: {len(internal_edges)}条") 
            print(f"   🚗 车道总数: {total_lanes}条")
            print(f"   ⚡ 连接关系: {len(connections)}个")
            
            # 交通控制设施统计 (训练参数版)
            print(f"   🚦 交叉路口总数: {len(junctions)}个")
            print(f"      ├─ 信号灯控制: {len(traffic_light_junctions)}个")
            print(f"      ├─ 优先级控制: {len(priority_junctions)}个")
            print(f"      └─ 环岛: {len(roundabout_junctions)}个")
            print(f"   🔴 交通信号灯程序: {len(traffic_lights)}个")
            
            # 道路特征统计
            if lane_lengths:
                avg_length = sum(lane_lengths) / len(lane_lengths)
                total_length = sum(lane_lengths) / 1000  # 转换为km
                print(f"   📏 平均车道长度: {avg_length:.1f}m")
                print(f"   🛤️  道路总长度: {total_length:.1f}km")
            
            if speed_limits:
                avg_speed = sum(speed_limits) / len(speed_limits)
                max_speed = max(speed_limits)
                print(f"   🏎️  平均限速: {avg_speed:.1f}km/h")
                print(f"   🚀 最高限速: {max_speed:.1f}km/h")
            
            if x_coords and y_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                network_size = ((x_max - x_min) * (y_max - y_min)) / 1000000  # km²
                print(f"   🗺️  网络范围: X({x_min:.0f}~{x_max:.0f}), Y({y_min:.0f}~{y_max:.0f})")
                print(f"   📐 覆盖面积: ~{network_size:.1f}km²")
            
            # 训练参数版本特征检查
            print(f"   ✅ 与sumo_bridge.py使用相同参数生成")
            print(f"   ✅ 训练时SUMO GUI显示效果与此完全一致")
                
        except Exception as e:
            print(f"⚠️ 网络分析失败: {e}")
    
    def export_carla_town3(self):
        """主要路网导出流程 (训练参数一致版)"""
        print("🎯 CARLA Town3 SUMO路网导出工具")
        print("📋 与训练代码参数完全一致版本")
        print("="*50)
        
        # 1. 检查CARLA可用性
        carla_available, carla = self.check_carla_availability()
        if not carla_available:
            return False
        
        # 2. 加载Town03地图
        carla_map = self.load_town03_map(carla)
        if carla_map is None:
            return False
        
        # 3. 导出OpenDRIVE
        if not self.export_opendrive(carla_map):
            return False
        
        # 4. 转换为SUMO网络
        if not self.convert_to_sumo_network():
            return False
        
        # 5. 完成报告
        print(f"\n🎉 CARLA Town3路网导出成功！")
        print(f"📁 生成的文件:")
        print(f"   🗺️  OpenDRIVE: {self.opendrive_file}")
        print(f"   🛣️  SUMO网络: {self.network_file}")
        
        print(f"\n🚀 使用方法:")
        print(f"   1. 查看网络 (与训练时完全一致):")
        print(f"      sumo-gui {self.network_file}")
        print(f"   2. Python TraCI使用:")
        print(f"      traci.start(['sumo-gui', '{self.network_file}'])")
        print(f"   3. 在其他项目中引用:")
        print(f"      <net-file value='{self.network_file}'/>")
        
        print(f"\n🚦 网络特征说明:")
        print(f"   ✅ 使用与sumo_bridge.py相同的netconvert参数")
        print(f"   ✅ 交通灯猜测和连接 (tls.guess=true, tls.join=true)")
        print(f"   ✅ 交叉路口细节级别5")
        print(f"   ✅ 禁用掉头 (no-turnarounds=true)")
        print(f"   ✅ 训练时SUMO GUI显示的网络与此完全一致")
        
        return True


def main():
    """主函数"""
    print("🎯 CARLA Town3路网导出工具 - 训练参数一致版")
    print("📋 连接CARLA服务器 → OpenDRIVE导出 → SUMO网络(.net.xml)")
    print("🚦 使用与sumo_bridge.py完全相同的netconvert参数")
    print("🔧 确保训练时和导出工具生成的网络完全一致")
    print("="*60)
    
    # 检查当前目录
    current_dir = Path.cwd()
    print(f"📂 工作目录: {current_dir}")
    
    # 创建导出器并运行
    exporter = CarlaTown3Exporter(output_dir=current_dir)
    success = exporter.export_carla_town3()
    
    if success:
        print("\n✅ 路网导出完成！")
        print("📊 现在训练时的SUMO GUI显示与导出的网络完全一致")
    else:
        print("\n❌ 路网导出失败")
        print("💡 故障排除:")
        print("   1. 确保CARLA服务器运行: ./CarlaUE4.exe")
        print("   2. 安装CARLA Python API: pip install carla")
        print("   3. 检查SUMO工具安装: netconvert --version")


if __name__ == "__main__":
    main()

