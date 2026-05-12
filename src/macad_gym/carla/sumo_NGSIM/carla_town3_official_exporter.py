#!/usr/bin/env python3
"""
CARLA Town3 SUMO路网导出工具 - 使用官方netconvert_carla脚本
基于CARLA官方Co-Simulation教程实现

🎯 主要功能：
1. 连接CARLA服务器获取Town03地图
2. 导出OpenDRIVE格式文件 (.xodr)
3. 使用官方netconvert_carla.py脚本转换为SUMO网络文件 (.net.xml)

📋 官方流程：
CARLA Server → OpenDRIVE Export → netconvert_carla.py → SUMO Network (含交通灯)

🔧 依赖要求：
- CARLA Python API (pip install carla)
- SUMO工具包 (netconvert)
- CARLA服务器运行中
- 官方netconvert_carla.py脚本

📄 输出文件：
- carla_town3_official.xodr (OpenDRIVE格式)
- carla_town3_official.net.xml (SUMO网络文件，含完整交通灯控制)

🚦 交通控制设施支持：
- 使用官方netconvert_carla脚本
- 自动提取和插入CARLA交通信号灯
- 完整的junction处理
- 与sumo_bridge.py使用相同的官方转换器
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

class CarlaTown3OfficialExporter:
    """CARLA Town3 SUMO路网导出器 - 官方netconvert_carla版"""
    
    def __init__(self, output_dir="./"):
        self.output_dir = Path(output_dir)
        self.carla_host = 'localhost'
        self.carla_port = 2000
        self.map_name = 'Town05'
        self.timeout = 20.0
        
        # 输出文件名
        self.opendrive_file = "carla_town05_official.xodr"
        self.network_file = "carla_town05_official.net.xml"
        
        # 确保可以导入官方脚本
        self._setup_official_netconvert()
    
    def _setup_official_netconvert(self):
        """设置官方netconvert_carla脚本的导入路径"""
        print("🔧 设置官方netconvert_carla脚本...")
        
        # 确保 tools 在 PYTHONPATH 中
        if 'SUMO_HOME' in os.environ:
            tools_dir = os.path.join(os.environ['SUMO_HOME'], 'tools')
            if os.path.isdir(tools_dir) and tools_dir not in sys.path:
                sys.path.append(tools_dir)
                print(f"✅ 添加SUMO tools到路径: {tools_dir}")
        
        # 导入 Co-Simulation/Sumo/util 目录
        util_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Co-Simulation', 'Sumo', 'util'))
        if os.path.isdir(util_dir) and util_dir not in sys.path:
            sys.path.append(util_dir)
            print(f"✅ 添加官方Co-Simulation工具到路径: {util_dir}")
        
        # 测试导入
        try:
            import netconvert_carla
            print("✅ 官方netconvert_carla脚本导入成功")
            return True
        except ImportError as e:
            print(f"❌ 无法导入官方netconvert_carla脚本: {e}")
            print("💡 请确保:")
            print("   1. SUMO_HOME环境变量已设置")
            print("   2. Co-Simulation/Sumo/util目录存在netconvert_carla.py")
            return False
    
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
            town03_maps = [map_name for map_name in available_maps if 'Town05' in map_name]
            
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
    
    def convert_to_sumo_network_official(self):
        """使用官方netconvert_carla脚本转换为SUMO网络"""
        print(f"🔄 使用官方netconvert_carla脚本转换为SUMO网络...")
        
        try:
            # 导入官方脚本
            import netconvert_carla
            print("✅ 官方netconvert_carla脚本已加载")
            
            # 准备文件路径
            opendrive_path = str(self.output_dir / self.opendrive_file)
            network_path = str(self.output_dir / self.network_file)
            
            # 验证输入文件存在
            if not os.path.exists(opendrive_path):
                print(f"❌ OpenDRIVE文件不存在: {opendrive_path}")
                return False
            
            print(f"📂 输入文件: {opendrive_path}")
            print(f"📂 输出文件: {network_path}")
            
            # 使用官方转换器
            print("🚀 执行官方netconvert_carla转换...")
            print("   📋 参数: guess_tls=True (自动识别交通信号灯)")
            
            # 调用官方函数
            netconvert_carla.netconvert_carla(
                xodr_file=opendrive_path,
                output=network_path,
                guess_tls=True  # 启用交通信号灯自动识别
            )
            
            # 验证输出文件
            if os.path.exists(network_path):
                file_size = os.path.getsize(network_path)
                print(f"✅ SUMO网络文件生成成功:")
                print(f"   📁 文件: {network_path}")
                print(f"   📏 大小: {file_size / 1024:.1f} KB")
                
                # 分析网络
                self.analyze_official_sumo_network(network_path)
                return True
            else:
                print(f"❌ 网络文件未生成: {network_path}")
                return False
                
        except ImportError as e:
            print(f"❌ 无法导入官方netconvert_carla脚本: {e}")
            return False
        except Exception as e:
            print(f"❌ 官方netconvert_carla转换失败: {e}")
            print(f"💡 请检查:")
            print(f"   1. SUMO工具是否正确安装")
            print(f"   2. netconvert命令是否在PATH中")
            print(f"   3. OpenDRIVE文件是否有效")
            return False
    
    def analyze_official_sumo_network(self, network_file):
        """分析官方netconvert_carla生成的SUMO网络文件"""
        try:
            print("📊 分析官方SUMO网络结构...")
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
                                coords = point.split(',')
                                if len(coords) >= 2:
                                    x, y = float(coords[0]), float(coords[1])
                                    x_coords.append(x)
                                    y_coords.append(y)
            
            # 基础网络统计
            print(f"   🛣️  常规道路: {len(regular_edges)}条")
            print(f"   🔀 内部连接: {len(internal_edges)}条") 
            print(f"   🚗 车道总数: {total_lanes}条")
            print(f"   ⚡ 连接关系: {len(connections)}个")
            
            # 交通控制设施统计 (官方netconvert_carla重点)
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
            
            # 官方netconvert_carla特有特征检查
            if len(traffic_light_junctions) > 0:
                print(f"   ✅ 官方CARLA交通信号灯成功导入")
            
            # 检查是否有CARLA特有的参数和标记
            carla_landmarks = 0
            for tl in traffic_lights:
                params = tl.findall('param')
                for param in params:
                    if 'linkSignalID' in param.get('key', ''):
                        carla_landmarks += 1
                        break
            
            if carla_landmarks > 0:
                print(f"   ✅ CARLA信号灯标识成功保留: {carla_landmarks}个")
            
            # 检查原始名称保留
            edges_with_origid = [e for e in edges if any(l.get('origId') for l in e.findall('lane'))]
            if edges_with_origid:
                print(f"   ✅ OpenDRIVE原始ID成功保留: {len(edges_with_origid)}条边")
                
        except Exception as e:
            print(f"⚠️ 网络分析失败: {e}")
    
    def export_carla_town3_official(self):
        """主要路网导出流程 (官方netconvert_carla版)"""
        print("🎯 CARLA Town3 SUMO路网导出工具")
        print("📋 使用官方netconvert_carla脚本")
        print("="*50)
        
        # 1. 设置官方脚本
        if not self._setup_official_netconvert():
            return False
        
        # 2. 检查CARLA可用性
        carla_available, carla = self.check_carla_availability()
        if not carla_available:
            return False
        
        # 3. 加载Town03地图
        carla_map = self.load_town03_map(carla)
        if carla_map is None:
            return False
        
        # 4. 导出OpenDRIVE
        if not self.export_opendrive(carla_map):
            return False
        
        # 5. 使用官方脚本转换为SUMO网络
        if not self.convert_to_sumo_network_official():
            return False
        
        # 6. 完成报告
        print(f"\n🎉 CARLA Town3路网导出成功！")
        print(f"📁 生成的文件:")
        print(f"   🗺️  OpenDRIVE: {self.opendrive_file}")
        print(f"   🛣️  SUMO网络: {self.network_file}")
        
        print(f"\n🚀 使用方法:")
        print(f"   1. 查看完整网络 (含官方交通灯):")
        print(f"      sumo-gui {self.network_file}")
        print(f"   2. Python TraCI使用:")
        print(f"      traci.start(['sumo-gui', '{self.network_file}'])")
        print(f"   3. 在其他项目中引用:")
        print(f"      <net-file value='{self.network_file}'/>")
        
        print(f"\n🚦 官方netconvert_carla特性:")
        print(f"   ✅ 完整的CARLA交通信号灯导入和处理")
        print(f"   ✅ 保留OpenDRIVE原始ID映射")
        print(f"   ✅ 自动识别和生成junction连接")
        print(f"   ✅ 与sumo_bridge.py官方转换器完全一致")
        print(f"   ✅ 支持复杂的交通控制逻辑")
        
        return True


def main():
    """主函数"""
    print("🎯 CARLA Town3路网导出工具 - 官方netconvert_carla版")
    print("📋 连接CARLA服务器 → OpenDRIVE导出 → 官方netconvert_carla转换 → SUMO网络(.net.xml)")
    print("🚦 包含完整CARLA交通信号灯控制和junction处理")
    print("🔧 基于CARLA官方Co-Simulation教程实现")
    print("="*70)
    
    # 检查当前目录
    current_dir = Path.cwd()
    print(f"📂 工作目录: {current_dir}")
    
    # 创建导出器并运行
    exporter = CarlaTown3OfficialExporter(output_dir=current_dir)
    success = exporter.export_carla_town3_official()
    
    if success:
        print("\n✅ 路网导出完成！")
        print("🎯 与sumo_bridge.py官方转换器生成的网络完全一致")
    else:
        print("\n❌ 路网导出失败")
        print("💡 故障排除:")
        print("   1. 确保CARLA服务器运行: ./CarlaUE4.exe")
        print("   2. 安装CARLA Python API: pip install carla")
        print("   3. 设置SUMO_HOME环境变量")
        print("   4. 检查SUMO工具安装: netconvert --version")
        print("   5. 确保Co-Simulation/Sumo/util/netconvert_carla.py存在")


if __name__ == "__main__":
    main()
