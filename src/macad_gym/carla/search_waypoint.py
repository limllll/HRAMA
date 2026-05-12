#!/usr/bin/env python

"""
CARLA道路信息获取工具
无需手动控制车辆，直接通过API获取道路位置信息
"""

import carla
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 解决matplotlib中文字体显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class CarlaRoadExplorer:
    def __init__(self, host='localhost', port=2000):
        """初始化CARLA客户端连接"""
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

    def get_all_roads_info(self):
        """获取所有道路的基本信息"""
        print("=== CARLA地图道路信息 ===")
        print(f"地图名称: {self.map.name}")

        # 获取所有道路拓扑
        topology = self.map.get_topology()
        print(f"道路段数量: {len(topology)}")

        road_info = {}
        for i, (waypoint_start, waypoint_end) in enumerate(topology):
            road_id = waypoint_start.road_id
            if road_id not in road_info:
                road_info[road_id] = {
                    'road_id': road_id,
                    'sections': [],
                    'lane_count': 0
                }

            section_info = {
                'section_id': waypoint_start.section_id,
                'lane_id': waypoint_start.lane_id,
                'start_location': waypoint_start.transform.location,
                'end_location': waypoint_end.transform.location,
                'lane_type': waypoint_start.lane_type,
                'lane_width': waypoint_start.lane_width
            }
            road_info[road_id]['sections'].append(section_info)
            road_info[road_id]['lane_count'] = max(road_info[road_id]['lane_count'],
                                                   abs(waypoint_start.lane_id))

        return road_info

    def get_road_waypoints(self, road_id, distance=2.0):
        """获取指定道路的所有waypoints"""
        print(f"\n=== 道路 {road_id} 的详细信息 ===")

        # 获取该道路的所有waypoints
        waypoints = []
        topology = self.map.get_topology()

        for waypoint_start, waypoint_end in topology:
            if waypoint_start.road_id == road_id:
                # 从起点开始，按指定距离生成waypoints
                current_wp = waypoint_start
                road_waypoints = [current_wp]

                while True:
                    next_wps = current_wp.next(distance)
                    if not next_wps or next_wps[0].road_id != road_id:
                        break
                    current_wp = next_wps[0]
                    road_waypoints.append(current_wp)

                waypoints.extend(road_waypoints)

        return waypoints

    def find_nearest_road(self, location):
        """根据坐标找到最近的道路"""
        target_location = carla.Location(x=location[0], y=location[1], z=location[2])
        waypoint = self.map.get_waypoint(target_location)

        if waypoint:
            return {
                'road_id': waypoint.road_id,
                'section_id': waypoint.section_id,
                'lane_id': waypoint.lane_id,
                'location': waypoint.transform.location,
                'lane_type': waypoint.lane_type,
                'lane_width': waypoint.lane_width,
                'is_junction': waypoint.is_junction
            }
        return None

    def get_spawn_points(self):
        """获取所有可用的生成点"""
        spawn_points = self.map.get_spawn_points()

        spawn_info = []
        for i, spawn_point in enumerate(spawn_points):
            waypoint = self.map.get_waypoint(spawn_point.location)
            spawn_info.append({
                'id': i,
                'location': spawn_point.location,
                'rotation': spawn_point.rotation,
                'road_id': waypoint.road_id if waypoint else None,
                'lane_id': waypoint.lane_id if waypoint else None
            })

        return spawn_info

    def export_road_data_to_csv(self, filename="carla_roads.csv"):
        """导出道路数据到CSV文件"""
        import csv

        roads_info = self.get_all_roads_info()

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['road_id', 'section_id', 'lane_id', 'start_x', 'start_y', 'start_z',
                          'end_x', 'end_y', 'end_z', 'lane_type', 'lane_width']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for road_id, road_data in roads_info.items():
                for section in road_data['sections']:
                    writer.writerow({
                        'road_id': road_id,
                        'section_id': section['section_id'],
                        'lane_id': section['lane_id'],
                        'start_x': section['start_location'].x,
                        'start_y': section['start_location'].y,
                        'start_z': section['start_location'].z,
                        'end_x': section['end_location'].x,
                        'end_y': section['end_location'].y,
                        'end_z': section['end_location'].z,
                        'lane_type': str(section['lane_type']),
                        'lane_width': section['lane_width']
                    })

        print(f"道路数据已导出到 {filename}")

    def visualize_map_2d(self, show_spawn_points=True):
        """2D可视化地图"""
        topology = self.map.get_topology()

        # 提取所有waypoints的坐标
        x_coords = []
        y_coords = []

        for waypoint_start, waypoint_end in topology:
            x_coords.extend([waypoint_start.transform.location.x, waypoint_end.transform.location.x])
            y_coords.extend([waypoint_start.transform.location.y, waypoint_end.transform.location.y])

        plt.figure(figsize=(12, 10))
        plt.scatter(x_coords, y_coords, s=1, alpha=0.6, label='Road Points')

        if show_spawn_points:
            spawn_points = self.get_spawn_points()
            spawn_x = [sp['location'].x for sp in spawn_points]
            spawn_y = [sp['location'].y for sp in spawn_points]
            plt.scatter(spawn_x, spawn_y, s=50, c='red', marker='x', label='Spawn Points')

        plt.xlabel('X Coordinate (meters)')
        plt.ylabel('Y Coordinate (meters)')
        plt.title(f'CARLA Map: {self.map.name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


def main():
    """主函数演示各种功能"""
    try:
        # 创建道路探索器
        explorer = CarlaRoadExplorer()

        print("CARLA道路信息获取工具启动...")

        # 1. 获取所有道路信息
        roads_info = explorer.get_all_roads_info()
        print(f"\n找到 {len(roads_info)} 条道路")

        # 2. 显示前几条道路的详细信息
        for i, (road_id, road_data) in enumerate(list(roads_info.items())[:3]):
            print(f"\n道路 {road_id}: 车道数 {road_data['lane_count']}, 路段数 {len(road_data['sections'])}")

            # 获取该道路的waypoints
            waypoints = explorer.get_road_waypoints(road_id)
            print(f"  共有 {len(waypoints)} 个waypoints")

            if waypoints:
                print(f"  起点: ({waypoints[0].transform.location.x:.2f}, {waypoints[0].transform.location.y:.2f})")
                print(f"  终点: ({waypoints[-1].transform.location.x:.2f}, {waypoints[-1].transform.location.y:.2f})")

        # 3. 获取生成点信息
        spawn_points = explorer.get_spawn_points()
        print(f"\n找到 {len(spawn_points)} 个车辆生成点")

        # 4. 测试位置查询功能
        test_locations = [
            [100, 200, 0],
            [0, 0, 0],
            [-100, -100, 0]
        ]

        print("\n=== 位置查询测试 ===")
        for loc in test_locations:
            road_info = explorer.find_nearest_road(loc)
            if road_info:
                print(f"位置 {loc} 最近的道路: 道路ID {road_info['road_id']}, 车道ID {road_info['lane_id']}")
            else:
                print(f"位置 {loc} 附近没有找到道路")

        # 5. 导出数据
        explorer.export_road_data_to_csv()

        # 6. Visualization (optional)
        print("\nShow map visualization? (y/n): ", end="")
        if input().lower() == 'y':
            explorer.visualize_map_2d()

    except Exception as e:
        print(f"错误: {e}")
        print("请确保CARLA服务器正在运行 (默认端口2000)")


if __name__ == "__main__":
    main()