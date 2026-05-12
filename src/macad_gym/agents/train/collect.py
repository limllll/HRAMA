# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """简单的CARLA数据采集脚本"""

# import glob
# import os
# import sys
# import time
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from queue import Queue
#
# # 添加CARLA路径
# try:
#     carla_path = glob.glob('src/macad_gym/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major, sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))
#     if carla_path:
#         sys.path.append(carla_path[0])
# except:
#     pass
#
# # 尝试其他路径
# try:
#     sys.path.append('src/macad_gym/PythonAPI/carla')
# except:
#     pass
#
# import carla
#
#
# def main():
#     actor_list = []
#
#     try:
#         # 连接服务器
#         client = carla.Client('localhost', 2000)
#         client.set_timeout(10.0)
#         world = client.load_world('Town04')
#
#         # 同步模式
#         settings = world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = 0.05
#         world.apply_settings(settings)
#
#         # 生成Lincoln车辆
#         bp_lib = world.get_blueprint_library()
#         vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
#         spawn_point = np.random.choice(world.get_map().get_spawn_points())
#         vehicle = world.spawn_actor(vehicle_bp, spawn_point)
#         actor_list.append(vehicle)
#         vehicle.set_autopilot(True)
#         print(f"✅ 车辆已生成")
#
#         # 摄像头
#         cam_bp = bp_lib.find('sensor.camera.rgb')
#         cam_bp.set_attribute('image_size_x', '800')
#         cam_bp.set_attribute('image_size_y', '600')
#         cam_transform = carla.Transform(carla.Location(x=2.0, z=1.8))
#         camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
#         actor_list.append(camera)
#
#         # 激光雷达
#         lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
#         lidar_bp.set_attribute('channels', '32')
#         lidar_bp.set_attribute('range', '50')
#         lidar_bp.set_attribute('points_per_second', '100000')
#         lidar_transform = carla.Transform(carla.Location(z=2.5))
#         lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
#         actor_list.append(lidar)
#         print(f"✅ 传感器已配置")
#
#         # 采集数据
#         img_queue = Queue()
#         lidar_queue = Queue()
#         camera.listen(lambda data: img_queue.put(data))
#         lidar.listen(lambda data: lidar_queue.put(data))
#
#         time.sleep(2)
#         world.tick()
#
#         # 保存图像
#         img_data = img_queue.get(timeout=2.0)
#         array = np.frombuffer(img_data.raw_data, dtype=np.uint8)
#         array = np.reshape(array, (img_data.height, img_data.width, 4))[:, :, :3]
#         cv2.imwrite('camera.png', array[:, :, ::-1])
#         print(f"✅ 图像已保存: camera.png")
#
#         # 保存点云
#         lidar_data = lidar_queue.get(timeout=2.0)
#         points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
#         points = np.reshape(points, (-1, 4))
#         np.save('lidar.npy', points)
#
#         # 可视化点云
#         fig = plt.figure(figsize=(15, 5))
#         ax1 = fig.add_subplot(131, projection='3d')
#         ax1.scatter(points[::10, 0], points[::10, 1], points[::10, 2],
#                     c=points[::10, 2], s=1, cmap='viridis')
#         ax1.set_title('3D视图')
#
#         ax2 = fig.add_subplot(132)
#         ax2.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.5, cmap='viridis')
#         ax2.set_title('俯视图')
#         ax2.axis('equal')
#
#         ax3 = fig.add_subplot(133)
#         ax3.scatter(points[:, 0], points[:, 2], c=points[:, 3], s=0.5, cmap='hot')
#         ax3.set_title('侧视图')
#         ax3.axis('equal')
#
#         plt.tight_layout()
#         plt.savefig('lidar_viz.png', dpi=150)
#         print(f"✅ 点云已保存: lidar.npy, lidar_viz.png (共{len(points)}个点)")
#
#     finally:
#         # 清理
#         for actor in actor_list:
#             actor.destroy()
#         settings.synchronous_mode = False
#         world.apply_settings(settings)
#         print("✅ 完成")
#
#
# if __name__ == '__main__':
#     main()
# import glob
# import os
# import sys
# import time
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from queue import Queue
# from matplotlib.backends.backend_agg import FigureCanvasAgg
#
# # 添加CARLA路径
# try:
#     carla_path = glob.glob('src/macad_gym/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major, sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))
#     if carla_path:
#         sys.path.append(carla_path[0])
# except:
#     pass
#
# # 尝试其他路径
# try:
#     sys.path.append('src/macad_gym/PythonAPI/carla')
# except:
#     pass
#
# import carla
#
#
# def save_lidar_visualization(points, filename):
#     """保存点云可视化图片"""
#     fig = plt.figure(figsize=(15, 5))
#
#     # 3D视图
#     ax1 = fig.add_subplot(131, projection='3d')
#     ax1.scatter(points[::10, 0], points[::10, 1], points[::10, 2],
#                 c=points[::10, 2], s=1, cmap='viridis')
#     ax1.set_title('3D视图')
#     ax1.set_xlabel('X')
#     ax1.set_ylabel('Y')
#     ax1.set_zlabel('Z')
#
#     # 俯视图
#     ax2 = fig.add_subplot(132)
#     ax2.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.5, cmap='viridis')
#     ax2.set_title('俯视图')
#     ax2.set_xlabel('X')
#     ax2.set_ylabel('Y')
#     ax2.axis('equal')
#
#     # 侧视图
#     ax3 = fig.add_subplot(133)
#     ax3.scatter(points[:, 0], points[:, 2], c=points[:, 3], s=0.5, cmap='hot')
#     ax3.set_title('侧视图')
#     ax3.set_xlabel('X')
#     ax3.set_ylabel('Z')
#     ax3.axis('equal')
#
#     plt.tight_layout()
#     plt.savefig(filename, dpi=150, bbox_inches='tight')
#     plt.close(fig)  # 关闭图形释放内存
#
#
# def main():
#     actor_list = []
#
#     try:
#         # 连接服务器
#         client = carla.Client('localhost', 2000)
#         client.set_timeout(10.0)
#         world = client.load_world('Town04')
#
#         # 同步模式
#         settings = world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = 0.05
#         world.apply_settings(settings)
#
#         # 生成Lincoln车辆
#         bp_lib = world.get_blueprint_library()
#         vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
#         spawn_point = np.random.choice(world.get_map().get_spawn_points())
#         vehicle = world.spawn_actor(vehicle_bp, spawn_point)
#         actor_list.append(vehicle)
#         vehicle.set_autopilot(True)
#         print(f"✅ 车辆已生成")
#
#         # 摄像头
#         cam_bp = bp_lib.find('sensor.camera.rgb')
#         cam_bp.set_attribute('image_size_x', '800')
#         cam_bp.set_attribute('image_size_y', '600')
#         cam_transform = carla.Transform(carla.Location(x=2.0, z=1.8))
#         camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
#         actor_list.append(camera)
#
#         # 激光雷达
#         lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
#         lidar_bp.set_attribute('channels', '32')
#         lidar_bp.set_attribute('range', '50')
#         lidar_bp.set_attribute('points_per_second', '100000')
#         lidar_transform = carla.Transform(carla.Location(z=2.5))
#         lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
#         actor_list.append(lidar)
#         print(f"✅ 传感器已配置")
#
#         # 创建队列
#         img_queue = Queue()
#         lidar_queue = Queue()
#
#         # 监听传感器
#         camera.listen(lambda data: img_queue.put(data))
#         lidar.listen(lambda data: lidar_queue.put(data))
#
#         # 等待传感器初始化
#         time.sleep(2)
#
#         # 创建保存数据的文件夹
#         output_dir = "carla_data"
#         os.makedirs(output_dir, exist_ok=True)
#
#         # 采集5帧数据
#         for i in range(5):
#             print(f"\n🎬 正在采集第 {i + 1}/5 帧...")
#
#             # 让世界前进一帧
#             world.tick()
#
#             # 获取数据
#             try:
#                 # 获取图像
#                 img_data = img_queue.get(timeout=2.0)
#                 array = np.frombuffer(img_data.raw_data, dtype=np.uint8)
#                 array = np.reshape(array, (img_data.height, img_data.width, 4))[:, :, :3]
#
#                 # 保存图像
#                 img_filename = os.path.join(output_dir, f"camera_{i:03d}.png")
#                 cv2.imwrite(img_filename, array[:, :, ::-1])
#                 print(f"✅ 图像已保存: {img_filename}")
#
#                 # 获取点云
#                 lidar_data = lidar_queue.get(timeout=2.0)
#                 points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
#                 points = np.reshape(points, (-1, 4))
#
#                 # 保存点云数据
#                 lidar_filename = os.path.join(output_dir, f"lidar_{i:03d}.npy")
#                 np.save(lidar_filename, points)
#
#                 # 保存点云可视化
#                 viz_filename = os.path.join(output_dir, f"lidar_viz_{i:03d}.png")
#                 save_lidar_visualization(points, viz_filename)
#
#                 print(f"✅ 点云已保存: {lidar_filename}")
#                 print(f"✅ 可视化已保存: {viz_filename} (共{len(points)}个点)")
#
#                 # 显示当前车辆位置
#                 vehicle_location = vehicle.get_location()
#                 print(f"📍 车辆位置: ({vehicle_location.x:.2f}, {vehicle_location.y:.2f}, {vehicle_location.z:.2f})")
#
#             except Exception as e:
#                 print(f"❌ 第{i + 1}帧数据采集失败: {e}")
#                 continue
#
#             # 可选：短暂等待，让车辆移动一定距离
#             time.sleep(0.5)
#
#         print(f"\n🎉 数据采集完成！所有数据保存在 '{output_dir}' 文件夹中")
#
#     except KeyboardInterrupt:
#         print("\n⚠️ 用户中断")
#     except Exception as e:
#         print(f"\n❌ 程序出错: {e}")
#     finally:
#         # 清理
#         print("\n🧹 正在清理...")
#         for actor in actor_list:
#             if actor.is_alive:
#                 actor.destroy()
#
#         # 恢复异步模式
#         try:
#             settings = world.get_settings()
#             settings.synchronous_mode = False
#             world.apply_settings(settings)
#         except:
#             pass
#
#         print("✅ 清理完成")
#
#
# if __name__ == '__main__':
#     main()

import open3d as o3d
import numpy as np
points = np.load('carla_data/lidar_004.npy')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:,:3])
o3d.visualization.draw_geometries([pcd])