import numpy as np
import open3d as o3d
import torch
from macad_gym.PointNet2.models.pointnet2_cls_msg import get_model, get_loss

# 加载点云数据
point = np.load("./lidar/76.npy")
print("原始点云形状:", point.shape)

# 体素化降采样
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point[:,:3])
downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.1)
downsampled_points = np.asarray(downsampled_pcd.points)
print("降采样后点云形状:", downsampled_points.shape)

# 归一化到[-1,1]
def normalize(points, target_range=(-1,1)):
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    points = (points - min_point) / (max_point - min_point)
    points = points * (target_range[1] - target_range[0]) + target_range[0]
    return points

downsampled_points = normalize(downsampled_points)
print("归一化后点云范围:", np.min(downsampled_points), "到", np.max(downsampled_points))

# 初始化模型（不使用法向量）
model = get_model(num_class=40, normal_channel=False)

# 选择性加载权重（跳过法向量相关权重）
checkpoint = torch.load("../log/classification/pointnet2_msg_normals/checkpoints/best_model.pth")
model_dict = model.state_dict()

# 1. 筛选可加载的权重
pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if k in model_dict and v.shape == model_dict[k].shape}

# 2. 特殊处理第一层卷积权重（从6通道改为3通道）
for key in ['sa1.conv_blocks.0.0.weight', 'sa1.conv_blocks.1.0.weight', 'sa1.conv_blocks.2.0.weight']:
    if key in checkpoint['model_state_dict']:
        # 取前3个输入通道的权重（XYZ），舍弃后3个（法向量）
        pretrained_dict[key] = checkpoint['model_state_dict'][key][:, :3, :, :]

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print("成功加载匹配的权重参数")

model.eval()

# 准备输入数据（只使用XYZ坐标）
downsampled_points = torch.from_numpy(downsampled_points).unsqueeze(0).float().transpose(1, 2)
print("模型输入形状:", downsampled_points.shape)

# 推理
with torch.no_grad():
    pred, features, my = model(downsampled_points)
    # print("分类预测结果:", pred.shape)

    print("my特征向量:", my.shape)
    print("my特征向量:", np.array(my).shape)