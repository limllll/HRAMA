import pandas as pd
import time
import os

def extract_specific_vehicles(input_file, output_file, vehicle_ids):
    """
    从CSV文件中提取特定车辆ID的数据，并按照车辆ID升序排列
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        vehicle_ids: 需要提取的车辆ID列表
    """
    print(f"开始处理文件: {input_file}")
    if os.path.exists(input_file):
        file_size = os.path.getsize(input_file) / (1024 * 1024)
        print(f"文件大小: {file_size:.2f} MB")
    else:
        print(f"文件不存在: {input_file}")
        return
    
    start_time = time.time()
    
    # 读取CSV文件
    print("正在读取CSV文件...")
    try:
        # 假设车辆ID列名为'Vehicle_ID'，如果不是，请修改为正确的列名
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    print(f"原始数据行数: {len(df)}")
    
    # 检查数据中是否有车辆ID列
    if 'Vehicle_ID' not in df.columns:
        # 尝试查找可能的车辆ID列
        possible_id_columns = [col for col in df.columns if 'id' in col.lower() or 'vehicle' in col.lower()]
        if possible_id_columns:
            print(f"未找到'Vehicle_ID'列，可能的ID列有: {possible_id_columns}")
        else:
            print("未找到车辆ID列，请检查CSV文件结构")
            # 打印前几行数据以查看列名
            print("CSV文件的列名:")
            print(df.columns.tolist())
            print("CSV文件的前5行:")
            print(df.head())
            return
        
        # 让用户选择正确的列名或者手动设置
        vehicle_id_column = possible_id_columns[0] if possible_id_columns else None
        if not vehicle_id_column:
            return
        print(f"使用 {vehicle_id_column} 作为车辆ID列")
    else:
        vehicle_id_column = 'Vehicle_ID'
    
    # 筛选特定车辆ID的数据
    print(f"筛选车辆ID为 {vehicle_ids} 的数据...")
    filtered_df = df[df[vehicle_id_column].isin(vehicle_ids)]
    print(f"筛选后数据行数: {len(filtered_df)}")
    
    # 检查是否有Frame_ID列
    if 'Frame_ID' not in filtered_df.columns:
        # 尝试查找可能的帧ID列
        possible_frame_columns = [col for col in filtered_df.columns if 'frame' in col.lower()]
        frame_id_column = possible_frame_columns[0] if possible_frame_columns else None
        
        if not frame_id_column:
            print("警告：未找到Frame_ID列")
            frame_id_column = vehicle_id_column  # 使用车辆ID列作为替代
        else:
            print(f"使用 {frame_id_column} 作为帧ID列")
    else:
        frame_id_column = 'Frame_ID'
    
    # 为每个车辆ID创建单独的CSV文件
    print(f"为每个车辆ID创建单独的CSV文件...")
    
    # 检查输出目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取输出文件的基本名称（不包含扩展名）
    base_output_file = os.path.splitext(output_file)[0]
    
    # 为每个车辆ID创建一个单独的文件
    for vehicle_id in vehicle_ids:
        # 获取特定车辆ID的数据
        vehicle_data = filtered_df[filtered_df[vehicle_id_column] == vehicle_id]
        
        if len(vehicle_data) > 0:
            # 按Frame_ID升序排列
            vehicle_data_sorted = vehicle_data.sort_values(by=frame_id_column)
            
            # 创建输出文件名
            vehicle_output_file = f"{base_output_file}_{vehicle_id}.csv"
            
            # 保存到CSV文件
            print(f"保存车辆ID {vehicle_id} 的数据到 {vehicle_output_file}（{len(vehicle_data)}行）...")
            vehicle_data_sorted.to_csv(vehicle_output_file, index=False)
        else:
            print(f"警告：未找到车辆ID为 {vehicle_id} 的数据")
    
    elapsed_time = time.time() - start_time
    print(f"处理完成！总耗时: {elapsed_time:.2f}秒")
    print(f"每个车辆的数据已分别保存为: {base_output_file}_[车辆ID].csv")

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "lankershim_filtered_data.csv"
    output_file = "lankershim_final.csv"
    
    # 需要提取的车辆ID列表
    vehicle_ids = [213, 214, 216, 215, 235, 239, 242, 219, 254, 266, 245, 221, 231, 236, 247]
    
    extract_specific_vehicles(input_file, output_file, vehicle_ids)
