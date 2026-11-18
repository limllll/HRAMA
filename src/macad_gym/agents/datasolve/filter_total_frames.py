import pandas as pd
import os
import time

def filter_by_total_frames(input_file, output_file, target_frames):
    """
    从CSV文件中筛选出指定Total_Frames值的数据
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        target_frames: 需要保留的Total_Frames值
    """
    print(f"开始处理文件: {input_file}")
    start_time = time.time()
    
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 {input_file}")
        return
    
    # 读取CSV文件
    try:
        df = pd.read_csv(input_file, low_memory=False)
        print(f"原始数据行数: {len(df)}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 检查Total_Frames列是否存在
    if 'Total_Frames' not in df.columns:
        # 尝试查找可能的帧总数列
        possible_columns = [col for col in df.columns if 'total' in col.lower() and 'frame' in col.lower()]
        if possible_columns:
            total_frames_column = possible_columns[0]
            print(f"使用 {total_frames_column} 替代 Total_Frames")
        else:
            # 显示列名供参考
            print("错误: 未找到Total_Frames列或类似列。可用的列名有:")
            print(df.columns.tolist())
            return
    else:
        total_frames_column = 'Total_Frames'
    
    # 筛选Total_Frames列值为target_frames的行
    # 将目标值转为字符串进行比较，确保能匹配字符串格式的值
    filtered_df = df[df[total_frames_column].astype(str) == str(target_frames)]
    print(f"筛选后数据行数: {len(filtered_df)}")
    
    if len(filtered_df) == 0:
        print(f"警告: 没有找到Total_Frames为\"{target_frames}\"的数据")
        # 显示Total_Frames列的唯一值以供参考
        unique_values = [str(v) for v in sorted(df[total_frames_column].unique())]
        print(f"现有的{total_frames_column}值有: {unique_values}")
        return
    
    # 保存到新的CSV文件
    filtered_df.to_csv(output_file, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"处理完成！总耗时: {elapsed_time:.2f}秒")
    print(f"数据已保存到: {output_file}")

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "lankershim_final_266.csv"
    output_file = "lankershim_final_266_frame398.csv"
    
    # 目标Total_Frames值
    target_frames = "398"
    
    filter_by_total_frames(input_file, output_file, target_frames)
