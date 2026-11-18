import argparse
import os
import time

import pandas as pd


def slice_by_frame_range(input_file, output_file, start_frame, frame_length, frame_column="Frame_ID"):
    """
    从 CSV 文件中按帧范围截取数据。

    参数:
        input_file: 输入 CSV 文件路径
        output_file: 输出 CSV 文件路径
        start_frame: 起始帧号（包含）
        frame_length: 截取帧长
        frame_column: 帧列名，默认使用 Frame_ID
    """
    print("开始处理文件: {}".format(input_file))
    start_time = time.time()

    if not os.path.exists(input_file):
        raise FileNotFoundError("输入文件不存在: {}".format(input_file))

    if frame_length <= 0:
        raise ValueError("frame_length 必须大于 0")

    df = pd.read_csv(input_file, low_memory=False)
    print("原始数据行数: {}".format(len(df)))

    if frame_column not in df.columns:
        possible_columns = [col for col in df.columns if "frame" in col.lower()]
        if not possible_columns:
            raise ValueError(
                "未找到帧列 '{}'，可用列名为: {}".format(frame_column, df.columns.tolist())
            )
        frame_column = possible_columns[0]
        print("未找到 Frame_ID，改用列: {}".format(frame_column))

    df[frame_column] = pd.to_numeric(df[frame_column], errors="coerce")
    df = df.dropna(subset=[frame_column])
    df[frame_column] = df[frame_column].astype(int)

    end_frame = start_frame + frame_length - 1
    sliced_df = df[(df[frame_column] >= start_frame) & (df[frame_column] <= end_frame)].copy()
    sliced_df = sliced_df.sort_values(by=frame_column)

    print("截取帧范围: {} ~ {}".format(start_frame, end_frame))
    print("截取后数据行数: {}".format(len(sliced_df)))

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sliced_df.to_csv(output_file, index=False)

    elapsed_time = time.time() - start_time
    print("处理完成，总耗时: {:.2f} 秒".format(elapsed_time))
    print("输出文件: {}".format(output_file))


def main():
    parser = argparse.ArgumentParser(description="从 CSV 中按起始帧和固定帧长切片")
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", required=True, help="输出 CSV 文件路径")
    parser.add_argument("--start-frame", required=True, type=int, help="起始帧号（包含）")
    parser.add_argument("--length", required=True, type=int, help="截取帧长")
    parser.add_argument("--frame-column", default="Frame_ID", help="帧列名，默认 Frame_ID")
    args = parser.parse_args()

    slice_by_frame_range(
        input_file=args.input,
        output_file=args.output,
        start_frame=args.start_frame,
        frame_length=args.length,
        frame_column=args.frame_column,
    )


if __name__ == "__main__":
    # 直接在代码中指定参数时，将 USE_INLINE_CONFIG 设为 True。
    # 需要命令行传参时，将其设为 False。
    USE_INLINE_CONFIG = True

    if USE_INLINE_CONFIG:
        input_file = "lankershim_final_266_frame398.csv"
        output_file = "lankershim_final_266_slice_145.csv"
        start_frame = 1630
        frame_length = 145
        frame_column = "Frame_ID"

        slice_by_frame_range(
            input_file=input_file,
            output_file=output_file,
            start_frame=start_frame,
            frame_length=frame_length,
            frame_column=frame_column,
        )
    else:
        main()
