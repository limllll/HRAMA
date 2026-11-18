#!/usr/bin/env python3
"""
Map the sliced Lankershim vehicle set onto CARLA Town04 Edge40/39 by reusing the
original NGSIM->SUMO controller logic.

This script intentionally does not modify the original controller file.
It wraps the original controller, switches the default input CSV to the local
`lankershim_genchi_final_145_from_1635.csv`, and normalizes lane ids for this dataset.
"""

import os
import sys
import time
from collections import defaultdict

import pandas as pd
import traci


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from macad_gym.carla.sumo_NGSIM.NGSIM_to_SUMO_town04_edge4039_optimized_inte import (  # noqa: E402
    CarlaTown4Edge40NgsimController,
)


class LankershimGenchiEdge4039Controller(CarlaTown4Edge40NgsimController):
    """
    Reuse the original controller, but adapt defaults for the sliced Lankershim
    dataset in this folder.
    """

    def __init__(self):
        super(LankershimGenchiEdge4039Controller, self).__init__()

    def load_ngsim_data(self, csv_file):
        print("\n=== 加载 Lankershim 切分数据并映射到 Town04 Edge40/39 ===")
        print("输入文件: {}".format(csv_file))

        df = pd.read_csv(csv_file, low_memory=False)

        numeric_columns = [
            "Local_X",
            "Local_Y",
            "v_Vel",
            "v_Acc",
            "Frame_ID",
            "Vehicle_ID",
            "Lane_ID",
        ]
        for col in numeric_columns:
            if col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Local_X", "Local_Y", "v_Vel", "Frame_ID", "Vehicle_ID", "Lane_ID"]).copy()
        df["Frame_ID"] = df["Frame_ID"].astype(int)
        df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
        df["Lane_ID"] = df["Lane_ID"].astype(int)

        original_lane_ids = sorted(df["Lane_ID"].unique().tolist())
        print("Lane_ID 集合: {}".format(original_lane_ids))

        df = df.sort_values(by=["Frame_ID", "Vehicle_ID"]).reset_index(drop=True)

        self.vehicle_trajectories = {}
        self.frame_vehicles = defaultdict(list)
        self.total_vehicles = df["Vehicle_ID"].nunique()
        self.current_frame = int(df["Frame_ID"].min())
        self.max_frame = int(df["Frame_ID"].max())

        for vehicle_id, vehicle_data in df.groupby("Vehicle_ID"):
            self.vehicle_trajectories[int(vehicle_id)] = vehicle_data

        grouped_frames = df.groupby("Frame_ID")["Vehicle_ID"].apply(list)
        for frame_id, vehicle_ids in grouped_frames.items():
            self.frame_vehicles[int(frame_id)] = [int(v) for v in vehicle_ids]

        print("车辆数: {}".format(self.total_vehicles))
        print("帧范围: {} ~ {}".format(self.current_frame, self.max_frame))

        self.analyze_lane_y_ranges(df)
        self.setup_coordinate_transform(df)
        return df

    def run_simulation(self, csv_file, net_file):
        print("\n" + "=" * 80)
        print("Lankershim Genchi -> CARLA Town04 Edge40/39 映射仿真")
        print("=" * 80)
        print("CSV: {}".format(csv_file))
        print("NET: {}".format(net_file))

        try:
            df = self.load_ngsim_data(csv_file)
            if df is None:
                return

            route_file = os.path.join(CURRENT_DIR, "lankershim_genchi_edge4039.rou.xml")
            config_file = os.path.join(CURRENT_DIR, "lankershim_genchi_edge4039.sumocfg")

            self.create_carla_route_file(route_file)
            self.create_carla_config(net_file, route_file, config_file)

            config_abs = os.path.abspath(config_file)
            sumo_binary = os.environ.get(
                "SUMO_BINARY",
                "/home/simplexity/cybian/sumo-1.22.0/bin/sumo-gui",
            )
            sumo_cmd = [
                sumo_binary,
                "-c",
                config_abs,
                "--start",
                "--collision.action",
                "warn",
                "--step-length",
                "0.1",
            ]

            print("启动 SUMO: {}".format(sumo_binary))
            traci.start(sumo_cmd, port=8821, numRetries=10)
            time.sleep(1)
            traci.simulationStep()

            total_frames = self.max_frame - self.current_frame + 1
            start_time = time.time()

            for frame_id in range(self.current_frame, self.max_frame + 1):
                try:
                    self.current_frame = frame_id

                    current_vehicles = set(self.frame_vehicles.get(frame_id, []))
                    active_vehicles = set(self.active_vehicles.keys())

                    new_vehicles = current_vehicles - active_vehicles
                    for vehicle_id in new_vehicles:
                        if vehicle_id in self.vehicle_trajectories:
                            trajectory = self.vehicle_trajectories[vehicle_id]
                            frame_data = trajectory[trajectory["Frame_ID"] == frame_id]
                            if len(frame_data) > 0:
                                row = frame_data.iloc[0]
                                self.add_vehicle(vehicle_id, row, int(row["Lane_ID"]))

                    vehicles_to_remove = active_vehicles - current_vehicles
                    for vehicle_id in vehicles_to_remove:
                        self.remove_vehicle(vehicle_id)

                    for vehicle_id in list(self.active_vehicles.keys()):
                        if vehicle_id in self.vehicle_trajectories:
                            trajectory = self.vehicle_trajectories[vehicle_id]
                            frame_data = trajectory[trajectory["Frame_ID"] == frame_id]
                            if len(frame_data) > 0:
                                row = frame_data.iloc[0]
                                self.update_vehicle(vehicle_id, row)

                    if frame_id % 20 == 0:
                        progress = (frame_id - int(df["Frame_ID"].min())) / max(1, total_frames) * 100.0
                        print(
                            "Frame {}/{} | progress={:.1f}% | active={}".format(
                                frame_id, self.max_frame, progress, len(self.active_vehicles)
                            )
                        )

                    traci.simulationStep()
                    time.sleep(0.05)

                except KeyboardInterrupt:
                    print("用户中断仿真，结束于帧 {}".format(frame_id))
                    break
                except Exception as exc:
                    print("帧 {} 更新失败: {}".format(frame_id, exc))
                    continue

            elapsed = time.time() - start_time
            print("\n仿真完成")
            print("总车辆数: {}".format(self.total_vehicles))
            print("总帧数: {}".format(total_frames))
            print("总耗时: {:.2f}s".format(elapsed))
            print("平均 FPS: {:.2f}".format(total_frames / max(elapsed, 1e-6)))

        finally:
            try:
                traci.close()
            except Exception:
                pass


def main():
    csv_file = os.path.join(CURRENT_DIR, "lankershim_genchi_final_145_from_1635.csv")
    net_file = os.path.join(
        SRC_ROOT,
        "macad_gym",
        "carla",
        "sumo_NGSIM",
        "carla_town04_official.net.xml",
    )

    if not os.path.exists(csv_file):
        raise FileNotFoundError("找不到输入 CSV: {}".format(csv_file))
    if not os.path.exists(net_file):
        raise FileNotFoundError("找不到 Town04 net 文件: {}".format(net_file))

    controller = LankershimGenchiEdge4039Controller()
    controller.run_simulation(csv_file, net_file)


if __name__ == "__main__":
    main()
