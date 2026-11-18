#!/usr/bin/env python3
"""
Rewritten NGSIM -> SUMO controller for CARLA Town04 Edge40/39.

This version keeps the original goal of replaying NGSIM trajectories on the
Town04 road network, but it avoids hard-coding most geometric assumptions.
Key route and lane geometry are parsed from the SUMO network file so that edge
selection, lane placement, and route progression stay consistent.
"""

import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import sumolib
import traci
from xml.dom import minidom


@dataclass
class LaneGeometry:
    lane_id: str
    index: int
    edge_id: str
    y_center: float
    x_start: float
    x_end: float
    width: float
    length: float


@dataclass
class EdgeGeometry:
    edge_id: str
    x_start: float
    x_end: float
    min_y: float
    max_y: float
    length: float
    driving_lanes: Dict[int, LaneGeometry]


class CarlaTown4Edge4039RewriteController:
    def __init__(self) -> None:
        self.feet_to_meters = 0.3048
        self.current_frame = 1
        self.max_frame = 1
        self.total_vehicles = 0

        self.vehicle_trajectories: Dict[int, pd.DataFrame] = {}
        self.frame_vehicles: Dict[int, List[int]] = defaultdict(list)
        self.active_vehicles: Dict[int, Dict[str, object]] = {}
        self.vehicle_position_history: Dict[int, List[Tuple[float, float]]] = {}
        self.vehicle_control_modes: Dict[int, str] = {}
        self.control_mode_switch_frame: Dict[int, int] = {}
        self.safety_overrides: Dict[int, bool] = {}

        self.transform_params: Dict[str, object] = {}
        self.edge_geometries: Dict[str, EdgeGeometry] = {}
        self.lane_x_ranges: Dict[int, Dict[str, float]] = {}
        self.route_edges: List[str] = ["40", "39"]
        self.route_total_length = 0.0
        self.ngsim_lane_to_sumo_lane = {1: 4, 2: 3, 3: 2}

        self.hybrid_control_params = {
            "ngsim_guidance_frames": 30,
            "transition_frames": 20,
            "enable_sumo_takeover": True,
        }
        self.sumo_models = {
            "carFollowModel": "Krauss",
            "laneChangeModel": "LC2013",
            "tau": 1.5,
            "sigma": 0.5,
        }
        self.safety_params = {
            "min_gap": 2.5,
            "collision_threshold": 2.0,
            "speed_adaptation_rate": 0.4,
            "max_decel_comfort": 3.0,
            "max_decel_emergency": 7.0,
        }
        self.lateral_params = {
            "vehicle_half_width": 0.9,
            "safety_margin": 0.1,
            "enable_debug": True,
        }
        self.stats = {
            "vehicles_added": 0,
            "vehicles_removed": 0,
            "total_updates": 0,
            "mode_ngsim": 0,
            "mode_hybrid": 0,
            "mode_sumo": 0,
            "collision_avoidances": 0,
            "emergency_brakes": 0,
            "safety_overrides_count": 0,
        }
        self.gui_params = {
            "view_id": "View #0",
            "delay_ms": 750,
            "schema": "real world",
        }

    def analyze_lane_x_ranges(self, df: pd.DataFrame) -> None:
        self.lane_x_ranges = {}
        for lane_id in sorted(df["Lane_ID"].unique()):
            lane_data = df[df["Lane_ID"] == lane_id]
            local_x_values = lane_data["Local_X"].dropna() * self.feet_to_meters
            if len(local_x_values) == 0:
                continue

            x_min = float(local_x_values.min())
            x_max = float(local_x_values.max())
            self.lane_x_ranges[int(lane_id)] = {
                "min": x_min,
                "max": x_max,
                "center": 0.5 * (x_min + x_max),
            }

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return angle % 360.0

    def resolve_sumo_binary(self) -> str:
        preferred_binaries = [
            "/home/simplexity/cybian/sumo-1.22.0/bin/sumo-gui",
            "/home/simplexity/cybian/sumo-1.22.0/bin/sumo",
        ]
        for preferred in preferred_binaries:
            if os.path.exists(preferred):
                return preferred

        explicit = os.environ.get("SUMO_BINARY")
        if explicit and os.path.exists(explicit):
            return explicit

        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            for binary_name in ("sumo-gui", "sumo"):
                candidate = os.path.join(sumo_home, "bin", binary_name)
                if os.path.exists(candidate):
                    return candidate

        for candidate in ("sumo-gui", "sumo"):
            try:
                return sumolib.checkBinary(candidate)
            except Exception:
                continue

        raise RuntimeError(
            "Unable to locate a SUMO binary. Checked the fixed project path, "
            "SUMO_BINARY, SUMO_HOME, and system PATH."
        )

    def parse_network_geometry(self, net_file: str) -> None:
        tree = ET.parse(net_file)
        root = tree.getroot()

        parsed_edges: Dict[str, EdgeGeometry] = {}
        total_length = 0.0

        for edge_id in self.route_edges:
            edge_elem = root.find(f".//edge[@id='{edge_id}']")
            if edge_elem is None:
                raise ValueError(f"Missing edge {edge_id} in {net_file}")

            driving_lanes: Dict[int, LaneGeometry] = {}
            edge_min_y = float("inf")
            edge_max_y = float("-inf")
            edge_x_start = None
            edge_x_end = None
            edge_length = 0.0

            for lane_elem in edge_elem.findall("lane"):
                lane_type = lane_elem.get("type", "")
                if lane_type != "driving":
                    continue

                lane_id = lane_elem.attrib["id"]
                lane_index = int(lane_elem.attrib["index"])
                lane_length = float(lane_elem.attrib["length"])
                lane_width = float(lane_elem.attrib.get("width", "3.5"))
                shape = lane_elem.attrib["shape"]
                coords = [tuple(map(float, point.split(",")[:2])) for point in shape.split()]
                xs = [point[0] for point in coords]
                ys = [point[1] for point in coords]

                lane = LaneGeometry(
                    lane_id=lane_id,
                    index=lane_index,
                    edge_id=edge_id,
                    y_center=float(sum(ys) / len(ys)),
                    x_start=xs[0],
                    x_end=xs[-1],
                    width=lane_width,
                    length=lane_length,
                )
                driving_lanes[lane_index] = lane
                edge_min_y = min(edge_min_y, min(ys))
                edge_max_y = max(edge_max_y, max(ys))
                edge_length = max(edge_length, lane_length)

                if edge_x_start is None:
                    edge_x_start = xs[0]
                    edge_x_end = xs[-1]

            if not driving_lanes:
                raise ValueError(f"Edge {edge_id} has no driving lanes")

            parsed_edges[edge_id] = EdgeGeometry(
                edge_id=edge_id,
                x_start=float(edge_x_start),
                x_end=float(edge_x_end),
                min_y=edge_min_y,
                max_y=edge_max_y,
                length=edge_length,
                driving_lanes=driving_lanes,
            )
            total_length += edge_length

        self.edge_geometries = parsed_edges
        self.route_total_length = total_length

    def load_ngsim_data(self, csv_file: str) -> pd.DataFrame:
        df = pd.read_csv(csv_file)
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

        df = df.dropna(subset=["Local_X", "Local_Y", "v_Vel", "Frame_ID", "Vehicle_ID", "Lane_ID"])
        df["Frame_ID"] = df["Frame_ID"].astype(int)
        df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
        df["Lane_ID"] = df["Lane_ID"].astype(int)

        self.total_vehicles = df["Vehicle_ID"].nunique()
        self.current_frame = int(df["Frame_ID"].min())
        self.max_frame = int(df["Frame_ID"].max())

        for vehicle_id, vehicle_data in df.groupby("Vehicle_ID"):
            self.vehicle_trajectories[int(vehicle_id)] = vehicle_data

        grouped_frames = df.groupby("Frame_ID")["Vehicle_ID"].apply(list)
        for frame_id, vehicle_ids in grouped_frames.items():
            self.frame_vehicles[int(frame_id)] = [int(v) for v in vehicle_ids]

        self.analyze_lane_x_ranges(df)
        self.setup_coordinate_transform(df)
        return df

    def setup_coordinate_transform(self, df: pd.DataFrame) -> None:
        local_y_values = df["Local_Y"] * self.feet_to_meters
        local_x_values = df["Local_X"] * self.feet_to_meters

        ngsim_y_min = float(local_y_values.min())
        ngsim_y_max = float(local_y_values.max())
        ngsim_x_min = float(local_x_values.min())
        ngsim_x_max = float(local_x_values.max())
        actual_distance = ngsim_y_max - ngsim_y_min

        self.transform_params = {
            "ngsim_y_range": (ngsim_y_min, ngsim_y_max),
            "ngsim_x_range": (ngsim_x_min, ngsim_x_max),
            "actual_distance": actual_distance,
        }

    def determine_route_position(self, local_y: float) -> Tuple[str, float]:
        local_y_m = float(local_y) * self.feet_to_meters
        y_min, y_max = self.transform_params["ngsim_y_range"]
        if y_max <= y_min:
            route_distance = 0.0
        else:
            normalized = (local_y_m - y_min) / (y_max - y_min)
            normalized = max(0.0, min(1.0, normalized))
            route_distance = normalized * self.route_total_length

        edge40_length = self.edge_geometries["40"].length
        if route_distance <= edge40_length:
            return "40", route_distance
        return "39", route_distance - edge40_length

    def map_coordinates(self, local_y: float, local_x: float, lane_id: int) -> Dict[str, object]:
        edge_id, edge_distance = self.determine_route_position(local_y)
        edge_geom = self.edge_geometries[edge_id]

        source_lane_id = int(lane_id)
        lane_index = self.ngsim_lane_to_sumo_lane.get(source_lane_id, 3)
        if lane_index not in edge_geom.driving_lanes:
            lane_index = sorted(edge_geom.driving_lanes)[len(edge_geom.driving_lanes) // 2]

        lane_geom = edge_geom.driving_lanes[lane_index]
        edge_progress = 0.0 if edge_geom.length <= 1e-6 else edge_distance / edge_geom.length
        edge_progress = max(0.0, min(1.0, edge_progress))
        x = lane_geom.x_start + (lane_geom.x_end - lane_geom.x_start) * edge_progress

        local_x_m = float(local_x) * self.feet_to_meters
        lane_half_width = lane_geom.width * 0.5
        safety_margin = self.lateral_params["safety_margin"]
        vehicle_half_width = self.lateral_params["vehicle_half_width"]

        base_max_offset = max(0.0, lane_half_width - vehicle_half_width - safety_margin)
        safe_min = -base_max_offset
        safe_max = base_max_offset

        if source_lane_id == 1:
            safe_max = min(safe_max, 0.6)
        elif source_lane_id == 3:
            safe_min = max(safe_min, -0.6)

        lateral_offset = 0.0
        lane_range = self.lane_x_ranges.get(source_lane_id)
        if lane_range is not None and lane_range["max"] > lane_range["min"]:
            x_deviation = local_x_m - lane_range["center"]
            x_normalized = x_deviation / (lane_range["max"] - lane_range["min"]) * 2.0
            max_safe_offset = min(abs(safe_min), abs(safe_max))
            lateral_offset = x_normalized * max_safe_offset
            lateral_offset = max(safe_min, min(safe_max, lateral_offset))
        else:
            ngsim_x_min, ngsim_x_max = self.transform_params["ngsim_x_range"]
            if ngsim_x_max > ngsim_x_min:
                x_normalized = (local_x_m - (ngsim_x_min + ngsim_x_max) * 0.5) / (ngsim_x_max - ngsim_x_min)
                max_safe_offset = min(abs(safe_min), abs(safe_max))
                lateral_offset = x_normalized * max_safe_offset
                lateral_offset = max(safe_min, min(safe_max, lateral_offset))

        y = lane_geom.y_center + lateral_offset

        lower_bound = edge_geom.min_y + self.lateral_params["vehicle_half_width"] + safety_margin
        upper_bound = edge_geom.max_y - self.lateral_params["vehicle_half_width"] - safety_margin
        y = max(lower_bound, min(upper_bound, y))

        return {
            "edge_id": edge_id,
            "lane_index": lane_index,
            "lane_id": lane_geom.lane_id,
            "x": x,
            "y": y,
        }

    def calculate_vehicle_angle(
        self,
        vehicle_id: int,
        current_pos: Tuple[float, float],
        prev_pos: Optional[Tuple[float, float]] = None,
    ) -> float:
        if vehicle_id not in self.vehicle_position_history:
            self.vehicle_position_history[vehicle_id] = []

        history = self.vehicle_position_history[vehicle_id]
        history.append(current_pos)
        if len(history) > 3:
            history.pop(0)

        if prev_pos is None:
            if len(history) < 2:
                return 90.0
            prev_pos = history[-2]

        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        if math.hypot(dx, dy) < 1e-3:
            previous_angle = self.active_vehicles.get(vehicle_id, {}).get("angle", 90.0)
            return float(previous_angle)

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        return self.normalize_angle(90.0 - angle_deg)

    def determine_control_mode(self, vehicle_id: int) -> str:
        vehicle_info = self.active_vehicles[vehicle_id]
        frames_since_added = self.current_frame - int(vehicle_info["added_frame"])
        ngsim_frames = self.hybrid_control_params["ngsim_guidance_frames"]
        transition_frames = self.hybrid_control_params["transition_frames"]

        if not self.hybrid_control_params["enable_sumo_takeover"]:
            return "ngsim"
        if frames_since_added < ngsim_frames:
            return "ngsim"
        if frames_since_added < ngsim_frames + transition_frames:
            return "hybrid"
        return "sumo"

    def get_control_influence_factor(self, vehicle_id: int, mode: str) -> Tuple[float, float]:
        if mode == "ngsim":
            return 1.0, 0.0
        if mode == "sumo":
            return 0.0, 1.0

        vehicle_info = self.active_vehicles[vehicle_id]
        frames_since_added = self.current_frame - int(vehicle_info["added_frame"])
        ngsim_frames = self.hybrid_control_params["ngsim_guidance_frames"]
        transition_frames = self.hybrid_control_params["transition_frames"]
        progress = (frames_since_added - ngsim_frames) / max(1, transition_frames)
        progress = max(0.0, min(1.0, progress))
        return 1.0 - progress, progress

    def check_safety_distance(self, vehicle_id: int, target_speed: float) -> Tuple[float, bool]:
        sumo_id = str(self.active_vehicles[vehicle_id]["sumo_id"])
        try:
            leader_info = traci.vehicle.getLeader(sumo_id)
        except traci.TraCIException:
            return target_speed, False

        if leader_info is None:
            return target_speed, False

        leader_id, distance = leader_info
        current_speed = traci.vehicle.getSpeed(sumo_id)
        min_gap = self.safety_params["min_gap"]
        tau = self.sumo_models["tau"]
        safe_gap = min_gap + tau * current_speed

        if distance < self.safety_params["collision_threshold"]:
            emergency_speed = max(
                0.0, current_speed - self.safety_params["max_decel_emergency"] * 0.1
            )
            self.stats["collision_avoidances"] += 1
            self.stats["emergency_brakes"] += 1
            return emergency_speed, True

        if distance < safe_gap:
            try:
                leader_speed = traci.vehicle.getSpeed(leader_id)
            except traci.TraCIException:
                leader_speed = target_speed
            v_safe = max(0.0, min(target_speed, leader_speed + (distance - min_gap) / tau))
            adapted = current_speed + self.safety_params["speed_adaptation_rate"] * (v_safe - current_speed)
            return adapted, False

        return target_speed, False

    def apply_intelligent_control(
        self,
        vehicle_id: int,
        ngsim_speed: float,
    ) -> Tuple[str, float, bool]:
        mode = self.determine_control_mode(vehicle_id)
        previous_mode = self.vehicle_control_modes.get(vehicle_id)
        if previous_mode != mode:
            self.vehicle_control_modes[vehicle_id] = mode
            self.control_mode_switch_frame[vehicle_id] = self.current_frame

        self.stats[f"mode_{mode}"] += 1

        safe_speed, emergency = self.check_safety_distance(vehicle_id, ngsim_speed)
        if emergency:
            self.safety_overrides[vehicle_id] = True
            self.stats["safety_overrides_count"] += 1
            return "emergency", safe_speed, True

        if mode == "ngsim":
            return mode, safe_speed, False

        if mode == "hybrid":
            ngsim_factor, sumo_factor = self.get_control_influence_factor(vehicle_id, mode)
            sumo_id = str(self.active_vehicles[vehicle_id]["sumo_id"])
            sumo_speed = traci.vehicle.getSpeed(sumo_id)
            blended = ngsim_factor * ngsim_speed + sumo_factor * sumo_speed
            return mode, min(safe_speed, blended), False

        return mode, safe_speed, False

    def setup_intelligent_vehicle(self, sumo_id: str) -> None:
        traci.vehicle.setTau(sumo_id, self.sumo_models["tau"])
        traci.vehicle.setImperfection(sumo_id, self.sumo_models["sigma"])
        traci.vehicle.setMinGap(sumo_id, self.safety_params["min_gap"])
        traci.vehicle.setMaxSpeed(sumo_id, 25.0)
        traci.vehicle.setAccel(sumo_id, 2.6)
        traci.vehicle.setDecel(sumo_id, self.safety_params["max_decel_comfort"])
        traci.vehicle.setEmergencyDecel(sumo_id, self.safety_params["max_decel_emergency"])
        traci.vehicle.setSpeedMode(sumo_id, 0)
        traci.vehicle.setLaneChangeMode(sumo_id, 0)

    def create_route_file(self, route_file: str) -> None:
        routes = ET.Element("routes")
        routes.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        routes.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

        vtype = ET.SubElement(routes, "vType")
        vtype.set("id", "ngsim_vehicle")
        vtype.set("vClass", "passenger")
        vtype.set("length", "5.0")
        vtype.set("width", str(self.lateral_params["vehicle_half_width"] * 2))
        vtype.set("height", "1.5")
        vtype.set("minGap", str(self.safety_params["min_gap"]))
        vtype.set("maxSpeed", "25.0")
        vtype.set("accel", "2.6")
        vtype.set("decel", str(self.safety_params["max_decel_comfort"]))
        vtype.set("emergencyDecel", str(self.safety_params["max_decel_emergency"]))
        vtype.set("sigma", str(self.sumo_models["sigma"]))
        vtype.set("tau", str(self.sumo_models["tau"]))
        vtype.set("carFollowModel", self.sumo_models["carFollowModel"])
        vtype.set("laneChangeModel", self.sumo_models["laneChangeModel"])
        vtype.set("guiShape", "passenger")

        route = ET.SubElement(routes, "route")
        route.set("id", "carla_main_route")
        route.set("edges", " ".join(self.route_edges))

        rough_string = ET.tostring(routes, "unicode")
        reparsed = minidom.parseString(rough_string)
        with open(route_file, "w", encoding="utf-8") as handle:
            handle.write(reparsed.toprettyxml(indent="    "))

    def create_config_file(self, net_file: str, route_file: str, config_file: str) -> None:
        frame_count = self.max_frame - self.current_frame + 1
        end_seconds = max(10.0, frame_count * 0.1 + 5.0)

        config = ET.Element("configuration")
        config.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        config.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")

        input_elem = ET.SubElement(config, "input")
        ET.SubElement(input_elem, "net-file").set("value", net_file)
        ET.SubElement(input_elem, "route-files").set("value", route_file)

        time_elem = ET.SubElement(config, "time")
        ET.SubElement(time_elem, "begin").set("value", "0")
        ET.SubElement(time_elem, "end").set("value", f"{end_seconds:.1f}")
        ET.SubElement(time_elem, "step-length").set("value", "0.1")

        processing_elem = ET.SubElement(config, "processing")
        ET.SubElement(processing_elem, "collision.action").set("value", "warn")
        ET.SubElement(processing_elem, "collision.check-junctions").set("value", "true")

        rough_string = ET.tostring(config, "unicode")
        reparsed = minidom.parseString(rough_string)
        with open(config_file, "w", encoding="utf-8") as handle:
            handle.write(reparsed.toprettyxml(indent="    "))

    def move_vehicle_to_mapped_position(
        self,
        sumo_vehicle_id: str,
        mapped: Dict[str, object],
        pos: Tuple[float, float],
        angle: float,
    ) -> None:
        primary_edge = str(mapped["edge_id"])
        fallback_edge = self.route_edges[0]

        strategies = [
            lambda: traci.vehicle.moveToXY(
                vehID=sumo_vehicle_id,
                edgeID=primary_edge,
                lane=-1,
                x=pos[0],
                y=pos[1],
                angle=angle,
                keepRoute=1,
            ),
            lambda: traci.vehicle.moveToXY(
                vehID=sumo_vehicle_id,
                edgeID=fallback_edge,
                lane=-1,
                x=pos[0],
                y=pos[1],
                angle=angle,
                keepRoute=1,
            ),
            lambda: traci.vehicle.moveToXY(
                vehID=sumo_vehicle_id,
                edgeID=primary_edge,
                lane=-1,
                x=pos[0],
                y=pos[1],
                keepRoute=1,
            ),
            lambda: traci.vehicle.moveToXY(
                vehID=sumo_vehicle_id,
                edgeID=primary_edge,
                lane=int(mapped["lane_index"]),
                x=pos[0],
                y=pos[1],
                angle=angle,
                keepRoute=1,
            ),
        ]

        last_error = None
        for strategy in strategies:
            try:
                strategy()
                return
            except Exception as exc:
                last_error = exc

        raise last_error

    def add_vehicle(self, vehicle_id: int, frame_data: pd.Series) -> bool:
        sumo_vehicle_id = f"ngsim_{vehicle_id}"
        mapped = self.map_coordinates(frame_data["Local_Y"], frame_data["Local_X"], int(frame_data["Lane_ID"]))
        initial_speed = float(frame_data["v_Vel"]) * self.feet_to_meters
        pos = (float(mapped["x"]), float(mapped["y"]))
        angle = self.calculate_vehicle_angle(vehicle_id, pos)

        if sumo_vehicle_id not in traci.vehicle.getIDList():
            traci.vehicle.add(sumo_vehicle_id, "carla_main_route", typeID="ngsim_vehicle")
            self.setup_intelligent_vehicle(sumo_vehicle_id)

        self.move_vehicle_to_mapped_position(sumo_vehicle_id, mapped, pos, angle)
        traci.vehicle.setSpeed(sumo_vehicle_id, initial_speed)

        self.active_vehicles[vehicle_id] = {
            "sumo_id": sumo_vehicle_id,
            "prev_pos": pos,
            "prev_speed": initial_speed,
            "added_frame": self.current_frame,
            "last_lane_id": int(frame_data["Lane_ID"]),
            "angle": angle,
        }
        self.vehicle_control_modes[vehicle_id] = "ngsim"
        self.safety_overrides[vehicle_id] = False
        self.stats["vehicles_added"] += 1
        return True

    def update_vehicle(self, vehicle_id: int, frame_data: pd.Series) -> bool:
        vehicle_info = self.active_vehicles.get(vehicle_id)
        if not vehicle_info:
            return False

        sumo_id = str(vehicle_info["sumo_id"])
        if sumo_id not in traci.vehicle.getIDList():
            self.active_vehicles.pop(vehicle_id, None)
            return False

        mapped = self.map_coordinates(frame_data["Local_Y"], frame_data["Local_X"], int(frame_data["Lane_ID"]))
        ngsim_speed = float(frame_data["v_Vel"]) * self.feet_to_meters
        target_pos = (float(mapped["x"]), float(mapped["y"]))
        control_mode, final_speed, emergency = self.apply_intelligent_control(vehicle_id, ngsim_speed)

        if control_mode == "sumo" and not emergency:
            traci.vehicle.setSpeed(sumo_id, -1)
        else:
            target_angle = self.calculate_vehicle_angle(vehicle_id, target_pos, vehicle_info.get("prev_pos"))
            self.move_vehicle_to_mapped_position(sumo_id, mapped, target_pos, target_angle)
            if control_mode == "hybrid":
                traci.vehicle.slowDown(sumo_id, final_speed, 0.5)
            else:
                traci.vehicle.setSpeed(sumo_id, final_speed)
            vehicle_info["prev_pos"] = target_pos
            vehicle_info["angle"] = target_angle

        if control_mode == "sumo" and not emergency:
            vehicle_info["prev_pos"] = traci.vehicle.getPosition(sumo_id)
            vehicle_info["angle"] = traci.vehicle.getAngle(sumo_id)

        vehicle_info["prev_speed"] = final_speed
        vehicle_info["last_lane_id"] = int(frame_data["Lane_ID"])
        self.stats["total_updates"] += 1
        return True

    def remove_vehicle(self, vehicle_id: int) -> None:
        vehicle_info = self.active_vehicles.pop(vehicle_id, None)
        if not vehicle_info:
            return

        sumo_id = str(vehicle_info["sumo_id"])
        if sumo_id in traci.vehicle.getIDList():
            traci.vehicle.remove(sumo_id)

        self.vehicle_control_modes.pop(vehicle_id, None)
        self.control_mode_switch_frame.pop(vehicle_id, None)
        self.safety_overrides.pop(vehicle_id, None)
        self.vehicle_position_history.pop(vehicle_id, None)
        self.stats["vehicles_removed"] += 1

    def run_simulation(self, csv_file: str, net_file: str) -> None:
        self.parse_network_geometry(net_file)
        df = self.load_ngsim_data(csv_file)

        route_file = "carla_ngsim_edge4039_rewrite.rou.xml"
        config_file = "carla_ngsim_edge4039_rewrite.sumocfg"
        self.create_route_file(route_file)
        self.create_config_file(net_file, route_file, config_file)

        sumo_binary = self.resolve_sumo_binary()
        sumo_cmd = [
            sumo_binary,
            "-c",
            os.path.abspath(config_file),
            "--start",
            "--collision.action",
            "warn",
            "--step-length",
            "0.1",
        ]

        try:
            traci.start(sumo_cmd, numRetries=10)
            traci.simulationStep()
            try:
                traci.gui.setDelay(self.gui_params["view_id"], self.gui_params["delay_ms"])
                traci.gui.setSchema(self.gui_params["view_id"], self.gui_params["schema"])
            except Exception:
                pass

            total_frames = self.max_frame - self.current_frame + 1
            start_time = time.time()

            for frame_id in range(self.current_frame, self.max_frame + 1):
                self.current_frame = frame_id
                current_vehicles = set(self.frame_vehicles.get(frame_id, []))
                active_vehicles = set(self.active_vehicles.keys())

                for vehicle_id in current_vehicles - active_vehicles:
                    trajectory = self.vehicle_trajectories.get(vehicle_id)
                    if trajectory is None:
                        continue
                    frame_rows = trajectory[trajectory["Frame_ID"] == frame_id]
                    if not frame_rows.empty:
                        self.add_vehicle(vehicle_id, frame_rows.iloc[0])

                for vehicle_id in active_vehicles - current_vehicles:
                    self.remove_vehicle(vehicle_id)

                for vehicle_id in list(self.active_vehicles.keys()):
                    trajectory = self.vehicle_trajectories.get(vehicle_id)
                    if trajectory is None:
                        continue
                    frame_rows = trajectory[trajectory["Frame_ID"] == frame_id]
                    if not frame_rows.empty:
                        self.update_vehicle(vehicle_id, frame_rows.iloc[0])

                if frame_id % 50 == 0:
                    progress = (frame_id - int(df["Frame_ID"].min())) / max(1, total_frames) * 100.0
                    print(
                        f"frame={frame_id} progress={progress:.1f}% active={len(self.active_vehicles)} "
                        f"updates={self.stats['total_updates']}"
                    )

                traci.simulationStep()

            elapsed = time.time() - start_time
            print("Simulation finished.")
            print(f"Vehicles processed: {self.total_vehicles}")
            print(f"Frames: {total_frames}")
            print(f"Elapsed: {elapsed:.2f}s")
            print(f"Average FPS: {total_frames / max(elapsed, 1e-6):.2f}")
            print(f"Mode stats: ngsim={self.stats['mode_ngsim']}, hybrid={self.stats['mode_hybrid']}, sumo={self.stats['mode_sumo']}")
            print(f"Safety: emergency={self.stats['emergency_brakes']}, avoided={self.stats['collision_avoidances']}")
        finally:
            try:
                traci.close()
            except Exception:
                pass


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    csv_file = args[0] if len(args) >= 1 else "ngsim_vehicle_final_145.csv"
    net_file = args[1] if len(args) >= 2 else "carla_town04_official.net.xml"

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Missing NGSIM CSV: {csv_file}")
    if not os.path.exists(net_file):
        raise FileNotFoundError(f"Missing SUMO net file: {net_file}")

    controller = CarlaTown4Edge4039RewriteController()
    controller.run_simulation(csv_file, net_file)


if __name__ == "__main__":
    main()
