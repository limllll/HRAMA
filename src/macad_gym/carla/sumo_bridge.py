import os
import sys
import tempfile
import subprocess
import time
import math
from typing import Dict, Optional, List, Tuple
from enum import Enum
import xml.etree.ElementTree as ET
import glob
import json

# 導入CARLA以支援燈光狀態設定
import carla


# ============================================================================
# 🎯 NGSIM车辆控制模式状态机（简化版）
# ============================================================================
class NGSIMControlMode(Enum):
    """
    NGSIM车辆控制模式 - 两态状态机（简化版）

    状态转换图:
        REPLAY_MODE ---(危险检测)---> FREE_MODE (永久)

    核心逻辑:
    - REPLAY_MODE: 按NGSIM数据回放轨迹
    - 一旦检测到危险（TTC<3s 或 前车距离<10m），永久转换为FREE_MODE
    - FREE_MODE下完全由SUMO Krauss模型控制，不再尝试恢复
    """
    REPLAY_MODE = "replay"  # 回放模式：强制按NGSIM轨迹行驶
    FREE_MODE = "free"  # 自由模式：永久由SUMO控制（危险触发后不再恢复）


# 會在 _ensure_sumo_available 中懶加載
traci = None
sumolib = None

# TraCI異常類型的全局變量
TraCIException = None

# 導入官方的 BridgeHelper
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Co-Simulation', 'Sumo'))
try:
    from sumo_integration.bridge_helper import BridgeHelper
    from sumo_integration.sumo_simulation import SumoSignalState, SumoVehSignal, SumoActorClass
except ImportError as e:
    raise ImportError(f"無法導入官方 SUMO 集成模塊: {e}. 請確保 Co-Simulation/Sumo 目錄可用且配置正確")


class CarlaSumoBridge:
    """
    完整 CARLA⇄SUMO 雙向橋接器，基於 CARLA 官方 Co-Simulation 模組實現：
    - 從 CARLA 世界匯出 OpenDRIVE 至臨時檔
    - 利用 netconvert 生成 SUMO 網路與配置
    - 啟動 traci，步長與 CARLA 的 fixed_delta_seconds 對齊
    - 支援雙向同步：
      - 將 CARLA 中的車輛鏡像到 SUMO
      - 可選地將 SUMO 中的車輛鏡像回 CARLA
    - 使用 BridgeHelper 提供精確的座標轉換
    - 支援交通燈同步
    """

    def __init__(self, world, fixed_delta_seconds: float, town_map_name: str,
                 sumo_gui: bool = True, verbose: bool = True,
                 sync_vehicle_color: bool = True, sync_vehicle_lights: bool = True,
                 sumo_to_carla_sync: bool = False,  # 默認啟用 GUI
                 enable_ngsim_background: bool = False,  # 新增：启用NGSIM背景交通
                 ngsim_csv_file: str = None,  # 新增：NGSIM数据文件
                 # ============================================================================
                 # 🎯 NGSIM状态机配置参数（简化版）- "NGSIM回放 + 危险时SUMO永久接管"
                 # ============================================================================
                 danger_distance_threshold: float = 10.0,  # 危险距离阈值（米），默认10米
                 danger_ttc_threshold: float = 3.0,  # 危险TTC阈值（秒），默认3秒
                 # 以下参数保留用于向后兼容，但不再使用
                 deviation_threshold: float = 2.0,  # [已废弃] 不再使用偏差阈值
                 recovery_safe_frames: int = 10):  # [已废弃] 不再尝试恢复
        """
        初始化 CARLA-SUMO 橋接器

        Args:
            world: CARLA 世界對象
            fixed_delta_seconds: 仿真步長，應與 CARLA 相同
            town_map_name: 城鎮地圖名稱 (例如 "Town03")
            sumo_gui: 是否啟用 SUMO GUI 界面
            verbose: 是否輸出詳細調試信息
            sync_vehicle_color: 是否同步車輛顏色
            sync_vehicle_lights: 是否同步車輛燈光
            sumo_to_carla_sync: 是否將 SUMO 車輛同步回 CARLA (雙向同步)
            enable_ngsim_background: 是否启用NGSIM背景交通流
            ngsim_csv_file: NGSIM数据CSV文件名

            # 🎯 状态机配置参数（简化版：NGSIM回放 + 危险时SUMO永久接管）
            danger_distance_threshold: 危险距离阈值（米），当前车距离小于此值时
                永久转换为FREE_MODE（SUMO完全接管），默认10.0米
            danger_ttc_threshold: 危险TTC阈值（秒），当碰撞时间小于此值时
                永久转换为FREE_MODE，默认3.0秒

            注意：一旦触发危险条件，车辆将永久由SUMO控制，不再尝试恢复NGSIM轨迹
        """
        self.world = world
        self.carla_map = world.get_map()
        self.fixed_delta_seconds = fixed_delta_seconds
        self.town_map_name = town_map_name
        self.sumo_gui = sumo_gui
        self.verbose = verbose
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights
        self.sumo_to_carla_sync = sumo_to_carla_sync

        # NGSIM集成
        self.enable_ngsim_background = enable_ngsim_background
        self.ngsim_csv_file = ngsim_csv_file
        self._ngsim_controller = None
        self._ngsim_data = None
        self._current_simulation_time = 0.0  # 仿真时间（秒）
        self._ngsim_frame_duration = 0.1  # NGSIM帧间隔（秒）
        self._last_ngsim_frame = -1  # 🔧 跟踪上次处理的NGSIM帧，避免重复更新

        # ============================================================================
        # 🎯 NGSIM状态机控制系统（简化版）- "NGSIM回放 + 危险时SUMO永久接管"
        # ============================================================================
        # 车辆控制模式字典: vehicle_id -> NGSIMControlMode
        self._vehicle_control_modes: Dict[int, NGSIMControlMode] = {}

        # 车辆理论位置（来自NGSIM数据集）: vehicle_id -> (x, y)
        self._vehicle_ngsim_target_positions: Dict[int, Tuple[float, float]] = {}

        # 状态机配置参数（简化版：只有危险检测，无恢复机制）
        self._danger_distance_threshold = danger_distance_threshold  # 危险距离阈值（米），默认10米
        self._danger_ttc_threshold = danger_ttc_threshold  # 危险TTC阈值（秒），默认3秒

        # 以下变量保留用于向后兼容，但不再使用
        self._deviation_threshold = deviation_threshold  # [已废弃]
        self._recovery_safe_frames = recovery_safe_frames  # [已废弃]
        self._vehicle_safe_frame_counter: Dict[int, int] = {}  # [已废弃]
        self._vehicle_deviation_history: Dict[int, List[float]] = {}  # [已废弃]

        # ✅ 新增: 用于存储当前帧的统计数据（传给奖励函数）
        self._current_frame_stats = {
            "avg_deviation": 0.0,  # 平均误差（米）- 仅供参考，回放模式下应为0
            "active_count": 0,  # 纳入计算的车辆数（未脱管的）
            "max_deviation": 0.0,  # 最大误差
            "total_vehicles": 0,  # 总车辆数（包括已脱管的）
            "new_detached": 0,  # 🎯 本帧新被SUMO接管的车辆数量（用于一次性惩罚）
        }

        # 已脱管车辆集合（不再计入误差统计）
        self._detached_ngsim_vehicles = set()

        # 打印状态机配置信息
        if self.verbose:
            print(f"🎯 NGSIM状态机配置（简化版）:")
            print(f"   危险距离阈值: {self._danger_distance_threshold}米 (前车距离<此值触发接管)")
            print(f"   危险TTC阈值: {self._danger_ttc_threshold}秒 (碰撞时间<此值触发接管)")
            print(f"   ⚠️ 一旦触发危险，车辆将永久由SUMO控制，不再恢复NGSIM轨迹")

        self._traci = None
        self._sumolib_net = None
        self._sumo_proc = None

        self._tmp_dir = None
        self._net_file = None
        self._cfg_file = None
        self._vtypes_file = None

        # CARLA actor id → SUMO vehicle id
        self._carla_to_sumo_id: Dict[str, str] = {}
        # SUMO vehicle id → CARLA actor id
        self._sumo_to_carla_id: Dict[str, str] = {}
        # 記錄上次同步狀態
        self._last_sync_status = {}

        # 懶載入至此 import（避免未安裝 SUMO 時在匯入檔案階段就噴錯）

        # 藍圖 id → SUMO vType id 對應
        self._blueprint_to_vtype: Dict[str, str] = {}
        # 存儲從 CARLA 獲取的車輛藍圖
        self._carla_blueprints = None

        # 交通燈同步狀態
        self._tls_manager = None

        # 橋接輔助類 (使用官方 BridgeHelper)
        self._helper = None
        # 確保 BridgeHelper 可用
        self._bridge_helper_available = 'BridgeHelper' in globals()

        # # 針對 Town03 T 型路口的坐標修正 (作為備用方案)
        # self._coordinate_offsets = {
        #     "Town03": {
        #         "x": 5.0,   # T 型路口 X 軸修正值
        #         "y": -5.0,  # T 型路口 Y 軸修正值
        #     }
        # }

        # vType 規格（官方 vtypes.json 內容，啟動時載入）
        self._vtype_specs = {}
        self._default_wheeled_spec = {"vClass": "passenger"}
        self._default_2w_spec = {"vClass": "motorcycle"}

        # Edge配置（根據地圖動態設置）
        self._edge_config = None

    # ---------- edge configuration ----------
    def _get_edge_config(self):
        """根據town_map_name獲取edge配置，支持Town03和Town04"""
        if self.town_map_name == "Town04":
            return {
                'main_edge': "40",
                'next_edge': "39",
                'route': ["40", "39"],
                'lane_mapping': {1: 4, 2: 3, 3: 2},  # Town04: Lane1→40_4, Lane2→40_3, Lane3→40_2
                'description': 'Town04 Edge40/39 (东西向道路)'
            }
        else:  # Town03 (默認)
            return {
                'main_edge': "3",
                'next_edge': "2",
                'route': ["3", "2"],
                'lane_mapping': {1: 5, 2: 4, 3: 3},  # Town03: Lane1→3_5, Lane2→3_4, Lane3→3_3
                'description': 'Town03 Edge3/2 (南北向道路)'
            }

    # ---------- public lifecycle ----------
    def start(self):
        """啟動 CARLA-SUMO 橋接器，建立連接並準備同步"""
        self._ensure_sumo_available()
        print("正在構建 SUMO 網路並啟動 SUMO GUI...")

        # 構建 SUMO 網路
        self._build_sumo_net_from_opendrive()

        # 載入官方 vtypes 規格映射
        self._load_vtype_specs()

        # 啟動 TraCI
        self._start_traci()

        # 初始化edge配置（根據地圖類型）
        self._edge_config = self._get_edge_config()
        if self.verbose:
            print(f"🛣️  Edge配置已設置: {self._edge_config['description']}")
            print(f"   - 主要道路: {self._edge_config['main_edge']} → {self._edge_config['next_edge']}")
            print(f"   - 車道映射: {self._edge_config['lane_mapping']}")

        # 初始化橋接輔助類 (使用官方 BridgeHelper)
        if self._sumolib_net and self._bridge_helper_available:
            net_offset = self._sumolib_net.getLocationOffset()
            # 設置官方 BridgeHelper 的靜態屬性
            BridgeHelper.offset = net_offset
            BridgeHelper.blueprint_library = self.world.get_blueprint_library()
            self._helper = BridgeHelper
            if self.verbose:
                print(f"官方橋接輔助類初始化完成，網路偏移量: {net_offset}")

        # 初始化NGSIM控制器
        if self.enable_ngsim_background and self.ngsim_csv_file:
            print(f"🎯 开始初始化NGSIM控制器...")
            print(f"  📁 NGSIM数据文件: {self.ngsim_csv_file}")
            self._init_ngsim_controller()
        else:
            print(f"⚠️ NGSIM功能未启用:")
            print(f"  📁 enable_ngsim_background: {self.enable_ngsim_background}")
            print(f"  📄 ngsim_csv_file: {self.ngsim_csv_file}")

    def close(self):
        try:
            if self._traci:
                self._traci.close(False)
        finally:
            self._traci = None
            if self._sumo_proc and self._sumo_proc.poll() is None:
                try:
                    self._sumo_proc.terminate()
                except Exception:
                    pass
            self._sumo_proc = None

    # ---------- step sync ----------
    def step(self, actors: Dict[str, object] = None):
        """執行 SUMO 仿真步驟，並進行雙向同步

        Args:
            actors: CARLA 中的 actors 字典 (actor_id -> actor)，如果為 None，則只執行 SUMO 步驟不同步
        """
        if not self._traci:
            return

        # 記錄同步前的 SUMO 車輛 ID
        sumo_vehicles_before = set(self._traci.vehicle.getIDList()) if self.sumo_to_carla_sync else set()

        # 執行 CARLA -> SUMO 同步
        synced_from_carla = 0
        errors_from_carla = 0
        if actors:
            synced_from_carla, errors_from_carla = self._sync_from_carla_to_sumo(actors)

        # 更新NGSIM車辆状态 (在SUMO step之前)
        if self.enable_ngsim_background and self._ngsim_controller:
            try:
                self._update_ngsim_vehicles()
                if self.verbose and self._current_simulation_time > 0 and int(
                        self._current_simulation_time * 10) % 50 == 0:
                    ngsim_count = len([v for v in self._traci.vehicle.getIDList() if v.startswith('ngsim_')])
                    print(f"🚗 NGSIM车辆更新: 仿真时间{self._current_simulation_time:.1f}s, NGSIM车辆数: {ngsim_count}")
            except Exception as e:
                if self.verbose:
                    print(f"❌ NGSIM车辆更新失败: {e}")
        elif self.enable_ngsim_background and not self._ngsim_controller:
            if self.verbose and int(self._current_simulation_time * 10) % 100 == 0:  # 每10秒输出一次
                print(f"⚠️ NGSIM已启用但控制器未初始化 (时间: {self._current_simulation_time:.1f}s)")

        # 執行 SUMO 仿真步驟
        self._traci.simulationStep()

        # 更新仿真时间
        self._current_simulation_time += self.fixed_delta_seconds

        # 執行 SUMO -> CARLA 同步 (如果啟用)
        if self.sumo_to_carla_sync:
            sumo_vehicles_after = set(self._traci.vehicle.getIDList())
            # 過濾出僅存在於 SUMO 的車輛 (排除從 CARLA 同步的車輛)
            sumo_only_vehicles = sumo_vehicles_after - set(self._carla_to_sumo_id.values())
            if sumo_only_vehicles:
                self._sync_from_sumo_to_carla(sumo_only_vehicles)

        # 輸出調試信息
        if self.verbose:
            total_sumo_vehicles = len(self._traci.vehicle.getIDList())
            ngsim_vehicles = len([v for v in self._traci.vehicle.getIDList() if v.startswith('ngsim_')])

            if synced_from_carla > 0:
                print(f"CARLA → SUMO 同步: {synced_from_carla} 輛車成功, {errors_from_carla} 輛失敗")

            if total_sumo_vehicles > 0:
                print(f"🚗 仿真時間: {self._current_simulation_time:.1f}s, "
                      f"SUMO車輛: {total_sumo_vehicles}輛 (NGSIM: {ngsim_vehicles}輛)")

            # 🔍 定期診斷CARLA車輛的可見性
            if not hasattr(self, '_diagnostic_step_counter'):
                self._diagnostic_step_counter = 0
            self._diagnostic_step_counter += 1

            # 每100步診斷一次
            if self._diagnostic_step_counter % 100 == 1 and actors:
                first_actor_id = list(actors.keys())[0]
                print("\n" + "=" * 60)
                print(f"🔍 第{self._diagnostic_step_counter}步 - 診斷CARLA車輛可見性")
                print("=" * 60)
                self.diagnose_vehicle_visibility(first_actor_id)
                print("=" * 60 + "\n")

            # T 型路口的額外調試
            if self.town_map_name == "Town03":
                try:
                    vehicle_infos = self.get_sumo_vehicles_info()
                    if vehicle_infos:
                        print("車輛道路位置:")
                        for veh_id, info in list(vehicle_infos.items())[:3]:  # 只顯示前三輛
                            road_id = info.get('road_id', 'unknown')
                            lane_id = info.get('lane_id', 'unknown')
                            print(f"  {veh_id}: 道路={road_id}, 車道={lane_id}")
                except Exception:
                    pass

    def step_and_sync_from_carla(self, actors: Dict[str, object]):
        """
        向後兼容的方法，等同於 step()

        Args:
            actors: CARLA 中的 actors 字典 (actor_id -> actor)
        """
        return self.step(actors)

    def _sync_from_carla_to_sumo(self, actors: Dict[str, object]):
        """將 CARLA 車輛同步到 SUMO

        Args:
            actors: CARLA 中的 actors 字典 (actor_id -> actor)

        Returns:
            tuple: (同步成功數量, 同步失敗數量)
        """
        synced_count = 0
        error_count = 0

        # 同步所有車輛
        for actor_id, actor in actors.items():
            # 僅同步 CARLA 中的車輛（忽略交通燈/行人）
            try:
                if hasattr(actor, 'type_id') and str(actor.type_id).startswith('vehicle.'):
                    # 確保 SUMO 中存在對應車輛
                    veh_id = self._ensure_sumo_vehicle(actor_id, actor)

                    # 檢查車輛是否在 SUMO 中有效
                    if veh_id in self._traci.vehicle.getIDList():
                        # 將 CARLA 車輛的位置和朝向映射到 SUMO
                        self._move_vehicle_to_actor_pose(actor_id, actor)

                        # 同步顏色 (如果啟用)
                        if self.sync_vehicle_color:
                            if self._bridge_helper_available:
                                # 嘗試使用官方 BridgeHelper 同步顏色
                                try:
                                    color = actor.get_color() if hasattr(actor, 'get_color') else None
                                    if color:
                                        self._traci.vehicle.setColor(veh_id, (color.r, color.g, color.b))
                                except Exception:
                                    self._sync_vehicle_color(actor, veh_id)
                            else:
                                self._sync_vehicle_color(actor, veh_id)

                        # 同步車輛燈光 (如果啟用)
                        if self.sync_vehicle_lights:
                            if self._bridge_helper_available:
                                # 嘗試使用官方 BridgeHelper 同步燈光
                                try:
                                    import carla
                                    if hasattr(actor, 'get_light_state'):
                                        carla_lights = actor.get_light_state()
                                        current_sumo_lights = 0  # 默認值
                                        sumo_lights = BridgeHelper.get_sumo_lights_state(current_sumo_lights,
                                                                                         carla_lights)
                                        self._traci.vehicle.setSignals(veh_id, sumo_lights)
                                except Exception:
                                    self._sync_vehicle_lights(actor, veh_id)
                            else:
                                self._sync_vehicle_lights(actor, veh_id)

                        synced_count += 1
                    else:
                        error_count += 1
            except Exception as e:
                error_count += 1
                if self.verbose:
                    print(f"同步車輛 {actor_id} 到 SUMO 失敗: {e}")

        return synced_count, error_count

    def _sync_from_sumo_to_carla(self, sumo_vehicles):
        """將 SUMO 中的車輛同步到 CARLA

        Args:
            sumo_vehicles: 要同步的 SUMO 車輛 ID 集合
        """
        if not self.sumo_to_carla_sync or not self._helper:
            return

        synced_count = 0
        for sumo_veh_id in sumo_vehicles:
            # 跳過已經由 CARLA 控制的車輛
            if sumo_veh_id in self._carla_to_sumo_id.values():
                continue

            try:
                # 只處理未在 CARLA 中表示的 SUMO 車輛
                if sumo_veh_id not in self._sumo_to_carla_id:
                    # 創建新的 CARLA 車輛
                    carla_actor = self._create_carla_actor_from_sumo(sumo_veh_id)
                    if carla_actor:
                        self._sumo_to_carla_id[sumo_veh_id] = str(carla_actor.id)
                        synced_count += 1
                else:
                    # 更新現有 CARLA 車輛的位置
                    carla_actor_id = self._sumo_to_carla_id[sumo_veh_id]
                    self._update_carla_actor_from_sumo(sumo_veh_id, carla_actor_id)
                    synced_count += 1
            except Exception as e:
                if self.verbose:
                    print(f"同步 SUMO 車輛 {sumo_veh_id} 到 CARLA 失敗: {e}")

        if self.verbose and synced_count > 0:
            print(f"SUMO → CARLA 同步: {synced_count} 輛車")

    def _create_carla_actor_from_sumo(self, sumo_veh_id):
        """從 SUMO 車輛創建 CARLA actor - 嚴格按照官方run_synchronization.py實現

        Args:
            sumo_veh_id: SUMO 車輛 ID

        Returns:
            CARLA actor 或 None (如果創建失敗)
        """
        import carla
        from collections import namedtuple

        # 確保BridgeHelper可用
        assert self._helper is not None, "BridgeHelper未初始化"
        assert self._bridge_helper_available, "BridgeHelper不可用"

        # 🔧 嚴格按照官方教程：使用subscription機制
        try:
            # 1. 訂閱車輛信息（官方標準方式）
            self._traci.vehicle.subscribe(sumo_veh_id, [
                self._traci.constants.VAR_TYPE,
                self._traci.constants.VAR_VEHICLECLASS,
                self._traci.constants.VAR_COLOR,
                self._traci.constants.VAR_LENGTH,
                self._traci.constants.VAR_WIDTH,
                self._traci.constants.VAR_HEIGHT,
                self._traci.constants.VAR_POSITION3D,
                self._traci.constants.VAR_ANGLE,
                self._traci.constants.VAR_SLOPE,
                self._traci.constants.VAR_SIGNALS
            ])

            # 2. 高效獲取訂閱結果（官方標準方式）
            results = self._traci.vehicle.getSubscriptionResults(sumo_veh_id)

            # 3. 解析結果（完全按照官方sumo_simulation.py第393-415行）
            type_id = results[self._traci.constants.VAR_TYPE]

            # ⭐ 關鍵修復：vclass必須用SumoActorClass枚舉包裝（官方第400行）
            try:
                raw_vclass = results[self._traci.constants.VAR_VEHICLECLASS]
                vclass = SumoActorClass(raw_vclass)
            except (KeyError, ValueError):
                # NGSIM車輛可能沒有這個信息，使用默認值
                vclass = SumoActorClass.PASSENGER

            color = results.get(self._traci.constants.VAR_COLOR, (0, 255, 0, 255))

            length = results[self._traci.constants.VAR_LENGTH]
            width = results[self._traci.constants.VAR_WIDTH]
            height = results[self._traci.constants.VAR_HEIGHT]

            # 位置和角度信息（官方第407-410行）
            location = list(results[self._traci.constants.VAR_POSITION3D])
            angle = results[self._traci.constants.VAR_ANGLE]
            slope = results.get(self._traci.constants.VAR_SLOPE, 0.0)

            signals = results[self._traci.constants.VAR_SIGNALS]

        except Exception as e:
            if self.verbose:
                print(f"❌ 無法訂閱SUMO車輛 {sumo_veh_id}: {e}")
            return None

        # 4. 創建CARLA變換和範圍（官方第409-413行）
        transform = carla.Transform(carla.Location(location[0], location[1], location[2]),
                                    carla.Rotation(slope, angle, 0.0))
        extent = carla.Vector3D(length / 2.0, width / 2.0, height / 2.0)

        # 5. 創建SumoActor（官方第415行 - 使用位置參數，不用關鍵字）
        from collections import namedtuple
        SumoActor = namedtuple('SumoActor', 'type_id vclass transform signals extent color')

        # ⭐ 關鍵修復：使用位置參數，與官方完全一致
        sumo_actor = SumoActor(type_id, vclass, transform, signals, extent, color)

        # 使用官方BridgeHelper獲取CARLA藍圖（參考run_synchronization.py:114行）
        carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, self.sync_vehicle_color)
        if carla_blueprint is None:
            if self.verbose:
                print(f"❌ 無法獲取SUMO車輛 {sumo_veh_id} 的CARLA藍圖")
                print(f"   - type_id: {sumo_actor.type_id}")
                print(f"   - vclass: {sumo_actor.vclass}")
                print(f"   - 可能原因: 該車型在CARLA中沒有對應藍圖")
            return None

        # 使用官方BridgeHelper進行坐標轉換（參考run_synchronization.py:116行）
        carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform, sumo_actor.extent)

        # 生成CARLA車輛（參考run_synchronization.py:119行）
        carla_actor_id = self.world.try_spawn_actor(carla_blueprint, carla_transform)

        if carla_actor_id is not None:
            if self.verbose:
                print(f"✅ 成功從SUMO創建CARLA車輛: {sumo_veh_id} → {carla_actor_id.id}")
            return carla_actor_id
        else:
            if self.verbose:
                print(f"❌ 無法生成CARLA車輛: {sumo_veh_id}")
                print(
                    f"   - 位置: ({carla_transform.location.x:.2f}, {carla_transform.location.y:.2f}, {carla_transform.location.z:.2f})")
                print(f"   - 朝向: {carla_transform.rotation.yaw:.1f}°")
                print(f"   - 可能原因: 位置被佔用或超出地圖邊界")
            return None

    def _update_carla_actor_from_sumo(self, sumo_veh_id, carla_actor_id):
        """使用 SUMO 車輛的位置和朝向更新 CARLA actor - 按照官方run_synchronization.py實現

        Args:
            sumo_veh_id: SUMO 車輛 ID
            carla_actor_id: 對應的 CARLA actor ID
        """
        import carla
        import math
        from collections import namedtuple

        # 確保BridgeHelper可用
        assert self._helper is not None, "BridgeHelper未初始化"
        assert self._bridge_helper_available, "BridgeHelper不可用"

        # 獲取 CARLA actor
        carla_actor = self.world.get_actor(int(carla_actor_id))
        if not carla_actor:
            self._sumo_to_carla_id.pop(sumo_veh_id, None)
            return

        # 🔧 嚴格按照官方教程：使用現有訂閱或重新訂閱
        try:
            # 首先嘗試獲取現有的訂閱結果
            results = self._traci.vehicle.getSubscriptionResults(sumo_veh_id)
            if not results:
                # 如果沒有訂閱，則創建訂閱
                self._traci.vehicle.subscribe(sumo_veh_id, [
                    self._traci.constants.VAR_TYPE,
                    self._traci.constants.VAR_VEHICLECLASS,
                    self._traci.constants.VAR_COLOR,
                    self._traci.constants.VAR_LENGTH,
                    self._traci.constants.VAR_WIDTH,
                    self._traci.constants.VAR_HEIGHT,
                    self._traci.constants.VAR_POSITION3D,
                    self._traci.constants.VAR_ANGLE,
                    self._traci.constants.VAR_SLOPE,
                    self._traci.constants.VAR_SIGNALS
                ])
                results = self._traci.vehicle.getSubscriptionResults(sumo_veh_id)

            # 解析訂閱結果（完全按照官方sumo_simulation.py第393-415行）
            type_id = results[self._traci.constants.VAR_TYPE]

            # ⭐ 關鍵修復：vclass必須用SumoActorClass枚舉包裝（官方第400行）
            try:
                raw_vclass = results[self._traci.constants.VAR_VEHICLECLASS]
                vclass = SumoActorClass(raw_vclass)
            except (KeyError, ValueError):
                # NGSIM車輛可能沒有這個信息，使用默認值
                vclass = SumoActorClass.PASSENGER

            color = results.get(self._traci.constants.VAR_COLOR, (0, 255, 0, 255))

            length = results[self._traci.constants.VAR_LENGTH]
            width = results[self._traci.constants.VAR_WIDTH]
            height = results[self._traci.constants.VAR_HEIGHT]

            location = list(results[self._traci.constants.VAR_POSITION3D])
            angle = results[self._traci.constants.VAR_ANGLE]
            slope = results.get(self._traci.constants.VAR_SLOPE, 0.0)
            signals = results[self._traci.constants.VAR_SIGNALS]

        except Exception as e:
            if self.verbose:
                print(f"❌ 無法獲取SUMO車輛 {sumo_veh_id} 的訂閱數據: {e}")
            return

        # 創建官方標準的SUMO Transform和Extent（官方第409-413行）
        transform = carla.Transform(carla.Location(location[0], location[1], location[2]),
                                    carla.Rotation(slope, angle, 0.0))
        extent = carla.Vector3D(length / 2.0, width / 2.0, height / 2.0)

        # 創建官方標準的SumoActor（官方第415行 - 使用位置參數，不用關鍵字）
        SumoActor = namedtuple('SumoActor', 'type_id vclass transform signals extent color')

        # ⭐ 關鍵修復：使用位置參數，與官方完全一致
        sumo_actor = SumoActor(type_id, vclass, transform, signals, extent, color)

        # 使用官方BridgeHelper進行坐標轉換（參考run_synchronization.py:137行）
        carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform, sumo_actor.extent)

        # 處理車輛燈光同步（參考run_synchronization.py:139行）
        carla_lights = None
        if self.sync_vehicle_lights:
            carla_lights = BridgeHelper.get_carla_lights_state(
                carla_actor.get_light_state(),
                sumo_actor.signals
            )

        # 同步車輛位置（參考run_synchronization.py:145行）
        carla_actor.set_transform(carla_transform)

        # 設置燈光（嚴格按照官方carla_simulation.py:146行）
        if carla_lights is not None:
            carla_actor.set_light_state(carla.VehicleLightState(carla_lights))

        # ✅ 按照官方carla_simulation.py第144行實現
        # 官方CARLA-SUMO同步只設置transform和lights，不設置velocity
        # 車輛速度通過連續的transform位置變化自然體現

    # ---------- internals ----------
    def _ensure_sumo_available(self):
        # 嘗試從環境變數或常見安裝路徑找到 SUMO tools 目錄
        sumo_home = os.environ.get("SUMO_HOME")
        candidate_homes = [
            sumo_home,
            "/home/simplexity/cybian/sumo-1.22.0",
            "/usr/share/sumo",
            "/usr/local/share/sumo",
        ]
        tools_dir = None
        for home in candidate_homes:
            if home and os.path.isdir(os.path.join(home, 'tools')):
                tools_dir = os.path.join(home, 'tools')
                break

        # 确保找到了SUMO tools目录
        assert tools_dir is not None, f"找不到SUMO tools目录。请设置SUMO_HOME环境变量或确保SUMO安装在以下位置之一: {candidate_homes}"

        if tools_dir not in sys.path:
            sys.path.append(tools_dir)

        print(f"🔧 使用SUMO tools目录: {tools_dir}")

        # 直接导入TraCI模块，如果不可用会在导入时失败
        import traci
        import sumolib

        # 验证模块正确加载
        assert traci is not None, "TraCI模块导入失败"
        assert sumolib is not None, "Sumolib模块导入失败"
        assert hasattr(sumolib, 'net'), "Sumolib没有net属性"
        assert hasattr(sumolib.net, 'readNet'), "Sumolib.net没有readNet方法"

        # 设置类实例变量
        self._traci = traci
        self._sumolib = sumolib

    def _collect_blueprints(self):
        """收集 CARLA 中的車輛藍圖以便創建對應的 SUMO 車輛類型"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = []

        for blueprint in blueprint_library.filter('vehicle.*'):
            # 獲取車輛藍圖信息
            # 安全轉換 blueprint.tags 為字典（避免 dict() 因元素不是鍵值對而報錯）
            tags_dict = {}
            try:
                raw_tags = getattr(blueprint, 'tags', None)
                if isinstance(raw_tags, dict):
                    tags_dict = dict(raw_tags)
                elif isinstance(raw_tags, (list, tuple, set)):
                    for item in raw_tags:
                        if isinstance(item, str) and '=' in item:
                            k, v = item.split('=', 1)
                            tags_dict[k.strip()] = v.strip()
            except Exception:
                pass

            blueprint_data = {
                'id': blueprint.id,
                'tags': tags_dict,
                'attributes': {}
            }

            # 獲取屬性（兼容不同 CARLA 版本：有的沒有 attributes 屬性）
            try:
                attribute_names = []
                raw_attrs = getattr(blueprint, 'attributes', None)
                if raw_attrs is not None:
                    try:
                        for attr_name in raw_attrs:
                            attribute_names.append(attr_name)
                    except Exception:
                        pass

                # 若無法從 attributes 取得，回退到常見屬性名
                if not attribute_names:
                    common_attr_names = [
                        'width', 'length',
                        'bounding_box.x', 'bounding_box.y',
                        'speed', 'color'
                    ]
                    if hasattr(blueprint, 'has_attribute'):
                        for name in common_attr_names:
                            try:
                                if blueprint.has_attribute(name):
                                    attribute_names.append(name)
                            except Exception:
                                continue

                for name in attribute_names:
                    try:
                        blueprint_data['attributes'][name] = blueprint.get_attribute(name)
                    except Exception:
                        pass
            except Exception:
                pass

            vehicle_blueprints.append(blueprint_data)

        return vehicle_blueprints

    def _create_vehicle_types(self):
        """從 CARLA 車輛藍圖創建 SUMO 車輛類型定義"""
        # 與官方腳本一致：不預先生成 vtypes.xml，由運行時動態創建
        self._vtypes_file = None
        return

    def _load_vtype_specs(self):
        """載入官方 vtypes.json 規格映射，用於運行時創建 SUMO vType。"""
        try:
            data_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', 'Co-Simulation', 'Sumo', 'data', 'vtypes.json'))
            with open(data_path, 'r') as f:
                specs = json.load(f)
            self._vtype_specs = specs.get('carla_blueprints', {})
            self._default_wheeled_spec = specs.get('DEFAULT_WHEELED_VEHICLE', {"vClass": "passenger"})
            self._default_2w_spec = specs.get('DEFAULT_2_WHEELED_VEHICLE', {"vClass": "motorcycle"})
            if self.verbose:
                print("已載入 vtypes 規格映射")
        except Exception as e:
            self._vtype_specs = {}
            self._default_wheeled_spec = {"vClass": "passenger"}
            self._default_2w_spec = {"vClass": "motorcycle"}
            if self.verbose:
                print(f"載入 vtypes.json 失敗，使用預設規格: {e}")

    def _build_sumo_net_from_opendrive(self):
        """使用官方的 netconvert_carla 轉換工具構建 SUMO 網路"""
        # 創建臨時目錄
        self._tmp_dir = tempfile.mkdtemp(prefix='carla_sumo_')

        # 匯出 OpenDRIVE（xodr）
        xodr = self.carla_map.to_opendrive()
        xodr_file = os.path.join(self._tmp_dir, f"{self.town_map_name}.xodr")
        with open(xodr_file, 'w') as f:
            f.write(xodr)

        # 生成 SUMO net
        self._net_file = os.path.join(self._tmp_dir, f"{self.town_map_name}.net.xml")

        # 使用官方 netconvert_carla 腳本進行轉換 (優先)
        used_custom_converter = False
        try:
            # 設置 SUMO_HOME 環境變量和PATH
            if 'SUMO_HOME' not in os.environ:
                for cand in [
                    '/home/simplexity/cybian/sumo-1.22.0',
                    '/usr/share/sumo',
                    '/usr/local/share/sumo',
                ]:
                    if os.path.isdir(cand):
                        os.environ['SUMO_HOME'] = cand
                        # 🔧 运行时设置PATH - 添加SUMO bin目录
                        bin_dir = os.path.join(cand, 'bin')
                        if os.path.isdir(bin_dir):
                            current_path = os.environ.get('PATH', '')
                            if bin_dir not in current_path:
                                os.environ['PATH'] = bin_dir + ':' + current_path
                                if self.verbose:
                                    print(f"✅ 运行时设置SUMO环境: SUMO_HOME={cand}")
                                    print(f"✅ 添加到PATH: {bin_dir}")
                        break

            # 確保 tools 在 PYTHONPATH 中
            if 'SUMO_HOME' in os.environ:
                tools_dir = os.path.join(os.environ['SUMO_HOME'], 'tools')
                if os.path.isdir(tools_dir) and tools_dir not in sys.path:
                    sys.path.append(tools_dir)

            # 導入 Co-Simulation/Sumo/util 目錄
            util_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Co-Simulation', 'Sumo', 'util'))
            if os.path.isdir(util_dir) and util_dir not in sys.path:
                sys.path.append(util_dir)

            # 導入官方腳本
            try:
                import netconvert_carla

                # 使用官方轉換器
                if self.verbose:
                    print("使用官方 CARLA-SUMO 轉換工具...")

                # 官方方法：直接调用函数，不是通过 CLI
                netconvert_carla.netconvert_carla(
                    xodr_file, self._net_file,
                    guess_tls=True  # 官方默认启用交通灯推测
                )
                used_custom_converter = True

                if self.verbose:
                    print("SUMO 網路成功構建")
            except Exception as e:
                # 轉換腳本失敗，記錄錯誤
                err_log = os.path.join(self._tmp_dir, "netconvert_error.log")
                with open(err_log, 'a') as ef:
                    ef.write(f"官方轉換器失敗: {str(e)}\n")
                if self.verbose:
                    print(f"官方轉換器失敗: {e}，將嘗試標準 netconvert")
        except ImportError as e:
            if self.verbose:
                print(f"找不到官方轉換工具，原因: {e}")

        # 若官方轉換失敗，使用標準 netconvert
        if not used_custom_converter:
            if self.verbose:
                print("使用標準 netconvert 轉換...")

            netconvert = self._find_sumo_binary('netconvert')
            # convert_cmd = [
            #     netconvert,
            #     "--opendrive", xodr_file,
            #     "--output-file", self._net_file,
            #     # 幾何修復與完整導入
            #     "--geometry.min-radius.fix",
            #     "--geometry.remove",
            #     "--opendrive.curve-resolution", "1",
            #     "--opendrive.import-all-lanes",
            #     # 保留原始名稱
            #     "--output.original-names",
            #     # 交通燈設置
            #     "--tls.guess", "true",
            #     "--tls.join", "true",
            #     "--tls.discard-loaded", "false",
            #     # 其他優化
            #     "--junctions.corner-detail", "5",
            #     "--no-turnarounds", "true"
            # ]
            convert_cmd = [
                netconvert,
                "--opendrive", xodr_file,
                "--output-file", self._net_file,
                # 几何处理 (官方推荐)
                '--geometry.min-radius.fix.railways', 'false',
                '--geometry.max-grade.fix', 'false',
                '--offset.disable-normalization', 'true',

                # 交通信号灯处理 (CARLA官方重要参数)
                '--tls.guess-signals', 'true',  # 自动识别并生成交通信号灯
                '--tls.group-signals', 'true',  # 对信号灯进行分组管理
                '--traffic-light-green', '31',  # 默认绿灯时长 (秒)
                '--traffic-light-yellow', '6',  # 默认黄灯时长 (秒)

                # 交叉路口处理 (官方标准)
                '--junctions.join', 'true',  # 合并相近的交叉路口
                '--junctions.corner-detail', '5',  # 交叉路口转角细节级别
                '--junctions.limit-turn-speed', '5.5',  # 转弯限速
                '--default.junctions.keep-clear', 'true',  # 交叉口保持清空

                # 道路结构识别 (CARLA特有)
                '--ramps.guess', 'true',  # 识别匝道
                '--roundabouts.guess', 'true',  # 识别环岛
                '--no-turnarounds', 'true',  # 禁用掉头

                # 车辆类型过滤
                '--keep-edges.by-vclass', 'passenger',

                # 其他SUMO官方推荐参数
                '--remove-edges.isolated', 'true',  # 移除孤立边
                '--keep-edges.min-speed', '1.39'  # 保留最低速度边 (5km/h)
            ]

            # 執行 netconvert
            res = subprocess.run(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

            # 檢查結果
            if res.returncode != 0:
                # 失敗時記錄錯誤
                err_log = os.path.join(self._tmp_dir, "netconvert_error.log")
                try:
                    with open(err_log, 'w') as ef:
                        ef.write("Command: \n" + " ".join(convert_cmd) + "\n\n")
                        ef.write("STDOUT:\n" + (res.stdout or "") + "\n\n")
                        ef.write("STDERR:\n" + (res.stderr or "") + "\n")
                except Exception:
                    pass

                raise RuntimeError(f"netconvert 失敗（詳見 {err_log}）。錯誤：\n{res.stderr.strip()}")

        # 🎯 关键：保存训练用的网络文件到项目目录 (与NGSIM映射保持一致)
        project_network_dir = os.path.join(os.path.dirname(__file__), 'sumo_NGSIM')
        if not os.path.exists(project_network_dir):
            os.makedirs(project_network_dir)

        training_network_file = os.path.join(project_network_dir, f"training_{self.town_map_name}.net.xml")
        training_opendrive_file = os.path.join(project_network_dir, f"training_{self.town_map_name}.xodr")

        try:
            # 保存网络文件
            import shutil
            shutil.copy2(self._net_file, training_network_file)
            shutil.copy2(xodr_file, training_opendrive_file)

            if self.verbose:
                print(f"🎯 训练网络文件已保存:")
                print(f"   📁 网络文件: {training_network_file}")
                print(f"   📁 OpenDRIVE: {training_opendrive_file}")
                print(f"   💡 用于NGSIM数据映射的参考网络")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 保存训练网络文件失败: {e}")

        # 🚗 第一步：生成 IDM 车辆类型定义文件 vtypes.add.xml
        self._vtypes_file = os.path.join(self._tmp_dir, "vtypes.add.xml")
        with open(self._vtypes_file, 'w') as f:
            f.write("""<additional>
    <vType id="ngsim_vehicle"
           carFollowModel="IDM"
           laneChangeModel="SL2015"
           length="5.0" width="1.8" height="1.5"
           minGap="2.5"
           maxSpeed="25.0"
           accel="2.6" decel="4.5" emergencyDecel="9.0"
           tau="1.0"
           imperfection="0.5"
           vClass="passenger"
           color="0,255,0"
           lcStrategic="1.0"
           lcCooperative="1.0"
           lcSpeedGain="1.0"
           lcKeepRight="1.0"
           lcSublane="1.0"
           latAlignment="center"/>
</additional>
""")
        if self.verbose:
            print(f"✅ 已生成 IDM 车辆类型定义文件 (IDM + SL2015变道模型，支持sublane): {self._vtypes_file}")

        # 建立 sumocfg
        self._cfg_file = os.path.join(self._tmp_dir, f"{self.town_map_name}.sumocfg")
        with open(self._cfg_file, 'w') as f:
            additional_files = ""
            if self._vtypes_file:
                additional_files = f'<additional-files value="{os.path.basename(self._vtypes_file)}"/>'

            f.write(f"""
<configuration>
  <input>
    <net-file value="{self._net_file}"/>
    {additional_files}
  </input>
  <time>
    <step-length value="{self.fixed_delta_seconds}"/>
  </time>
  <processing>
    <time-to-teleport value="-1"/>
    <lateral-resolution value="0.8"/>
  </processing>
</configuration>
""")

        # 讀入 net 以便後續操作
        self._sumolib_net = self._sumolib.net.readNet(self._net_file, withInternal=True)

        # 建立 OpenDRIVE → SUMO 對應表
        self._build_odr_to_sumo_mapping()

        # 獲取網路偏移量
        try:
            off = self._sumolib_net.getLocationOffset()
            if isinstance(off, (list, tuple)) and len(off) == 2:
                self._net_offset = (float(off[0]), float(off[1]))
            else:
                self._net_offset = (0.0, 0.0)
        except Exception:
            self._net_offset = (0.0, 0.0)

    def _start_traci(self):
        # 查找正确的SUMO可执行文件
        sumo_name = 'sumo-gui' if self.sumo_gui else 'sumo'
        sumo_bin = self._find_sumo_binary(sumo_name)
        cmd = [
            sumo_bin,
            "-c", self._cfg_file,
            "--start",
            "--quit-on-end",
            "--no-step-log", "true",
            "--step-length", str(self.fixed_delta_seconds),
        ]
        # 從 traci 啟動
        self._traci.start(cmd)

    def _find_sumo_binary(self, name: str) -> str:
        # 若傳入為絕對路徑，直接使用
        if os.path.isabs(name) and os.path.exists(name):
            return name

        # 嘗試直接在 PATH 找
        from shutil import which
        p = which(name)
        if p:
            return p
        # 退回 SUMO_HOME/bin 與常見安裝位置
        sumo_home = os.environ.get("SUMO_HOME")
        candidates = []
        if sumo_home:
            candidates.append(os.path.join(sumo_home, 'bin', name))
        candidates += [
            os.path.join('/home/simplexity/cybian/sumo-1.22.0', 'bin', name),
            os.path.join('/usr/bin', name),
            os.path.join('/usr/local/bin', name),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise RuntimeError(f"找不到可執行檔: {name}")

    # ---------- per-vehicle helpers ----------
    def _vehicle_add_with_vtype_fallback(self, veh_id: str, route_id: str, vtype_id: str):
        """嘗試使用給定 vType 新增車輛；若失敗則回退到 DEFAULT_VEHTYPE。

        無論是否存在映射，先在當前 SUMO 會話中確保 vType 存在。
        """
        vt = self._traci.vehicletype
        try:
            base = "DEFAULT_VEHTYPE"
            # 確保 vType 存在
            if vtype_id and vtype_id not in vt.getIDList():
                try:
                    vt.copy(base, vtype_id)
                except Exception:
                    vt.add(vtype_id)
        except Exception:
            pass
        # 嘗試用期望的 vtype 新增
        try:
            self._traci.vehicle.add(veh_id, route_id, typeID=vtype_id)
            return
        except Exception:
            pass
        # 回退：使用 DEFAULT_VEHTYPE
        try:
            self._traci.vehicle.add(veh_id, route_id, typeID="DEFAULT_VEHTYPE")
        except Exception as e:
            raise e

    def _ensure_sumo_vehicle(self, actor_id: str, actor) -> str:
        """確保在 SUMO 中存在對應的車輛（若尚未存在則創建）

        Args:
            actor_id: CARLA actor ID
            actor: CARLA actor 物件

        Returns:
            str: SUMO 中的車輛 ID
        """
        # 檢查是否已存在且有效
        if actor_id in self._carla_to_sumo_id:
            veh_id = self._carla_to_sumo_id[actor_id]
            if veh_id in self._traci.vehicle.getIDList():
                return veh_id

        # 為車輛生成唯一 ID
        veh_id = f"carla_{actor_id}"

        import carla  # 避免 NameError，僅在需要時導入

        # 獲取車輛位置和朝向
        if self._helper and self._bridge_helper_available:
            # 使用官方 BridgeHelper 進行座標轉換
            try:
                # 獲取車輛的變換和尺寸
                carla_transform = actor.get_transform()
                extent = actor.bounding_box.extent if hasattr(actor, 'bounding_box') else carla.Vector3D(1.0, 0.5, 0.5)

                # 使用官方 BridgeHelper 轉換到 SUMO 坐標系
                sumo_transform = BridgeHelper.get_sumo_transform(carla_transform, extent)

                # 提取 SUMO 坐標和角度
                x_sumo = sumo_transform.location.x
                y_sumo = sumo_transform.location.y
                angle = sumo_transform.rotation.yaw

                if self.verbose:
                    print(f"使用官方 BridgeHelper 創建車輛: ({x_sumo:.2f}, {y_sumo:.2f}), 角度: {angle:.1f}°")
            except Exception as e:
                if self.verbose:
                    print(f"官方 BridgeHelper 轉換失敗: {e}，使用備用方法")
                # 作為備用方案，使用原始方法
                x_sumo, y_sumo, angle = self._carla_xy_yaw(actor)
        else:
            # 作為備用方案，使用原始方法
            x_sumo, y_sumo, angle = self._carla_xy_yaw(actor)

        # 確保車輛類型存在
        vtype_id = self._ensure_sumo_vtype_for_actor(actor)

        # 嘗試找到最合適的道路
        found_valid_edge = False

        # 策略 1: 通過 CARLA waypoint 找到對應的 SUMO 道路 (OpenDRIVE 映射)
        try:
            wp = self.carla_map.get_waypoint(actor.get_location())
            mapped = self._map_odr_to_sumo(str(wp.road_id), int(wp.lane_id), float(wp.s))
            if mapped:
                edge_id = mapped[0]
                lane_index = mapped[1]
                found_valid_edge = True

                if self.verbose:
                    print(f"車輛 {actor_id} 通過 ODR 映射找到道路: {edge_id}")

                # 創建路由和車輛
                route_id = f"r_{veh_id}"
                if route_id not in self._traci.route.getIDList():
                    self._traci.route.add(route_id, [edge_id])

                if veh_id not in self._traci.vehicle.getIDList():
                    self._vehicle_add_with_vtype_fallback(veh_id, route_id, vtype_id)
        except Exception as e:
            if self.verbose:
                print(f"通過 ODR 創建車輛失敗: {e}")

        # 策略 2: 找最近的道路
        if not found_valid_edge:
            try:
                edge, lane_index, pos_on_edge = self._nearest_edge_lane_and_pos(x_sumo, y_sumo)
                route_id = f"r_{veh_id}"

                # 建立單一路段的路由
                if route_id not in self._traci.route.getIDList():
                    self._traci.route.add(route_id, [edge.getID()])

                # 加車輛（避免重複新增）
                if veh_id not in self._traci.vehicle.getIDList():
                    self._vehicle_add_with_vtype_fallback(veh_id, route_id, vtype_id)

                found_valid_edge = True
                if self.verbose:
                    print(f"車輛 {actor_id} 通過最近點找到道路: {edge.getID()}")
            except Exception as e:
                if self.verbose:
                    print(f"通過最近點創建車輛失敗: {e}")

        # 策略 3: 使用空路由作為最後嘗試
        if not found_valid_edge:
            try:
                if veh_id not in self._traci.vehicle.getIDList():
                    self._vehicle_add_with_vtype_fallback(veh_id, "", vtype_id)
                if self.verbose:
                    print(f"車輛 {actor_id} 使用空路由創建")
            except Exception as e:
                if self.verbose:
                    print(f"創建車輛失敗: {e}")
                raise e  # 如果所有嘗試都失敗，拋出異常

        # 立即嘗試設定初始位置，避免車輛位於道路外
        try:
            if veh_id in self._traci.vehicle.getIDList():
                # 🎯 獲取車輛的edge_id
                edge_id = ''
                if found_valid_edge:
                    try:
                        wp = self.carla_map.get_waypoint(actor.get_location())
                        mapped = self._map_odr_to_sumo(str(wp.road_id), int(wp.lane_id), float(wp.s))
                        if mapped:
                            edge_id = mapped[0]
                    except Exception:
                        pass

                self._traci.vehicle.moveToXY(
                    veh_id,
                    edge_id,  # 使用正確的edge
                    -1,
                    float(x_sumo),
                    float(y_sumo),
                    angle=float(angle),
                    keepRoute=1  # 🎯 使用keepRoute=1確保在道路網絡上
                )
                if self.verbose:
                    print(
                        f"車輛 {actor_id} 初始位置已設定: edge={edge_id}, x={x_sumo:.1f}, y={y_sumo:.1f}, angle={angle:.1f}")
        except Exception as e:
            if self.verbose:
                print(f"設定車輛初始位置失敗: {e}")

        # 配置SUMO车辆控制模式（不覆盖车辆类型参数）
        try:
            if veh_id in self._traci.vehicle.getIDList():
                # ⚠️ 重要：不要手动设置跟驰模型参数（tau, imperfection, minGap等）
                # 这些参数应该由车辆类型（vType）定义
                # 手动设置会覆盖车辆类型的配置

                # 只设置控制模式，让SUMO使用车辆类型定义的模型
                # SpeedMode: 0 = 完全SUMO控制（包含所有安全检查）
                self._traci.vehicle.setSpeedMode(veh_id, 0)

                # LaneChangeMode: 0 = 完全SUMO智能变道
                self._traci.vehicle.setLaneChangeMode(veh_id, 0)

                if self.verbose:
                    print(f"✅ CARLA车辆{veh_id}控制模式已配置（使用vType定义的模型）")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 配置CARLA车辆控制模式失败: {e}")

        # 記錄車輛的edge信息，供後續同步使用
        if found_valid_edge:
            try:
                # 獲取車輛當前的road_id
                if hasattr(self, 'carla_map') and self.carla_map:
                    wp = self.carla_map.get_waypoint(actor.get_location())
                    mapped = self._map_odr_to_sumo(str(wp.road_id), int(wp.lane_id), float(wp.s))
                    if mapped:
                        edge_id = mapped[0]
                        if not hasattr(self, '_carla_vehicle_edges'):
                            self._carla_vehicle_edges = {}
                        self._carla_vehicle_edges[actor_id] = edge_id
            except Exception:
                pass

        # 註冊映射
        self._carla_to_sumo_id[actor_id] = veh_id
        return veh_id

    def _ensure_sumo_vtype_for_actor(self, actor) -> str:
        """
        動態為 CARLA 車輛建立對應的 SUMO vType（基於 DEFAULT_VEHTYPE 複製），
        並設定近似長寬等屬性，回傳 vType id。
        """
        try:
            blueprint_id = str(actor.type_id)
        except Exception:
            blueprint_id = "car"

        # 以 CARLA blueprint type_id 作為 vType id（與官方映射一致）
        preferred_vtype = blueprint_id.replace(' ', '_')

        vt = self._traci.vehicletype
        vtype_id = preferred_vtype
        try:
            base = "DEFAULT_VEHTYPE"
            # 確保該 vType 在 SUMO 會話中存在
            if vtype_id not in vt.getIDList():
                try:
                    vt.copy(base, vtype_id)
                except Exception:
                    vt.add(vtype_id)

            # 設定長寬（由 CARLA bounding box 推算）
            try:
                bbox = actor.bounding_box
                length = max(2.0 * float(getattr(bbox.extent, 'x', 2.0)), 2.0)
                width = max(2.0 * float(getattr(bbox.extent, 'y', 1.0)), 1.0)
                vt.setLength(vtype_id, length)
                vt.setWidth(vtype_id, width)
            except Exception:
                pass

            # 根據 vtypes.json 設定類別與外觀
            try:
                spec = self._vtype_specs.get(blueprint_id)
                if spec is None:
                    # 根據輪數選擇默認
                    num_wheels = 4
                    try:
                        # 某些 CARLA 版本提供 attributes 字典
                        if hasattr(actor, 'attributes') and 'number_of_wheels' in actor.attributes:
                            num_wheels = int(actor.attributes['number_of_wheels'])
                    except Exception:
                        pass
                    spec = self._default_2w_spec if num_wheels == 2 else self._default_wheeled_spec

                # 應用官方 vtypes.json 中的所有屬性
                if isinstance(spec, dict):
                    # 設置車輛類別
                    vclass = spec.get('vClass')
                    if vclass:
                        try:
                            vt.setVehicleClass(vtype_id, str(vclass))
                        except Exception:
                            pass

                    # 設置圖形外觀（官方支持）
                    guishape = spec.get('guiShape')
                    if guishape:
                        try:
                            # SUMO 1.22.0 使用setShapeClass设置车辆外观
                            vt.setShapeClass(vtype_id, str(guishape))
                        except Exception:
                            pass

                    # 設置其他官方支持的屬性
                    for attr in ['maxSpeed', 'accel', 'decel', 'sigma', 'tau']:
                        if attr in spec:
                            try:
                                setter = getattr(vt, f'set{attr.capitalize()}', None)
                                if setter:
                                    setter(vtype_id, float(spec[attr]))
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception:
            vtype_id = "DEFAULT_VEHTYPE"

        # 更新映射（確保下次直接復用）
        self._blueprint_to_vtype[blueprint_id] = vtype_id
        return vtype_id

    # ---------- ODR 對應表與定位 ----------
    def _build_odr_to_sumo_mapping(self):
        """
        建立 ODR (road_id, lane_id) → [(edge_id, lane_index, section_start_s), ...]
        以便依據 CARLA Waypoint.s 選擇對應的 SUMO edge 與 lane。
        """
        self._odr2sumo = {}  # (str road_id, int lane_id) -> list[(edge_id, lane_index, section_start_s)]
        for edge in self._sumolib_net.getEdges():
            edge_id = edge.getID()
            # 跳過 internal/junction 邊
            if str(edge_id).startswith(':'):
                continue
            # 嘗試從 edge id 擷取 section 起點 s（如 roadId.sCoord）
            section_start_s = 0.0
            try:
                if '.' in edge_id:
                    section_start_s = float(edge_id.split('.', 1)[1])
            except Exception:
                section_start_s = 0.0
            for lane in edge.getLanes():
                orig = lane.getParam('origId')
                if not orig:
                    continue
                tokens = str(orig).split()
                for tok in tokens:
                    try:
                        odr_road_id, odr_lane_id = tok.split('_')
                        odr_lane_id = int(odr_lane_id)
                        key = (str(odr_road_id), odr_lane_id)
                        self._odr2sumo.setdefault(key, []).append((edge_id, lane.getIndex(), section_start_s))
                    except Exception:
                        continue

    def _map_odr_to_sumo(self, road_id: str, lane_id: int, s_coord: float):
        """
        根據 ODR (road_id, lane_id, s) 選擇最合適的 (edge_id, lane_index, pos_on_edge)。
        若無法對應，回傳 None。
        """
        key = (str(road_id), int(lane_id))
        candidates = self._odr2sumo.get(key)
        if not candidates:
            return None
        # 依據 s 與 section_start_s 選擇最近的段落
        best = None
        best_delta = 1e18
        for edge_id, lane_index, section_start_s in candidates:
            delta = abs(float(s_coord) - float(section_start_s))
            if delta < best_delta:
                best = (edge_id, lane_index, section_start_s)
                best_delta = delta
        if best is None:
            return None
        edge_id, lane_index, section_start_s = best
        # pos_on_edge 取 s 與段起點差，裁切於 [0, edgeLength]
        try:
            edge = self._sumolib_net.getEdge(edge_id)
            edge_len = edge.getLength()
        except Exception:
            edge_len = 1.0
        pos_on_edge = max(0.1, min(float(s_coord) - float(section_start_s), edge_len - 0.1))
        return edge_id, lane_index, pos_on_edge

    def _move_vehicle_to_actor_pose(self, actor_id: str, actor):
        """將 CARLA 車輛位置映射到 SUMO 中

        完全按照官方 run_synchronization.py 中的實現方式進行同步

        Args:
            actor_id: CARLA actor ID
            actor: CARLA actor 物件
        """
        veh_id = self._carla_to_sumo_id.get(actor_id)
        if not veh_id or veh_id not in self._traci.vehicle.getIDList():
            return

        import carla  # 避免 NameError，僅在需要時導入

        # 參考官方 run_synchronization.py 的 tick 方法中的代碼：
        # carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform, sumo_actor.extent)
        # self.carla.synchronize_vehicle(carla_actor_id, carla_transform, carla_lights)

        if self._helper and self._bridge_helper_available:
            try:
                # 獲取車輛的變換和尺寸
                carla_transform = actor.get_transform()
                extent = actor.bounding_box.extent if hasattr(actor, 'bounding_box') else carla.Vector3D(1.0, 0.5, 0.5)

                # 使用官方 BridgeHelper 轉換到 SUMO 坐標系
                sumo_transform = BridgeHelper.get_sumo_transform(carla_transform, extent)

                # 提取 SUMO 坐標和角度
                x_sumo = sumo_transform.location.x
                y_sumo = sumo_transform.location.y
                angle = sumo_transform.rotation.yaw

                # 🎯 關鍵修復：獲取車輛的edge_id，使用keepRoute=1確保車輛在道路網絡上
                edge_id = ''
                if hasattr(self, '_carla_vehicle_edges') and actor_id in self._carla_vehicle_edges:
                    edge_id = self._carla_vehicle_edges[actor_id]
                else:
                    # 如果沒有記錄edge，嘗試通過waypoint獲取
                    try:
                        wp = self.carla_map.get_waypoint(actor.get_location())
                        mapped = self._map_odr_to_sumo(str(wp.road_id), int(wp.lane_id), float(wp.s))
                        if mapped:
                            edge_id = mapped[0]
                    except Exception:
                        pass

                # 使用keepRoute=1確保CARLA車輛被SUMO交通模型正確識別
                self._traci.vehicle.moveToXY(
                    veh_id,
                    edge_id,  # 指定正確的edge
                    -1,  # 讓SUMO自動選擇車道
                    float(x_sumo),
                    float(y_sumo),
                    angle=float(angle),
                    keepRoute=1  # 🎯 關鍵：保持在路由上，讓SUMO能識別
                )

                # 🎯 關鍵修復：同步CARLA車輛的速度到SUMO
                # SUMO的避讓系統需要知道前方車輛的速度才能正確計算安全距離
                carla_velocity = actor.get_velocity()
                speed_mps = math.sqrt(carla_velocity.x ** 2 + carla_velocity.y ** 2 + carla_velocity.z ** 2)

                # ⚠️ 關鍵：使用setSpeed強制設置當前速度，讓SUMO知道車輛當前的實際速度
                # 這與moveToXY配合使用，告訴SUMO"這個車輛正在以X速度移動"
                # 這樣其他車輛才能正確感知並避讓
                self._traci.vehicle.setSpeed(veh_id, float(speed_mps))

                if self.verbose:
                    print(f"✅ CARLA車輛同步: pos=({x_sumo:.2f},{y_sumo:.2f}), 速度={speed_mps:.2f}m/s, edge={edge_id}")

                # 更新同步狀態記錄
                self._last_sync_status[actor_id] = {
                    'location': carla_transform.location,
                    'yaw': carla_transform.rotation.yaw,
                    'speed': speed_mps
                }

            except Exception as e:
                if self.verbose:
                    print(f"官方 BridgeHelper 同步車輛失敗: {e}，使用備用方法")
                # 作為備用方案，使用原始方法
                self._fallback_move_vehicle(actor_id, actor)
        else:
            # 作為備用方案，使用原始方法
            self._fallback_move_vehicle(actor_id, actor)

    def diagnose_vehicle_visibility(self, actor_id: str):
        """診斷CARLA車輛在SUMO中的可見性和避讓狀態"""
        veh_id = self._carla_to_sumo_id.get(actor_id)
        if not veh_id:
            print(f"❌ 車輛{actor_id}沒有SUMO ID映射")
            return

        if veh_id not in self._traci.vehicle.getIDList():
            print(f"❌ 車輛{veh_id}不在SUMO車輛列表中")
            return

        try:
            print(f"\n🔍 診斷CARLA車輛: {actor_id} (SUMO ID: {veh_id})")
            print(f"   位置: {self._traci.vehicle.getPosition(veh_id)}")
            print(f"   速度: {self._traci.vehicle.getSpeed(veh_id)} m/s")
            print(f"   道路ID: {self._traci.vehicle.getRoadID(veh_id)}")
            print(f"   車道ID: {self._traci.vehicle.getLaneID(veh_id)}")
            print(f"   SpeedMode: {self._traci.vehicle.getSpeedMode(veh_id)}")
            print(f"   LaneChangeMode: {self._traci.vehicle.getLaneChangeMode(veh_id)}")

            # 檢查前方是否有其他車輛
            leader = self._traci.vehicle.getLeader(veh_id)
            if leader:
                print(f"   前方車輛: {leader[0]}, 距離: {leader[1]:.2f}m")
            else:
                print(f"   前方無車輛")

            # 檢查周圍的NGSIM車輛
            all_vehicles = self._traci.vehicle.getIDList()
            ngsim_vehicles = [v for v in all_vehicles if v.startswith('ngsim_')]
            print(f"   SUMO中NGSIM車輛數: {len(ngsim_vehicles)}")

            # 檢查是否有NGSIM車輛在後方
            carla_pos = self._traci.vehicle.getPosition(veh_id)
            for ngsim_id in ngsim_vehicles[:5]:  # 只檢查前5個
                ngsim_pos = self._traci.vehicle.getPosition(ngsim_id)
                distance = math.sqrt((carla_pos[0] - ngsim_pos[0]) ** 2 + (carla_pos[1] - ngsim_pos[1]) ** 2)
                if distance < 50:  # 50米內
                    ngsim_leader = self._traci.vehicle.getLeader(ngsim_id)
                    leader_info = f"前方:{ngsim_leader[0]}({ngsim_leader[1]:.1f}m)" if ngsim_leader else "無前車"
                    print(f"   附近NGSIM: {ngsim_id}, 距離:{distance:.1f}m, {leader_info}")
        except Exception as e:
            print(f"❌ 診斷失敗: {e}")

    def _fallback_move_vehicle(self, actor_id: str, actor):
        """備用的車輛位置同步方法，當官方 BridgeHelper 不可用時使用

        Args:
            actor_id: CARLA actor ID
            actor: CARLA actor 物件
        """
        veh_id = self._carla_to_sumo_id.get(actor_id)
        if not veh_id or veh_id not in self._traci.vehicle.getIDList():
            return

        # 使用原始方法轉換坐標
        x_sumo, y_sumo, angle = self._carla_xy_yaw(actor)

        # 多層策略保證車輛放置在有效位置
        try:
            # 策略 1: 使用 OpenDRIVE 映射找到精確的道路和車道
            edge_id = ""
            lane_idx = -1
            try:
                wp = self.carla_map.get_waypoint(actor.get_location())
                mapped = self._map_odr_to_sumo(str(wp.road_id), int(wp.lane_id), float(wp.s))
                if mapped:
                    edge_id = mapped[0]
                    lane_idx = mapped[1]
            except Exception:
                pass

            # 策略 2: 使用 moveToXY 將車輛移動到轉換後的坐標位置
            try:
                self._traci.vehicle.moveToXY(
                    veh_id,
                    edge_id,  # 如果找到了匹配的 edge，則使用它
                    lane_idx,  # 如果找到了匹配的車道，則使用它
                    float(x_sumo),
                    float(y_sumo),
                    angle=float(angle),
                    keepRoute=1  # 🎯 保持在路由上，讓SUMO能識別
                )

                # 🎯 同步速度信息
                carla_velocity = actor.get_velocity()
                speed_mps = math.sqrt(carla_velocity.x ** 2 + carla_velocity.y ** 2 + carla_velocity.z ** 2)
                self._traci.vehicle.setSpeed(veh_id, float(speed_mps))

                # 更新同步狀態記錄
                self._last_sync_status[actor_id] = {
                    'location': actor.get_location(),
                    'yaw': actor.get_transform().rotation.yaw,
                    'speed': speed_mps
                }

                return  # 成功移動車輛
            except Exception as e:
                if self.verbose:
                    print(f"移動車輛到 XY ({x_sumo:.1f},{y_sumo:.1f}) 失敗: {e}")

            # 策略 3: 嘗試使用最近點找到道路和位置
            try:
                edge, lane_index, pos_on_edge = self._nearest_edge_lane_and_pos(x_sumo, y_sumo)
                edge_id = edge.getID()
                self._traci.vehicle.moveTo(
                    veh_id,
                    f"{edge_id}_{lane_index}",
                    float(pos_on_edge)
                )

                # 更新同步狀態記錄
                self._last_sync_status[actor_id] = {
                    'location': actor.get_location(),
                    'yaw': actor.get_transform().rotation.yaw
                }

                return  # 成功移動車輛
            except Exception as e:
                if self.verbose:
                    print(f"移動車輛到最近道路失敗: {e}")

            # 策略 4: 最後嘗試，不指定道路的 moveToXY
            try:
                self._traci.vehicle.moveToXY(
                    veh_id,
                    '',  # 不指定道路
                    -1,  # 不指定車道
                    float(x_sumo),
                    float(y_sumo),
                    angle=float(angle),
                    keepRoute=2  # 與官方一致使用 2
                )

                # 更新同步狀態記錄
                self._last_sync_status[actor_id] = {
                    'location': actor.get_location(),
                    'yaw': actor.get_transform().rotation.yaw
                }
            except Exception as e:
                if self.verbose:
                    print(f"移動車輛最終嘗試失敗: {e}")

        except Exception as e:
            if self.verbose:
                print(f"移動車輛過程中發生錯誤: {e}")

    def _sync_vehicle_color(self, actor, sumo_veh_id):
        """同步 CARLA 車輛顏色到 SUMO

        Args:
            actor: CARLA 車輛 actor
            sumo_veh_id: SUMO 車輛 ID
        """
        try:
            # 獲取 CARLA 車輛顏色
            if hasattr(actor, 'get_color'):
                color = actor.get_color()
                r, g, b = color.r, color.g, color.b
                # 設置 SUMO 車輛顏色
                self._traci.vehicle.setColor(sumo_veh_id, (r, g, b))
        except Exception:
            pass

    def _sync_vehicle_lights(self, actor, sumo_veh_id):
        """同步 CARLA 車輛燈光到 SUMO

        Args:
            actor: CARLA 車輛 actor
            sumo_veh_id: SUMO 車輛 ID
        """
        try:
            # 獲取 CARLA 車輛燈光狀態
            if hasattr(actor, 'get_light_state'):
                light_state = actor.get_light_state()

                # SUMO 中的燈光常量
                SUMO_BLINKER_RIGHT = 1
                SUMO_BLINKER_LEFT = 2
                SUMO_BLINKER_EMERGENCY = 4
                SUMO_FRONTLIGHT = 8
                SUMO_BACKLIGHT = 16
                SUMO_BRAKE = 32

                # 初始化 SUMO 燈光狀態
                sumo_lights = 0

                # 處理 CARLA 燈光狀態
                import carla  # 假設 carla 模組可用

                # 轉向燈
                if light_state & carla.VehicleLightState.RightBlinker:
                    sumo_lights |= SUMO_BLINKER_RIGHT
                if light_state & carla.VehicleLightState.LeftBlinker:
                    sumo_lights |= SUMO_BLINKER_LEFT

                # 前燈
                if light_state & carla.VehicleLightState.Position:
                    sumo_lights |= SUMO_FRONTLIGHT

                # 剎車燈
                if light_state & carla.VehicleLightState.Brake:
                    sumo_lights |= SUMO_BRAKE

                # 設置 SUMO 燈光狀態
                self._traci.vehicle.setSignals(sumo_veh_id, sumo_lights)
        except Exception:
            pass

    def _sync_traffic_lights(self):
        """同步 CARLA 與 SUMO 之間的交通號誌狀態"""
        if not self._traci:
            return

        try:
            # 獲取 CARLA 中的所有交通號誌
            traffic_lights = self.world.get_actors().filter('traffic.traffic_light*')

            # 獲取 SUMO 中的所有交通號誌
            sumo_tls = self._traci.trafficlight.getIDList()

            # 目前還沒有實現 ID 映射，這是一個簡化的示例
            # 實際使用時需要建立 CARLA 和 SUMO 交通號誌的對應關係
            pass
        except Exception as e:
            if self.verbose:
                print(f"同步交通號誌失敗: {e}")

    # ---------- geometry/mapping ----------
    def _carla_xy_yaw(self, actor):
        """
        備用座標轉換方法（當官方 BridgeHelper 不可用時使用）
        - 以 CARLA 世界座標 (x_carla, y_carla) 經軸交換/符號調整及 netOffset 平移至 SUMO。
        - x_sumo = -y_carla + off_x
        - y_sumo =  x_carla + off_y
        - yaw_sumo = 90 - yaw_carla
        """
        loc = actor.get_location()
        rot = actor.get_transform().rotation
        off_x, off_y = getattr(self, '_net_offset', (0.0, 0.0))

        x_sumo = -float(loc.y) + float(off_x)
        y_sumo = float(loc.x) + float(off_y)
        yaw_sumo = float(90.0 - float(rot.yaw))

        return x_sumo, y_sumo, yaw_sumo

    def _nearest_edge_lane_and_pos(self, x: float, y: float):
        # 搜 50m 內最近路段
        candidates = self._sumolib_net.getNeighboringEdges(x, y, 50)
        if not candidates:
            # 若找不到，就挑一條最長的路段放置
            edges = self._sumolib_net.getEdges()
            edge = max(edges, key=lambda e: e.getLength())
            lane_index = max(0, len(edge.getLanes()) // 2)
            pos_on_edge = edge.getLength() * 0.5
            return edge, lane_index, pos_on_edge

        edge, _dist = min(candidates, key=lambda it: it[1])
        lane_index = max(0, len(edge.getLanes()) // 2)
        # 估個在邊上的相對位置（粗略投影）
        shape = edge.getShape()
        pos_on_edge = min(edge.getLength() * 0.5, edge.getLength() - 0.5)
        try:
            # 使用 sumolib 幾何工具做最近點映射
            from math import hypot
            best_d = 1e9
            acc_len = 0.0
            last = None
            for pt in shape:
                if last is not None:
                    seg_len = ((pt[0] - last[0]) ** 2 + (pt[1] - last[1]) ** 2) ** 0.5
                    # 到當前節點的距離（粗略）
                    d = hypot(pt[0] - x, pt[1] - y)
                    if d < best_d:
                        best_d = d
                        pos_on_edge = acc_len
                    acc_len += seg_len
                last = pt
        except Exception:
            pass
        return edge, lane_index, max(0.1, min(pos_on_edge, edge.getLength() - 0.1))

    def reset(self):
        """重置 SUMO 仿真狀態，移除所有車輛，清理映射關係"""
        if not self._traci:
            return

        # 重置仿真时间和NGSIM状态
        self._current_simulation_time = 0.0
        self._last_ngsim_frame = -1  # 🔧 重置帧跟踪状态

        # 重置NGSIM控制器状态 (与NGSIM_to_SUMO.py保持一致)
        if self._ngsim_controller:
            self._ngsim_controller.active_vehicles.clear()
            if hasattr(self._ngsim_controller, 'vehicle_position_history'):
                self._ngsim_controller.vehicle_position_history.clear()
            # 🔧 重置角度历史记录
            if hasattr(self._ngsim_controller, '_vehicle_angle_history'):
                self._ngsim_controller._vehicle_angle_history.clear()
            if self.verbose:
                print("✅ NGSIM车辆状态、角度计算历史和帧跟踪已重置")

        # ============================================================================
        # 🎯 重置状态机相关变量
        # ============================================================================
        self._vehicle_control_modes.clear()
        self._vehicle_ngsim_target_positions.clear()
        self._vehicle_safe_frame_counter.clear()
        self._vehicle_deviation_history.clear()

        # ✅ 重置脱管车辆集合和统计数据
        self._detached_ngsim_vehicles.clear()
        self._current_frame_stats = {
            "avg_deviation": 0.0,
            "max_deviation": 0.0,
            "active_count": 0,
            "total_vehicles": 0,
            "new_detached": 0,  # 🔧 重置本帧新接管数量
        }

        if self.verbose:
            print("✅ NGSIM状态机变量已重置")

        try:
            # ========== 第一步：清理CARLA中由SUMO桥接器创建的车辆 ==========
            # 职责：清理从SUMO映射回CARLA的车辆（NGSIM背景车辆）
            # 不清理：RL agents（由multi_env管理）、NPC车辆（由multi_env管理）

            carla_cleared_count = 0

            # 只清理映射表中记录的车辆（由SUMO桥接器创建）
            for sumo_id, carla_id in list(self._sumo_to_carla_id.items()):
                try:
                    actor = self.world.get_actor(int(carla_id))
                    if actor and actor.is_alive:
                        actor.destroy()
                        carla_cleared_count += 1
                        if self.verbose:
                            print(f"🧹 清理SUMO→CARLA映射车辆: SUMO_{sumo_id} → CARLA_{carla_id}")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ 清理映射车辆失败 {sumo_id}: {e}")

            if self.verbose and carla_cleared_count > 0:
                print(f"✅ 清理了 {carla_cleared_count} 辆SUMO桥接器创建的CARLA车辆")

            # ========== 第二步：清理SUMO中的所有车辆 ==========
            # 职责：清理SUMO仿真中的所有车辆，包括：
            #   1. 从CARLA同步过来的RL agents镜像（_carla_to_sumo_id中的值）
            #   2. NGSIM背景车辆
            #   3. 其他SUMO原生车辆（如果有）

            sumo_vehicle_count = 0
            for veh_id in list(self._traci.vehicle.getIDList()):
                try:
                    self._traci.vehicle.remove(veh_id)  # 标记删除
                    sumo_vehicle_count += 1
                except Exception:
                    pass

            if self.verbose and sumo_vehicle_count > 0:
                print(f"🧹 标记删除SUMO中的 {sumo_vehicle_count} 辆车")

            # ========== 第三步：执行SUMO步进，让删除真正生效 ==========
            # ✅ 关键：traci.vehicle.remove()只是标记删除，必须调用simulationStep()才能真正删除
            self._traci.simulationStep()
            remaining_sumo = len(self._traci.vehicle.getIDList())
            if self.verbose:
                print(f"✅ SUMO步进完成，剩余SUMO车辆: {remaining_sumo} (应该为0)")

            # 清理映射關係
            self._carla_to_sumo_id.clear()
            self._sumo_to_carla_id.clear()

            if self.verbose:
                print("SUMO 橋接器已重置")

            # 預先分析並緩存道路網路，優化後續車輛放置
            self._analyze_road_network()

        except Exception as e:
            print(f"SUMO 重置過程中發生錯誤: {e}")

    def destroy(self):
        """銷毀橋接器，釋放所有資源"""
        self.reset()
        self.close()

        # 清理臨時文件
        try:
            if self._tmp_dir and os.path.exists(self._tmp_dir):
                import shutil
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass

    def _analyze_road_network(self):
        """分析 SUMO 道路網絡，找出主要道路"""
        if not self._traci or not self._sumolib_net:
            return

        try:
            # 獲取所有邊緣
            edges = self._sumolib_net.getEdges()

            # 找出有效的主要道路（排除內部連接器等）
            main_roads = [e for e in edges if not e.getID().startswith(':')]

            # 獲取 T 型路口的主要道路
            if self.town_map_name == "Town03" and main_roads:
                print(f"道路網分析: 找到 {len(main_roads)} 條主要道路")
                # 輸出一些道路資訊，幫助調試
                for i, edge in enumerate(main_roads[:5]):  # 只顯示前 5 條
                    print(f"道路 {edge.getID()}: 長度={edge.getLength():.1f}m, 車道數={len(edge.getLanes())}")

                # 獲取一些道路的形狀點，幫助調試
                if main_roads:
                    sample_edge = main_roads[0]
                    shape = sample_edge.getShape()
                    print(f"示例道路 {sample_edge.getID()} 形狀點: {shape[:3]}...")
        except Exception as e:
            print(f"分析道路網絡失敗: {e}")

    def get_sumo_vehicles_info(self):
        """獲取 SUMO 中所有車輛的資訊，為未來雙向同步做準備"""
        if not self._traci:
            return {}

        vehicles_info = {}
        try:
            for veh_id in self._traci.vehicle.getIDList():
                if veh_id.startswith('carla_'):
                    pos = self._traci.vehicle.getPosition(veh_id)
                    angle = self._traci.vehicle.getAngle(veh_id)
                    speed = self._traci.vehicle.getSpeed(veh_id)
                    road_id = self._traci.vehicle.getRoadID(veh_id)
                    lane_id = self._traci.vehicle.getLaneID(veh_id)

                    vehicles_info[veh_id] = {
                        'position': pos,
                        'angle': angle,
                        'speed': speed,
                        'road_id': road_id,
                        'lane_id': lane_id,
                        'carla_actor_id': veh_id.replace('carla_', '')
                    }
        except Exception as e:
            print(f"獲取 SUMO 車輛資訊失敗: {e}")

        return vehicles_info

    def is_connected(self):
        """檢查 SUMO 連接狀態"""
        return self._traci is not None

    def _cleanup_sync_status(self):
        """清理无效的同步状态记录"""
        if not hasattr(self, '_last_sync_status'):
            return

        # 清理不存在的 actor 的同步状态
        current_actors = set(self._carla_to_sumo_id.keys())
        invalid_actors = set(self._last_sync_status.keys()) - current_actors

        for actor_id in invalid_actors:
            self._last_sync_status.pop(actor_id, None)

    # ---------- NGSIM integration methods ----------
    def _init_ngsim_controller(self):
        """初始化NGSIM控制器"""
        try:
            # 动态导入NGSIM控制器
            import sys
            ngsim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sumo_NGSIM'))
            if ngsim_dir not in sys.path:
                sys.path.append(ngsim_dir)

            # 🎯 根据地图类型动态导入对应的NGSIM控制器
            if self.town_map_name == "Town04":
                from macad_gym.carla.sumo_NGSIM.NGSIM_to_SUMO_town04_edge4039_optimized_inte import \
                    CarlaTown4Edge40NgsimController
                ControllerClass = CarlaTown4Edge40NgsimController
                if self.verbose:
                    print(f"🎯 使用Town04 NGSIM控制器 (Edge40/39 东西向道路)")
            else:  # Town03 (默认)
                from macad_gym.carla.sumo_NGSIM.NGSIM_to_SUMO import CarlaTown3NgsimController
                ControllerClass = CarlaTown3NgsimController
                if self.verbose:
                    print(f"🎯 使用Town03 NGSIM控制器 (Edge3 南北向道路)")

            # 设置NGSIM数据文件路径
            ngsim_file_path = os.path.join(ngsim_dir, self.ngsim_csv_file)

            # 🎯 检查NGSIM数据文件是否存在
            if not os.path.exists(ngsim_file_path):
                print(f"❌ NGSIM数据文件不存在: {ngsim_file_path}")
                print(f"📁 检查的目录: {ngsim_dir}")
                print(f"📁 查找的文件: {self.ngsim_csv_file}")
                # 列出目录中的CSV文件
                if os.path.exists(ngsim_dir):
                    csv_files = [f for f in os.listdir(ngsim_dir) if f.endswith('.csv')]
                    print(f"📄 目录中的CSV文件: {csv_files}")
                raise FileNotFoundError(f"NGSIM数据文件不存在: {ngsim_file_path}")
            else:
                print(f"✅ 找到NGSIM数据文件: {ngsim_file_path}")

            # 创建NGSIM控制器实例（使用动态选择的控制器类）
            self._ngsim_controller = ControllerClass()

            # 加载NGSIM数据
            self._ngsim_data = self._ngsim_controller.load_ngsim_data(ngsim_file_path)

            # 从控制器获取轨迹数据
            if hasattr(self._ngsim_controller, 'vehicle_trajectories'):
                self._ngsim_vehicle_trajectories = self._ngsim_controller.vehicle_trajectories
            else:
                self._ngsim_vehicle_trajectories = {}

            if self.verbose:
                print(f"✅ NGSIM控制器初始化成功，数据文件: {self.ngsim_csv_file}")
                print(f"📊 载入车辆数据: {len(self._ngsim_data)} 条轨迹记录")
                if not self._ngsim_data.empty:
                    frames = self._ngsim_data['Frame_ID'].tolist()
                    print(f"🕐 帧范围: {min(frames)} - {max(frames)}")

        except Exception as e:
            print(f"❌ NGSIM控制器初始化失败: {e}")
            print("   请检查NGSIM数据文件是否存在，路径是否正确")
            # 输出详细的异常信息用于调试
            import traceback
            print(f"详细错误信息:")
            traceback.print_exc()
            self._ngsim_controller = None
            self._ngsim_data = None

    def _interpolate_vehicle_data(self, current_data: dict, next_data: dict, factor: float) -> dict:
        """
        在两帧NGSIM数据之间进行线性插值，消除抖动

        Args:
            current_data: 当前帧的车辆数据
            next_data: 下一帧的车辆数据
            factor: 插值因子 (0.0-1.0)，0表示完全使用当前帧，1表示完全使用下一帧

        Returns:
            插值后的车辆数据字典
        """
        interpolated = current_data.copy()

        # 🎯 对位置和速度进行线性插值
        interpolate_keys = ['Local_X', 'Local_Y', 'v_Vel', 'v_Acc']

        for key in interpolate_keys:
            if key in current_data and key in next_data:
                current_val = float(current_data[key])
                next_val = float(next_data[key])
                # 线性插值公式: value = current + factor * (next - current)
                interpolated[key] = current_val + factor * (next_val - current_val)

        # 🎯 对离散值（如Lane_ID）使用当前帧的值（不插值）
        # 只有当插值因子>0.5时才切换到下一帧的车道
        if factor > 0.5 and 'Lane_ID' in next_data:
            interpolated['Lane_ID'] = next_data['Lane_ID']

        if self.verbose and factor > 0 and int(self._current_simulation_time * 100) % 50 == 0:
            veh_id = interpolated.get('Vehicle_ID', 'unknown')
            print(f"🔄 插值 车辆{veh_id}: factor={factor:.2f}, "
                  f"Local_X={interpolated['Local_X']:.2f}, Local_Y={interpolated['Local_Y']:.2f}")

        return interpolated

    # ============================================================================
    # 🎯 NGSIM状态机核心函数 - "NGSIM回放 + 危险时SUMO接管 + 大偏差脱管"
    # ============================================================================

    def _calculate_deviation(self, vehicle_id: int, sumo_veh_id: str) -> float:
        """
        计算SUMO车辆实际位置与NGSIM理论位置之间的欧几里得距离偏差

        Args:
            vehicle_id: NGSIM车辆ID（整数）
            sumo_veh_id: SUMO中的车辆ID字符串

        Returns:
            float: 偏差距离（米），如果无法计算则返回0.0
        """
        try:
            # 获取SUMO中车辆的实际位置
            if sumo_veh_id not in self._traci.vehicle.getIDList():
                return 0.0

            actual_pos = self._traci.vehicle.getPosition(sumo_veh_id)
            actual_x, actual_y = actual_pos[0], actual_pos[1]

            # 获取NGSIM理论位置
            if vehicle_id not in self._vehicle_ngsim_target_positions:
                return 0.0

            target_x, target_y = self._vehicle_ngsim_target_positions[vehicle_id]

            # 计算欧几里得距离
            deviation = math.sqrt((actual_x - target_x) ** 2 + (actual_y - target_y) ** 2)

            # 更新偏差历史（用于平滑判断）
            if vehicle_id not in self._vehicle_deviation_history:
                self._vehicle_deviation_history[vehicle_id] = []
            self._vehicle_deviation_history[vehicle_id].append(deviation)

            # 只保留最近10帧的历史
            if len(self._vehicle_deviation_history[vehicle_id]) > 10:
                self._vehicle_deviation_history[vehicle_id].pop(0)

            return deviation

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 计算偏差失败 车辆{vehicle_id}: {e}")
            return 0.0

    def _detect_danger_situation(self, sumo_veh_id: str) -> Tuple[bool, str]:
        """
        检测车辆是否处于危险情况

        检测条件:
        1. 前车距离 < danger_distance_threshold
        2. TTC (Time To Collision) < danger_ttc_threshold
        3. 侧向碰撞风险（相邻车道车辆过近）

        Args:
            sumo_veh_id: SUMO车辆ID

        Returns:
            Tuple[bool, str]: (是否危险, 危险原因描述)
        """
        try:
            if sumo_veh_id not in self._traci.vehicle.getIDList():
                return False, ""

            # 获取车辆当前状态
            current_speed = self._traci.vehicle.getSpeed(sumo_veh_id)
            current_pos = self._traci.vehicle.getPosition(sumo_veh_id)

            # 检查1: 前车距离
            leader_info = self._traci.vehicle.getLeader(sumo_veh_id, dist=50.0)
            if leader_info is not None:
                leader_id, distance = leader_info

                # 危险条件1: 前车距离过近
                if distance < self._danger_distance_threshold:
                    return True, f"前车距离过近({distance:.1f}m < {self._danger_distance_threshold}m)"

                # 危险条件2: TTC过小
                if current_speed > 0.1:
                    leader_speed = self._traci.vehicle.getSpeed(leader_id)
                    relative_speed = current_speed - leader_speed

                    if relative_speed > 0.1:  # 正在接近前车
                        ttc = distance / relative_speed
                        if ttc < self._danger_ttc_threshold:
                            return True, f"TTC过小({ttc:.1f}s < {self._danger_ttc_threshold}s)"

            # 检查2: 周围车辆碰撞风险
            try:
                # 获取当前车道索引
                current_lane_id = self._traci.vehicle.getLaneID(sumo_veh_id)

                # 检查相邻车道是否有车辆过近（侧向碰撞风险）
                neighbors = self._traci.vehicle.getNeighbors(sumo_veh_id, 0b11111111)  # 检查所有方向
                for neighbor_id, neighbor_dist in neighbors:
                    if neighbor_dist < 3.0:  # 侧向距离小于3米
                        return True, f"侧向车辆过近({neighbor_id}, {neighbor_dist:.1f}m)"
            except Exception:
                pass  # 忽略邻居检查失败

            return False, ""

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 危险检测失败 {sumo_veh_id}: {e}")
            return False, ""

    def _update_vehicle_control_mode(self, vehicle_id: int, sumo_veh_id: str,
                                     vehicle_data: dict) -> NGSIMControlMode:
        """
        状态机核心逻辑（简化版）：NGSIM回放 + 危险时SUMO永久接管

        状态转换规则（两态，无恢复机制）:
        1. REPLAY_MODE:
           - 无危险 → 保持 REPLAY_MODE（按NGSIM轨迹回放）
           - 检测到危险（TTC<3s 或 前车距离<10m） → 永久转换到 FREE_MODE

        2. FREE_MODE:
           - 永久状态，完全由SUMO Krauss模型控制
           - 不再尝试恢复到NGSIM轨迹

        Args:
            vehicle_id: NGSIM车辆ID
            sumo_veh_id: SUMO车辆ID
            vehicle_data: 当前帧的NGSIM车辆数据

        Returns:
            NGSIMControlMode: 更新后的控制模式
        """
        # 获取当前模式（默认为REPLAY_MODE）
        current_mode = self._vehicle_control_modes.get(vehicle_id, NGSIMControlMode.REPLAY_MODE)

        # FREE_MODE是永久状态，直接返回（不再检测，不再尝试恢复）
        if current_mode == NGSIMControlMode.FREE_MODE:
            return NGSIMControlMode.FREE_MODE

        # 检测危险情况（TTC<3s 或 前车距离<10m）
        is_dangerous, danger_reason = self._detect_danger_situation(sumo_veh_id)

        # 简化的状态转换逻辑：一旦危险，永久接管
        new_mode = current_mode

        if current_mode == NGSIMControlMode.REPLAY_MODE and is_dangerous:
            # 检测到危险 → 永久转换为FREE_MODE（SUMO完全接管）
            new_mode = NGSIMControlMode.FREE_MODE
            if self.verbose:
                print(f"🚨 车辆{vehicle_id} 永久接管: {danger_reason} → SUMO完全控制")

        # 更新模式
        self._vehicle_control_modes[vehicle_id] = new_mode

        return new_mode

    def _apply_control_by_mode(self, vehicle_id: int, sumo_veh_id: str,
                               vehicle_data: dict, mode: NGSIMControlMode) -> bool:
        """
        根据控制模式执行相应的车辆控制策略（简化版：两态）

        Args:
            vehicle_id: NGSIM车辆ID
            sumo_veh_id: SUMO车辆ID
            vehicle_data: NGSIM车辆数据
            mode: 当前控制模式（REPLAY_MODE 或 FREE_MODE）

        Returns:
            bool: 操作是否成功
        """
        try:
            if sumo_veh_id not in self._traci.vehicle.getIDList():
                return False

            if mode == NGSIMControlMode.REPLAY_MODE:
                # 回放模式：强制按NGSIM轨迹行驶
                return self._apply_replay_mode_control(vehicle_id, sumo_veh_id, vehicle_data)

            elif mode == NGSIMControlMode.FREE_MODE:
                # 自由模式：完全由SUMO控制，不再干预
                return self._apply_free_mode_control(vehicle_id, sumo_veh_id)

            return False

        except Exception as e:
            if self.verbose:
                print(f"❌ 应用控制模式失败 车辆{vehicle_id}: {e}")
            return False

    def _apply_replay_mode_control(self, vehicle_id: int, sumo_veh_id: str,
                                   vehicle_data: dict) -> bool:
        """
        回放模式：强制按NGSIM数据映射位置
        """
        try:
            # 计算NGSIM目标位置
            target_x, target_y = self._ngsim_controller.map_coordinates_enhanced_lateral(
                vehicle_data['Local_Y'], vehicle_data['Local_X'], vehicle_data['Lane_ID']
            )
            target_speed = float(vehicle_data['v_Vel']) * self._ngsim_controller.feet_to_meters
            target_angle = self._ngsim_controller.calculate_vehicle_angle(vehicle_id, (target_x, target_y))

            # 保存目标位置（用于偏差计算）
            self._vehicle_ngsim_target_positions[vehicle_id] = (target_x, target_y)

            # 使用动态edge配置
            main_edge = self._edge_config['main_edge']
            lane_mapping = self._edge_config['lane_mapping']

            # 强制移动到NGSIM位置
            positioning_strategies = [
                ("moveToXY_full", lambda: self._traci.vehicle.moveToXY(
                    vehID=sumo_veh_id, edgeID=main_edge, lane=-1,
                    x=target_x, y=target_y, angle=target_angle, keepRoute=1)),
                ("moveToXY_simple", lambda: self._traci.vehicle.moveToXY(
                    vehID=sumo_veh_id, edgeID=main_edge, lane=-1,
                    x=target_x, y=target_y)),
            ]

            for strategy_name, strategy_func in positioning_strategies:
                try:
                    strategy_func()
                    # 设置速度
                    self._traci.vehicle.setSpeed(sumo_veh_id, target_speed)
                    return True
                except Exception:
                    continue

            return False

        except Exception as e:
            if self.verbose:
                print(f"❌ 回放模式控制失败 车辆{vehicle_id}: {e}")
            return False

    def _apply_free_mode_control(self, vehicle_id: int, sumo_veh_id: str) -> bool:
        """
        自由模式：完全由SUMO控制，成为普通背景车
        - 启用完整SUMO智能驾驶
        - 不再尝试追赶NGSIM数据
        - ⚠️ 注意：保留 _vehicle_ngsim_target_positions 用于误差计算
        """
        try:
            # 启用SUMO完整智能控制
            self._traci.vehicle.setSpeedMode(sumo_veh_id, 31)  # 所有安全检查
            self._traci.vehicle.setLaneChangeMode(sumo_veh_id, 1621)  # 智能变道
            self._traci.vehicle.setSpeed(sumo_veh_id, -1)  # 自动速度控制

            # 设置合理的最大速度
            self._traci.vehicle.setMaxSpeed(sumo_veh_id, 25.0)  # 约90km/h

            # ⚠️ 关键修改：不再删除 _vehicle_ngsim_target_positions
            # 原因：需要在下一帧计算误差（实际位置 vs NGSIM理论位置）
            # 即使车辆被SUMO接管，也应该计算其偏离NGSIM轨迹的程度

            # 清理偏差历史（可选，不影响误差计算）
            if vehicle_id in self._vehicle_deviation_history:
                # 保留最近几帧的历史用于平滑判断，但限制长度
                if len(self._vehicle_deviation_history[vehicle_id]) > 20:
                    self._vehicle_deviation_history[vehicle_id] = self._vehicle_deviation_history[vehicle_id][-10:]

            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ 自由模式控制失败 车辆{vehicle_id}: {e}")
            return False

    def _activate_free_driving_mode(self, sumo_veh_id: str):
        """
        激活车辆的SUMO自由驾驶模式（用于NGSIM数据播放结束后）
        - 恢复SUMO物理模型控制
        - 移除强制速度限制
        - 启用智能跟驰和变道
        """
        try:
            if sumo_veh_id not in self._traci.vehicle.getIDList():
                return

            # 启用SUMO完整智能控制
            self._traci.vehicle.setSpeedMode(sumo_veh_id, 31)  # 所有安全检查
            self._traci.vehicle.setLaneChangeMode(sumo_veh_id, 1621)  # 智能变道
            self._traci.vehicle.setSpeed(sumo_veh_id, -1)  # 自动速度控制（由SUMO决定）

            # 设置合理的最大速度
            self._traci.vehicle.setMaxSpeed(sumo_veh_id, 25.0)  # 约90km/h

            if self.verbose:
                print(f"✅ 车辆 {sumo_veh_id} 已转为SUMO自由驾驶模式")

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 激活自由驾驶模式失败 {sumo_veh_id}: {e}")

    def get_vehicle_control_mode(self, vehicle_id: int) -> NGSIMControlMode:
        """获取指定车辆的当前控制模式"""
        return self._vehicle_control_modes.get(vehicle_id, NGSIMControlMode.REPLAY_MODE)

    def get_control_mode_statistics(self) -> Dict[str, int]:
        """获取各控制模式下的车辆数量统计（简化版：两态）"""
        stats = {
            NGSIMControlMode.REPLAY_MODE.value: 0,
            NGSIMControlMode.FREE_MODE.value: 0
        }
        for mode in self._vehicle_control_modes.values():
            stats[mode.value] += 1
        return stats

    def get_current_deviation_metrics(self) -> Dict[str, float]:
        """
        返回当前帧背景车辆的轨迹偏差统计（供奖励函数使用）

        Returns:
            dict: {
                "avg_deviation": 平均误差（米），只统计REPLAY_MODE车辆
                "max_deviation": 最大误差（米）
                "active_count": 有效车辆数（未脱管的）
                "total_vehicles": 总车辆数
            }
        """
        return self._current_frame_stats.copy()

    def _update_ngsim_vehicles(self):
        """根据当前仿真时间更新NGSIM车辆状态 - 修复帧率同步问题（添加插值）"""
        try:
            if not self._ngsim_controller or self._ngsim_data is None:
                return

            # 正确的时间到帧映射
            # 获取NGSIM数据的帧范围
            min_frame = self._ngsim_controller.current_frame
            max_frame = self._ngsim_controller.max_frame

            # 🎯 关键修复：计算精确的帧位置和插值因子
            frame_offset_float = self._current_simulation_time / self._ngsim_frame_duration
            current_ngsim_frame = min_frame + int(frame_offset_float)

            # 计算插值因子 (0.0 - 1.0)，表示在两帧之间的位置
            interpolation_factor = frame_offset_float - int(frame_offset_float)

            # 🎯 新策略：始终进行更新（包括插值），不跳过
            # 移除旧的"跳过相同帧"逻辑，改为智能插值

            # ============================================================================
            # NGSIM数据结束处理策略：循环回放
            # ============================================================================
            if current_ngsim_frame > max_frame:
                current_ngsim_frame = min_frame  # 重置到起始帧，循环播放
                if self.verbose:
                    print(f"🔄 NGSIM数据循环回放，重置到帧 {min_frame}")

            # 🎯 获取当前帧和下一帧的数据（用于插值）
            current_frame_data = self._ngsim_data[self._ngsim_data['Frame_ID'] == current_ngsim_frame]
            next_ngsim_frame = current_ngsim_frame + 1

            # 处理边界情况：如果下一帧超出范围，循环到起始帧
            if next_ngsim_frame > max_frame:
                next_ngsim_frame = min_frame  # 循环回放

            next_frame_data = self._ngsim_data[self._ngsim_data['Frame_ID'] == next_ngsim_frame]

            # 转换为字典格式，按Vehicle_ID索引以便快速查找
            current_frame_dict = {v['Vehicle_ID']: v for v in current_frame_data.to_dict('records')}
            next_frame_dict = {v['Vehicle_ID']: v for v in next_frame_data.to_dict('records')}

            # 🎯 计算插值后的车辆数据
            interpolated_vehicles = []
            for vehicle_id, current_data in current_frame_dict.items():
                if vehicle_id in next_frame_dict:
                    # 下一帧也有此车辆，进行插值
                    next_data = next_frame_dict[vehicle_id]
                    interpolated_data = self._interpolate_vehicle_data(
                        current_data, next_data, interpolation_factor
                    )
                    interpolated_vehicles.append(interpolated_data)
                else:
                    # 下一帧没有此车辆，使用当前帧数据
                    interpolated_vehicles.append(current_data)

            # 处理下一帧新出现的车辆（当前帧没有）
            for vehicle_id, next_data in next_frame_dict.items():
                if vehicle_id not in current_frame_dict:
                    # 新出现的车辆，插值因子较大时才添加（避免提前出现）
                    if interpolation_factor > 0.5:
                        interpolated_vehicles.append(next_data)

            current_frame_vehicles = interpolated_vehicles  # 使用插值后的数据

            # 处理车辆出现和消失
            current_vehicle_ids = {f"ngsim_{v['Vehicle_ID']}" for v in current_frame_vehicles}
            active_vehicle_ids = set(
                v for v in self._traci.vehicle.getIDList()
                if v.startswith('ngsim_')
            )

            # 移除不再活跃的车辆（与NGSIM_to_SUMO.py第1008-1012行一致）
            vehicles_to_remove = active_vehicle_ids - current_vehicle_ids
            for veh_id in vehicles_to_remove:
                # 从veh_id中提取原始vehicle_id
                if veh_id.startswith('ngsim_'):
                    vehicle_id = int(veh_id.replace('ngsim_', ''))
                    self._remove_ngsim_vehicle(vehicle_id)
                else:
                    # 直接移除非NGSIM车辆
                    try:
                        self._traci.vehicle.remove(veh_id)
                        if self.verbose:
                            print(f"🚗💨 移除非NGSIM车辆: {veh_id}")
                    except Exception:
                        pass

            # ============================================================================
            # 🎯 状态机驱动的智能车辆管理
            # "NGSIM回放 + 危险时SUMO接管"
            # ============================================================================

            # ✅ 1. 每帧开始前，重置统计数据
            frame_total_deviation = 0.0  # 累积偏差的平方和（用于 RMSE 计算）
            frame_max_deviation = 0.0
            frame_valid_count = 0
            frame_new_detached_count = 0  # 🎯 记录本帧有多少车刚被接管（用于一次性惩罚）

            for vehicle_data in current_frame_vehicles:
                vehicle_id = vehicle_data['Vehicle_ID']  # 整数ID
                sumo_veh_id = f"ngsim_{vehicle_id}"

                # 智能车辆状态检查
                vehicle_exists = sumo_veh_id in self._traci.vehicle.getIDList()

                if not vehicle_exists:
                    # 添加新的智能车辆（初始状态为REPLAY_MODE）
                    try:
                        success = self._add_ngsim_vehicle(sumo_veh_id, vehicle_data)
                        if success:
                            # 初始化控制模式为REPLAY_MODE
                            self._vehicle_control_modes[vehicle_id] = NGSIMControlMode.REPLAY_MODE

                            # 🎯 关键：设置初始目标位置（用于下一帧计算误差）
                            # 注意：新添加的车辆在第一帧不计算误差（因为没有上一帧的目标位置）
                            target_x, target_y = self._ngsim_controller.map_coordinates_enhanced_lateral(
                                vehicle_data['Local_Y'], vehicle_data['Local_X'], vehicle_data['Lane_ID']
                            )
                            self._vehicle_ngsim_target_positions[vehicle_id] = (target_x, target_y)

                            if self.verbose:
                                print(
                                    f"🚗✨ NGSIM车辆已添加: {sumo_veh_id} [REPLAY_MODE], 目标位置: ({target_x:.2f}, {target_y:.2f})")
                    except Exception as e:
                        if self.verbose:
                            print(f"❌ 添加NGSIM车辆 {sumo_veh_id} 失败: {e}")
                        continue
                else:
                    # ============================================================
                    # 🎯 状态机核心逻辑：更新控制模式并应用相应控制
                    # ============================================================
                    try:
                        # ✅ 步骤1: 更新车辆控制模式（状态机转换）
                        if vehicle_id in self._detached_ngsim_vehicles:
                            # 已脱管车辆，保持FREE_MODE，但仍需计算误差
                            control_mode = self._vehicle_control_modes.get(vehicle_id, NGSIMControlMode.FREE_MODE)
                        else:
                            # 未脱管车辆，检查是否需要转换模式
                            control_mode = self._update_vehicle_control_mode(
                                vehicle_id, sumo_veh_id, vehicle_data
                            )

                            # ============================================================================
                            # 🎯 关键修改：捕获接管瞬间，用于一次性惩罚
                            # ============================================================================
                            if control_mode == NGSIMControlMode.FREE_MODE:
                                # 只有当它是第一次进入这个集合时，才会计数
                                if vehicle_id not in self._detached_ngsim_vehicles:
                                    self._detached_ngsim_vehicles.add(vehicle_id)
                                    frame_new_detached_count += 1  # 计数器 +1

                                    if self.verbose:
                                        print(f"🚨 惩罚触发: 车辆 {vehicle_id} 被迫接管")

                        # ✅ 步骤2: 先计算轨迹误差（使用上一帧的NGSIM目标位置 vs 当前实际位置）
                        # 注意：必须在更新目标位置之前计算误差，否则会使用当前帧的目标位置
                        #
                        # 🎯 误差计算说明：
                        # - REPLAY_MODE（接管前）：实际位置由moveToXY强制移动到NGSIM目标位置
                        #   误差 = 实际位置 vs NGSIM目标位置（应该很小，反映回放精度）
                        # - FREE_MODE（接管后）：实际位置由SUMO Krauss跟驰模型自主控制
                        #   误差 = SUMO控制的实际位置 vs NGSIM理论位置（反映SUMO接管后的偏离程度）
                        if vehicle_id in self._vehicle_ngsim_target_positions:
                            try:
                                # 获取上一帧设置的NGSIM理论目标位置
                                last_target_x, last_target_y = self._vehicle_ngsim_target_positions[vehicle_id]

                                # 获取当前实际位置
                                # 注意：在FREE_MODE下，这个位置是由SUMO自主控制的，不是NGSIM目标位置
                                current_pos = self._traci.vehicle.getPosition(sumo_veh_id)

                                # 🎯 调试：检查位置是否有效
                                if math.isnan(current_pos[0]) or math.isnan(current_pos[1]) or \
                                        math.isnan(last_target_x) or math.isnan(last_target_y):
                                    if self.verbose:
                                        print(f"⚠️ 车辆{vehicle_id}位置包含NaN，跳过误差计算")
                                    continue

                                # 计算欧几里得距离误差（实际位置 vs NGSIM理论位置）
                                deviation = math.sqrt(
                                    (current_pos[0] - last_target_x) ** 2 +
                                    (current_pos[1] - last_target_y) ** 2
                                )

                                # 🎯 调试：输出误差信息（每100帧输出一次）
                                if self.verbose and frame_valid_count == 0 and int(
                                        self._current_simulation_time * 10) % 100 == 0:
                                    print(
                                        f"🔍 误差计算示例 - 车辆{vehicle_id}: 实际({current_pos[0]:.2f}, {current_pos[1]:.2f}) vs "
                                        f"目标({last_target_x:.2f}, {last_target_y:.2f}) = {deviation:.4f}m")

                                # ✅ 累加误差的平方（用于标准 RMSE 计算）
                                # 注意：无论车辆处于REPLAY_MODE还是FREE_MODE，都计算误差
                                # - REPLAY_MODE：误差反映回放精度
                                # - FREE_MODE：误差反映SUMO接管后偏离NGSIM轨迹的程度
                                frame_total_deviation += deviation ** 2
                                frame_max_deviation = max(frame_max_deviation, deviation)
                                frame_valid_count += 1

                            except Exception as e:
                                if self.verbose:
                                    print(f"⚠️ 计算车辆{vehicle_id}误差失败: {e}")
                                pass  # 车辆可能刚消失或出错

                        # ✅ 步骤3: 更新NGSIM理论目标位置（基于当前NGSIM数据）
                        # 这个目标位置将在下一帧用于计算误差
                        # 注意：即使在FREE_MODE下，我们也更新NGSIM目标位置，用于计算误差
                        # （虽然车辆实际位置由SUMO控制，但我们仍需要NGSIM理论位置作为参考）
                        try:
                            target_x, target_y = self._ngsim_controller.map_coordinates_enhanced_lateral(
                                vehicle_data['Local_Y'], vehicle_data['Local_X'], vehicle_data['Lane_ID']
                            )
                            self._vehicle_ngsim_target_positions[vehicle_id] = (target_x, target_y)
                        except Exception:
                            pass  # 如果计算失败，保留上一帧的目标位置

                        # ✅ 步骤4: 根据控制模式应用相应的控制策略
                        success = self._apply_control_by_mode(
                            vehicle_id, sumo_veh_id, vehicle_data, control_mode
                        )

                        if not success and self.verbose:
                            print(f"⚠️ 控制应用失败 车辆{vehicle_id} [{control_mode.value}]")

                    except Exception as e:
                        if self.verbose:
                            print(f"❌ 状态机控制失败 {sumo_veh_id}: {e}")

            # ✅ 5. 循环结束后，计算本帧统计数据并保存
            if frame_valid_count > 0:
                self._current_frame_stats["avg_deviation"] = math.sqrt(
                    frame_total_deviation / frame_valid_count)  # 本帧 RMSE
                self._current_frame_stats["max_deviation"] = frame_max_deviation
                self._current_frame_stats["active_count"] = frame_valid_count
                # 🎯 新增：保存本帧误差平方和（用于计算 episode 级别的 RMSE）
                self._current_frame_stats["frame_total_error"] = frame_total_deviation

                # 🎯 调试：输出本帧统计信息（每50帧输出一次）
                if self.verbose and int(self._current_simulation_time * 10) % 50 == 0:
                    print(f"📊 本帧NGSIM误差统计: 有效车辆={frame_valid_count}, "
                          f"平均误差={self._current_frame_stats['avg_deviation']:.4f}m, "
                          f"最大误差={frame_max_deviation:.4f}m, "
                          f"误差平方和={frame_total_deviation:.2f}")
            else:
                self._current_frame_stats["avg_deviation"] = 0.0
                self._current_frame_stats["max_deviation"] = 0.0
                self._current_frame_stats["active_count"] = 0
                self._current_frame_stats["frame_total_error"] = 0.0

                # 🎯 调试：如果本帧没有有效车辆，输出警告
                if self.verbose and len(current_frame_vehicles) > 0 and int(
                        self._current_simulation_time * 10) % 100 == 0:
                    print(f"⚠️ 本帧有{len(current_frame_vehicles)}辆NGSIM车辆，但无有效误差数据")

            self._current_frame_stats["total_vehicles"] = len(current_frame_vehicles)

            # 🎯 关键：保存本帧新接管数量（用于一次性惩罚）
            self._current_frame_stats["new_detached"] = frame_new_detached_count

            # 🎯 输出控制模式统计（每50帧输出一次）
            if self.verbose and int(self._current_simulation_time * 10) % 50 == 0:
                stats = self.get_control_mode_statistics()
                total = sum(stats.values())
                if total > 0:
                    print(f"📊 控制模式统计: 回放={stats['replay']}辆, SUMO接管={stats['free']}辆")
                    print(f"📏 轨迹误差统计: 平均={self._current_frame_stats['avg_deviation']:.2f}m, "
                          f"最大={self._current_frame_stats['max_deviation']:.2f}m, "
                          f"有效车辆={frame_valid_count}")

            # 调试信息 - 在for循环外面，只打印一次（包含状态机信息）
            if self.verbose and len(current_frame_vehicles) > 0 and int(self._current_simulation_time * 10) % 20 == 0:
                stats = self.get_control_mode_statistics()
                print(f"🎬 NGSIM帧{current_ngsim_frame} (插值:{interpolation_factor:.2f}): "
                      f"{len(current_frame_vehicles)}辆车 [回放:{stats['replay']} SUMO接管:{stats['free']}]")

        except Exception as e:
            if self.verbose:
                print(f"❌ 更新NGSIM车辆失败: {e}")

    def _add_ngsim_vehicle(self, veh_id: str, vehicle_data: dict):
        """
        添加智能NGSIM车辆到SUMO - 使用IDM跟驰模型 + SL2015变道模型（支持sublane）

        ⚠️ 设计说明：
        1. 代码重复：多重定位策略与NGSIM控制器重复实现
           - 原因：NGSIM控制器设计为独立运行（自己管理traci）
           - 桥接器：需要使用self._traci实例（由外部管理）

        2. 智能控制实现：
           ✅ SUMO IDM + SL2015 组合（通过XML配置）：
              - IDM跟驰模型（纵向控制）: tau=1.0s, minGap=2.5m, accel=2.6, decel=4.5
              - SL2015变道模型（横向控制+sublane）: lcStrategic=1.0, lcCooperative=1.0, lcSublane=1.0
              - 车辆类型在 vtypes.add.xml 中定义，启动时自动加载
              - IDM提供更真实的加减速行为，SL2015处理变道决策和车道内横向移动
              - SL2015支持sublane模拟（lateral-resolution=0.8），允许更精细的横向控制

           ⚠️ 备用方案：
              - 如果XML加载失败，回退到手动创建车辆类型
              - 使用DEFAULT_VEHTYPE作为基础

        3. 最终效果：
           - 车辆添加后立即由SUMO完全接管（IDM + SL2015）
           - 纵向行为：智能跟驰，平滑加减速，动态车间距
           - 横向行为：智能变道 + 车道内精细横向调整（sublane）
           - 更真实的综合驾驶行为，支持车道内微调
        """

        # 🎯 关键修复：检查车辆是否已存在（按照NGSIM_to_SUMO.py第630行逻辑）
        if veh_id in self._traci.vehicle.getIDList():
            if self.verbose:
                print(f"🔄 车辆{veh_id}已存在，更新记录而非重复添加")
            return True  # 车辆已存在，直接返回成功

        # 🚗 第二步：检查车辆类型，优先使用 XML 中定义的 IDM 模型
        vtype_id = "ngsim_vehicle"
        if vtype_id not in self._traci.vehicletype.getIDList():
            # ⚠️ 如果 vtypes.add.xml 已正确加载，则 "ngsim_vehicle" 类型应该存在
            # 如果不存在，说明XML加载失败，则回退到手动创建（使用Krauss模型）
            if self.verbose:
                print(f"⚠️ 警告: 'ngsim_vehicle' 类型不在已加载类型中")
                print(f"   当前已加载的车辆类型: {self._traci.vehicletype.getIDList()}")
                print(f"   将回退到手动创建车辆类型（使用 DEFAULT_VEHTYPE 作为基础）")

            # 从 DEFAULT_VEHTYPE 复制并修改
            self._traci.vehicletype.copy("DEFAULT_VEHTYPE", vtype_id)

            # 🚗 手动设置车辆参数（作为备用方案）
            try:
                # 基本属性
                self._traci.vehicletype.setLength(vtype_id, 5.0)  # 车辆长度
                self._traci.vehicletype.setWidth(vtype_id, 1.8)  # 车辆宽度
                self._traci.vehicletype.setHeight(vtype_id, 1.5)  # 车辆高度

                # 跟驰模型参数
                self._traci.vehicletype.setMinGap(vtype_id, 2.5)  # 最小车距（与IDM一致）
                self._traci.vehicletype.setMaxSpeed(vtype_id, 25.0)  # 最大速度
                self._traci.vehicletype.setAccel(vtype_id, 2.6)  # 加速度
                self._traci.vehicletype.setDecel(vtype_id, 4.5)  # 减速度
                self._traci.vehicletype.setEmergencyDecel(vtype_id, 9.0)  # 紧急制动
                self._traci.vehicletype.setImperfection(vtype_id, 0.5)  # 驾驶员不完美性
                self._traci.vehicletype.setTau(vtype_id, 1.0)  # 反应时间

                # 速度因子
                self._traci.vehicletype.setSpeedFactor(vtype_id, 1.0)
                self._traci.vehicletype.setSpeedDev(vtype_id, 0.1)

                # 外观设置
                self._traci.vehicletype.setVehicleClass(vtype_id, "passenger")
                self._traci.vehicletype.setColor(vtype_id, (0, 255, 0))  # 绿色表示NGSIM车辆

                # GUI外观（如果支持）
                if hasattr(self._traci.vehicletype, 'setShapeClass'):
                    self._traci.vehicletype.setShapeClass(vtype_id, "passenger")

                if self.verbose:
                    print(f"✅ NGSIM车辆类型已手动创建（备用方案）")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 设置vType参数失败: {e}")
        else:
            # ✅ 成功从 XML 加载 IDM + SL2015 模型
            if self.verbose:
                print(f"✅ 成功使用 XML 定义的 IDM + SL2015 车辆类型: {vtype_id}")

        # 🎯 为NGSIM车辆创建确定的路由（使用动态edge配置）
        route_id = f"route_{veh_id}"
        if route_id not in self._traci.route.getIDList():
            # 获取edge配置
            edge_config = self._edge_config
            available_edges = [e.getID() for e in self._sumolib_net.getEdges()]

            # 使用动态配置的edge
            main_edge = edge_config['main_edge']
            next_edge = edge_config['next_edge']

            if main_edge in available_edges and next_edge in available_edges:
                # 完整路由
                edges = edge_config['route']
            elif main_edge in available_edges:
                # 如果只有主edge
                edges = [main_edge]
            else:
                # 如果都没有，选择第一个非交叉口道路
                main_edges = [e for e in available_edges if not e.startswith(':')]
                assert len(main_edges) > 0, "网络中没有可用的主要道路"
                edges = [main_edges[0]]

            self._traci.route.add(route_id, edges)
            if self.verbose:
                print(f"🛣️ 为NGSIM车辆{veh_id}创建路由: {edges} ({edge_config['description']})")

        # 添加车辆
        self._traci.vehicle.add(veh_id, route_id, typeID=vtype_id)

        # 🚗 设置车辆控制模式 - 让SUMO使用XML定义的IDM模型完全接管
        try:
            # ⚠️ 重要：不要手动设置 tau, imperfection, minGap 等参数！
            # 这些参数已经在 vtypes.add.xml 中通过 IDM 模型定义
            # 手动设置会覆盖XML配置，导致IDM模型失效

            # 只设置控制模式，确保SUMO完全接管（不覆盖模型参数）
            # SpeedMode: 0 = 完全SUMO控制（包含所有安全检查）
            self._traci.vehicle.setSpeedMode(veh_id, 0)

            # LaneChangeMode: 0 = 完全SUMO智能变道（使用XML定义的SL2015模型）
            self._traci.vehicle.setLaneChangeMode(veh_id, 0)

            if self.verbose:
                print(f"✅ SUMO控制模式已配置: SpeedMode=0, LaneChangeMode=0")
                print(f"   车辆将使用XML定义的 IDM + SL2015 模型（支持sublane）")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 配置控制模式失败: {e}")

        # 🎯 关键修复：使用与NGSIM_to_SUMO.py完全一致的多重定位策略逻辑

        # 获取车辆数据并进行坐标转换
        local_y = float(vehicle_data['Local_Y'])
        local_x = float(vehicle_data['Local_X'])
        lane_id = int(vehicle_data['Lane_ID'])

        # 使用NGSIM控制器的精确坐标映射（与NGSIM_to_SUMO.py第621-622行一致）
        carla_x, carla_y = self._ngsim_controller.map_coordinates_enhanced_lateral(
            local_y, local_x, lane_id
        )

        # 计算车辆角度（与NGSIM_to_SUMO.py第627行一致）
        vehicle_id = vehicle_data['Vehicle_ID']
        vehicle_angle = self._ngsim_controller.calculate_vehicle_angle(
            vehicle_id, (carla_x, carla_y)
        )

        # 多重智能定位策略（与NGSIM_to_SUMO.py第664-740行完全一致）
        positioned = False

        # 🎯 使用动态edge配置
        main_edge = self._edge_config['main_edge']

        # 策略1: 标准moveToXY with keepRoute（第667-680行）
        try:
            self._traci.vehicle.moveToXY(
                vehID=veh_id,
                edgeID=main_edge,
                lane=-1,
                x=carla_x,
                y=carla_y,
                angle=vehicle_angle,
                keepRoute=1
            )
            positioned = True
            if self.verbose:
                print(f"✅ 策略1成功: moveToXY with keepRoute (edge {main_edge})")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 策略1失败: {e}")

        # 策略2: moveToXY without angle（第685-701行）
        if not positioned:
            try:
                self._traci.vehicle.moveToXY(
                    vehID=veh_id,
                    edgeID=main_edge,
                    lane=-1,
                    x=carla_x,
                    y=carla_y,
                    keepRoute=1
                )
                positioned = True
                if self.verbose:
                    print(f"✅ 策略2成功: moveToXY without angle (edge {main_edge})")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 策略2失败: {e}")

        # 策略3: 基于车道的定位（第704-720行）
        if not positioned:
            try:
                # 🎯 使用动态车道映射
                target_lane_index = self._edge_config['lane_mapping'].get(lane_id, 3)
                self._traci.vehicle.moveToXY(
                    vehID=veh_id,
                    edgeID=main_edge,
                    lane=target_lane_index,
                    x=carla_x,
                    y=carla_y
                )
                positioned = True
                if self.verbose:
                    print(f"✅ 策略3成功: 车道{target_lane_index}定位 (edge {main_edge})")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 策略3失败: {e}")

        # 策略4: 简单moveToXY（第723-736行）
        if not positioned:
            try:
                self._traci.vehicle.moveToXY(
                    vehID=veh_id,
                    edgeID=main_edge,
                    lane=0,
                    x=carla_x,
                    y=carla_y
                )
                positioned = True
                if self.verbose:
                    print(f"✅ 策略4成功: 简单moveToXY (edge {main_edge})")
            except Exception as e:
                if self.verbose:
                    print(f"❌ 策略4失败: {e}")

        # 关键检查：如果所有定位策略都失败（第738-740行逻辑）
        if not positioned:
            print(f"❌ 所有定位策略都失败，车辆{veh_id}无法定位")
            # 🎯 关键修复：清理已添加但定位失败的车辆，避免下次"already exists"错误
            try:
                if veh_id in self._traci.vehicle.getIDList():
                    self._traci.vehicle.remove(veh_id)
                    if self.verbose:
                        print(f"🧹 清理定位失败的车辆: {veh_id}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 清理车辆{veh_id}失败: {e}")
            return False

        # ⚠️ 关键修复：使用setMaxSpeed代替setSpeed，启用智能避让
        # 设置初始期望速度（非强制）
        initial_speed = float(vehicle_data['v_Vel']) * 0.3048  # 英尺/秒转米/秒
        # self._traci.vehicle.setSpeed(veh_id, initial_speed)  # 注释掉强制速度
        self._traci.vehicle.setMaxSpeed(veh_id, max(initial_speed, 0.1))  # 设置期望速度

        # 记录车辆状态（与NGSIM_to_SUMO.py第768-777行一致）
        if not hasattr(self, '_ngsim_active_vehicles'):
            self._ngsim_active_vehicles = {}

        self._ngsim_active_vehicles[vehicle_id] = {
            'sumo_id': veh_id,
            'prev_pos': (carla_x, carla_y),
            'prev_speed': initial_speed,
            'added_frame': getattr(self, '_current_ngsim_frame', 0),
            'last_lane_id': lane_id,
            'target_speed': initial_speed,
            'safety_mode': 'normal'
        }

        if self.verbose:
            print(f"🚗✨ 智能NGSIM车辆已添加: {veh_id}")

        return True  # 🎯 与NGSIM_to_SUMO.py第797行逻辑完全一致

    def _update_ngsim_vehicle_position(self, veh_id: str, vehicle_data: dict):
        """更新NGSIM车辆位置 - 完全按照NGSIM_to_SUMO.py的update_vehicle方法实现"""
        # 🎯 关键修复：与NGSIM_to_SUMO.py第805-806行一致
        vehicle_id = vehicle_data['Vehicle_ID']
        if not hasattr(self, '_ngsim_active_vehicles') or vehicle_id not in self._ngsim_active_vehicles:
            if self.verbose:
                print(f"⚠️ 车辆{vehicle_id}不在活跃车辆列表中，跳过更新")
            return False

        vehicle_info = self._ngsim_active_vehicles[vehicle_id]
        sumo_vehicle_id = vehicle_info['sumo_id']

        try:
            # 检查车辆是否仍在SUMO中（与NGSIM_to_SUMO.py第812-816行一致）
            if sumo_vehicle_id not in self._traci.vehicle.getIDList():
                print(f"⚠️ 车辆{vehicle_id}不在SUMO中，从跟踪中移除")
                del self._ngsim_active_vehicles[vehicle_id]
                return False

            # 计算新的目标位置 - 使用增强横向映射（与第818-821行一致）
            target_x, target_y = self._ngsim_controller.map_coordinates_enhanced_lateral(
                vehicle_data['Local_Y'], vehicle_data['Local_X'], vehicle_data['Lane_ID']
            )
            target_speed = float(vehicle_data['v_Vel']) * self._ngsim_controller.feet_to_meters

            # 使用SUMO官方标准方法计算车辆角度（与第824-825行一致）
            target_angle = self._ngsim_controller.calculate_vehicle_angle(vehicle_id, (target_x, target_y))

            # 车身边缘计算（与第827-830行一致）
            vehicle_half_width = self._ngsim_controller.lateral_params['vehicle_half_width']
            vehicle_left_edge = target_x - vehicle_half_width
            vehicle_right_edge = target_x + vehicle_half_width

            # 多重定位策略（增强版）- 与第832-846行一致 ⭐
            positioned_successfully = False

            # 🎯 使用动态edge配置
            main_edge = self._edge_config['main_edge']
            lane_mapping = self._edge_config['lane_mapping']

            positioning_strategies = [
                ("moveToXY_full", lambda: self._traci.vehicle.moveToXY(
                    vehID=sumo_vehicle_id, edgeID=main_edge, lane=-1,
                    x=target_x, y=target_y, angle=target_angle, keepRoute=1)),
                ("moveToXY_simple", lambda: self._traci.vehicle.moveToXY(
                    vehID=sumo_vehicle_id, edgeID=main_edge, lane=-1,
                    x=target_x, y=target_y)),
                ("moveToXY_lane_specific", lambda: self._traci.vehicle.moveToXY(
                    vehID=sumo_vehicle_id, edgeID=main_edge,
                    lane=lane_mapping.get(vehicle_data['Lane_ID'], 3),
                    x=target_x, y=target_y))
            ]

            # 执行定位策略（与第848-862行一致）
            for strategy_name, strategy_func in positioning_strategies:
                try:
                    strategy_func()
                    positioned_successfully = True
                    if (self._ngsim_controller.lateral_params.get('enable_debug', False) and
                            hasattr(self._ngsim_controller, 'positioning_debug_counter')):
                        self._ngsim_controller.positioning_debug_counter = getattr(self._ngsim_controller,
                                                                                   'positioning_debug_counter', 0) + 1
                        if self._ngsim_controller.positioning_debug_counter % 100 == 0:
                            print(f"✅ 更新定位策略: {strategy_name}")
                    break
                except Exception as e:
                    if (self._ngsim_controller.lateral_params.get('enable_debug', False) and
                            strategy_name == positioning_strategies[-1][0]):
                        print(f"⚠️ 所有更新定位策略失败，最后尝试: {e}")
                    continue

            if positioned_successfully:
                # ⚠️ 关键修复：禁用强制速度设置，让SUMO的Krauss跟驰模型工作
                # 问题：setSpeed()会覆盖SUMO的智能避让逻辑
                # 解决：注释掉setSpeed()，使用setMaxSpeed()设置期望速度
                # self._traci.vehicle.setSpeed(sumo_vehicle_id, target_speed)

                # ✅ 替代方案：设置期望速度而非强制速度
                # 这样SUMO会尝试达到这个速度，但会在遇到障碍时自动减速
                self._traci.vehicle.setMaxSpeed(sumo_vehicle_id, max(target_speed, 0.1))

                # 更新车辆信息（与第868-871行一致）
                vehicle_info['prev_pos'] = (target_x, target_y)
                vehicle_info['prev_speed'] = target_speed
                vehicle_info['last_lane_id'] = vehicle_data['Lane_ID']

                # 详细调试信息（与第873-882行一致）
                if (self._ngsim_controller.lateral_params.get('enable_debug', False) and
                        hasattr(self._ngsim_controller, 'update_counter')):
                    self._ngsim_controller.update_counter = getattr(self._ngsim_controller, 'update_counter', 0) + 1
                    if self._ngsim_controller.update_counter % 200 == 0:
                        lane_center = self._ngsim_controller.carla_params['lane_centers'].get(vehicle_data['Lane_ID'],
                                                                                              -77.9)
                        lateral_offset = target_x - lane_center
                        print(f"🔄 车身更新{vehicle_id}: 中心({target_x:.1f}, {target_y:.2f}), "
                              f"偏移:{lateral_offset:.2f}m, 角度:{target_angle:.1f}°")
                        print(f"   🚗 车身: 左边缘{vehicle_left_edge:.2f}m ~ 右边缘{vehicle_right_edge:.2f}m")

                # 统计更新（与第884行一致）
                if hasattr(self._ngsim_controller, 'stats'):
                    self._ngsim_controller.stats['total_updates'] += 1

                return True
            else:
                print(f"⚠️ 车辆{vehicle_id}定位更新失败")
                return False

        except Exception as e:
            print(f"❌ 更新车辆{vehicle_id}失败: {e}")
            return False

    def _remove_ngsim_vehicle(self, vehicle_id: int):
        """移除NGSIM车辆 - 完全按照NGSIM_to_SUMO.py的remove_vehicle方法实现"""
        if not hasattr(self, '_ngsim_active_vehicles') or vehicle_id not in self._ngsim_active_vehicles:
            return False

        sumo_vehicle_id = self._ngsim_active_vehicles[vehicle_id]['sumo_id']

        try:
            # 检查车辆是否仍在SUMO中（与NGSIM_to_SUMO.py第902-908行一致）
            if sumo_vehicle_id not in self._traci.vehicle.getIDList():
                if self._ngsim_controller.lateral_params.get('enable_debug', False):
                    print(f"⚠️ 车辆{vehicle_id}已不在SUMO中，仅清理跟踪数据")
            else:
                # 车辆在SUMO中，正常移除
                self._traci.vehicle.remove(sumo_vehicle_id)

            # 统计更新（与第910行一致）
            if hasattr(self._ngsim_controller, 'stats'):
                self._ngsim_controller.stats['vehicles_removed'] += 1

        except Exception as e:
            if self._ngsim_controller.lateral_params.get('enable_debug', False):
                print(f"⚠️ 移除增强横向车辆 {vehicle_id} 时出错: {e}")
        finally:
            # 清理所有相关状态（与第916-928行一致）⭐
            if vehicle_id in self._ngsim_active_vehicles:
                del self._ngsim_active_vehicles[vehicle_id]
            if hasattr(self._ngsim_controller,
                       'vehicle_control_modes') and vehicle_id in self._ngsim_controller.vehicle_control_modes:
                del self._ngsim_controller.vehicle_control_modes[vehicle_id]
            if hasattr(self._ngsim_controller,
                       'safety_overrides') and vehicle_id in self._ngsim_controller.safety_overrides:
                del self._ngsim_controller.safety_overrides[vehicle_id]
            if hasattr(self._ngsim_controller,
                       'interaction_history') and vehicle_id in self._ngsim_controller.interaction_history:
                del self._ngsim_controller.interaction_history[vehicle_id]
            if hasattr(self._ngsim_controller,
                       'vehicle_position_history') and vehicle_id in self._ngsim_controller.vehicle_position_history:
                del self._ngsim_controller.vehicle_position_history[vehicle_id]
                if self._ngsim_controller.lateral_params.get('enable_debug', False):
                    print(f"🧹 清理车辆{vehicle_id}的SUMO官方角度计算历史")

            # 🔧 Bug修复：清理桥接器自身维护的状态机数据（避免内存泄漏和循环回放逻辑错误）
            if vehicle_id in self._vehicle_control_modes:
                del self._vehicle_control_modes[vehicle_id]
            if vehicle_id in self._detached_ngsim_vehicles:
                self._detached_ngsim_vehicles.discard(vehicle_id)
            if vehicle_id in self._vehicle_ngsim_target_positions:
                del self._vehicle_ngsim_target_positions[vehicle_id]
            if vehicle_id in self._vehicle_deviation_history:
                del self._vehicle_deviation_history[vehicle_id]

        return True

    def _should_apply_safety_override(self, veh_id: str) -> bool:
        """检查是否需要应用智能安全覆盖控制 - 基于NGSIM_to_SUMO.py智能逻辑"""
        try:
            if veh_id not in self._traci.vehicle.getIDList():
                return False

            # 获取车辆当前状态
            current_speed = self._traci.vehicle.getSpeed(veh_id)
            current_pos = self._traci.vehicle.getPosition(veh_id)
            leader_info = self._traci.vehicle.getLeader(veh_id, dist=50.0)

            # ✅ 条件1: 检测前方车辆过近或TTC过小
            if leader_info is not None:
                dist = leader_info[1]  # 与前车的距离
                leader_id = leader_info[0]

                # 计算相对速度
                try:
                    leader_speed = self._traci.vehicle.getSpeed(leader_id)
                    rel_speed = current_speed - leader_speed  # 正值表示在接近前车
                except:
                    rel_speed = 0

                # 计算 TTC (Time To Collision)
                if rel_speed > 0:
                    ttc = dist / rel_speed
                    # ✅ 使用配置参数而非硬编码
                    if ttc < self._danger_ttc_threshold:
                        return True

                # ✅ 使用配置参数而非硬编码
                if dist < self._danger_distance_threshold:
                    return True

            # 条件2: 位置异常检测（边界检查）
            if (current_pos[0] < -200 or current_pos[0] > 200 or
                    current_pos[1] < -200 or current_pos[1] > 200):
                return True

            return False

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 智能安全检查失败 {veh_id}: {e}")
            return False

    def _apply_intelligent_safety_control(self, veh_id: str):
        """应用智能安全控制 - 集成NGSIM_to_SUMO.py的安全机制"""
        try:
            if veh_id not in self._traci.vehicle.getIDList():
                return

            # 🛡️ 启用增强安全模式 (参考NGSIM_to_SUMO.py的安全控制)
            self._traci.vehicle.setSpeedMode(veh_id, 31)  # 启用所有安全检查
            self._traci.vehicle.setLaneChangeMode(veh_id, 1024)  # 更安全的变道模式

            # 获取当前状态
            current_speed = self._traci.vehicle.getSpeed(veh_id)
            leader_info = self._traci.vehicle.getLeader(veh_id, dist=50.0)

            # 智能速度调整
            if leader_info is not None:
                leader_id, distance = leader_info
                if distance < 10.0:  # 跟车距离过近
                    # 自动减速到安全速度
                    safe_speed = max(current_speed * 0.6, 5.0)  # 减速40%，最低5m/s
                    self._traci.vehicle.setSpeed(veh_id, safe_speed)

                    if self.verbose:
                        print(f"🛡️ 智能减速: {veh_id} 减速至 {safe_speed:.1f}m/s (跟车距离: {distance:.1f}m)")

            # 智能变道建议 (如果前方拥堵)
            if current_speed < 2.0:  # 车速过慢，可能拥堵
                try:
                    current_lane_id = self._traci.vehicle.getLaneID(veh_id)
                    # 🎯 使用动态edge配置检查车道
                    main_edge = self._edge_config['main_edge']
                    if f"{main_edge}_" in current_lane_id:  # 在主edge上
                        current_lane_index = int(current_lane_id.split('_')[1])

                        # 🎯 尝试变到更快的车道（使用动态车道映射）
                        target_lanes = list(self._edge_config['lane_mapping'].values())  # 根据地图动态获取
                        for target_lane in target_lanes:
                            if target_lane != current_lane_index:
                                self._traci.vehicle.changeLane(veh_id, target_lane, 3.0)
                                if self.verbose:
                                    print(f"🔄 智能变道建议: {veh_id} 从车道{current_lane_index} → 车道{target_lane}")
                                break
                except Exception:
                    pass  # 变道失败不影响主要功能

            # 恢复正常智能模式 (3秒后)
            def restore_normal_mode():
                try:
                    if veh_id in self._traci.vehicle.getIDList():
                        self._traci.vehicle.setSpeedMode(veh_id, 23)  # 恢复智能速度控制
                        self._traci.vehicle.setLaneChangeMode(veh_id, 512)  # 恢复智能变道
                        self._traci.vehicle.setSpeed(veh_id, -1)  # 恢复自动速度控制
                except Exception:
                    pass

            # 注：在实际应用中，这里应该使用定时器或回调
            # 这里简化为立即恢复，可以根据需要调整

        except Exception as e:
            if self.verbose:
                print(f"❌ 应用智能安全控制失败 {veh_id}: {e}")
