import numpy as np
from pprint import pprint
from macad_gym.carla.Utils import calculate_distance


class Reward(object):
    def __init__(self):
        self.reward = 0.0
        self.prev = None
        self.curr = None

    def compute_reward(self, prev_measurement, curr_measurement, ego_prev_measurement, ego_curr_measurement, flag):
        self.prev = prev_measurement
        self.curr = curr_measurement
        self.ego_prev = ego_prev_measurement
        self.ego_curr = ego_curr_measurement
        # pprint (curr_measurement["next_command"])
        if flag == "custom":
            return 0
            # return self.egoReward()
        elif flag == "advrs":
            return self.advCollisionRewardv5()
        elif flag == "advrs1":
            return self.advCollisionRewardv1()
        elif flag == "advrs4":
            return self.advCollisionRewardv4()
        elif flag == "none":
            return self._None_()
        elif flag == "corl2017":
            return self.compute_reward_corl2017()
        elif flag == "madarl":
            return self.MADARL()
        elif flag == "myRewardv1":
            return self.myRewardv1()

    def _None_(self):
        return 0.0

    '''
    advCollisionReward() version1

    only consider the situation that NPCs collide with ego vehicle,then give NPC 
    
    if ego collides to npc_vehicle_i, then npc_vehicle_i got a collisionReward
    elif ego collides to other_actors, all npc_vehicle got a global_reward
    elif ego offline or offroad, all npc_vehicle got a global_reward
    '''

    def advCollisionRewardv1(self):
        self.reward = 0.0
        distance = self.prev["distance_to_ego"] - self.curr["distance_to_ego"]
        self.reward += np.clip(distance * 0.5, -5, 5)
        speed_change = self.curr["forward_speed"] - self.prev["forward_speed"]
        if self.curr["forward_speed"] < 2:
            self.reward -= 0.5
        elif self.curr["forward_speed"] < 10:
            self.reward += np.clip(speed_change, -1, 1)
        # self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) * 0.01

        new_collision_to_ego = (
                self.curr["collision_to_ego"] - self.prev["collision_to_ego"]
        )
        new_collision_to_others = (
                self.curr["intersection_offroad"] + self.curr["collision_vehicles"] + self.curr["collision_other"] +
                self.curr["collision_pedestrians"]
                - self.prev["intersection_offroad"] - self.prev["collision_vehicles"] - self.prev["collision_other"] -
                self.prev["collision_pedestrians"]
        )
        new_own_offlane = (  # offline和offroad区分开
                self.curr["intersection_otherlane"]
                - self.prev["intersection_otherlane"]
        )
        if new_collision_to_ego:
            self.reward += new_collision_to_ego * 100
        # if new_ego_offlane + new_ego_collision_to_others:
        #     self.reward += new_ego_offlane * 0.5 + new_ego_collision_to_others
        if new_collision_to_others:
            self.reward -= new_collision_to_others * 5
        if new_own_offlane:
            self.reward += new_own_offlane * 0.5

        return self.reward

    '''
    advCollisionReward() version2 

    add distance_reward, consider the distance to ego_vehicle 

    only consider the situation that NPCs collide with ego vehicle,then give NPC 
    
    if ego collides to npc_vehicle_i, then npc_vehicle_i got a collisionReward
    elif ego collides to other_actors, all npc_vehicle got a global_reward
    elif ego offline or offroad, all npc_vehicle got a global_reward
    '''

    def myRewardv1(self):
        self.reward = 0.0
        # 检查车辆是否处于碰撞状态
        has_collision = (self.curr["collision_vehicles"] > 0 or
                         self.curr["collision_pedestrians"] > 0 or
                         self.curr["collision_other"] > 0)
        # 鼓励保持在ego身边 狩猎奖励 (Proximity Reward)
        dist_to_ego = self.curr["distance_to_ego"]
        if dist_to_ego < 20.0:
            # 距离越近 奖励越多
            # 诱导car1 2 3对ego产生包围圈
            # 但要限制速度
            # min_speed_threshold = 30.0 / 3.6 # 30 km/h 8.33m/s
            # if self.curr["forward_speed"] >= min_speed_threshold:
            #     self.reward += 0.5 * (1.0 - dist_to_ego / 20.0)
            self.reward += 0.5 * (1.0 - dist_to_ego / 20.0)

        # 碰撞与违规奖励  保持大权重 这是比较重要的稀疏奖励
        new_collision_to_ego = (
                self.curr["collision_to_ego"] - self.prev["collision_to_ego"]
        )
        if new_collision_to_ego:
            self.reward += 100.0

        # 撞到其他的
        new_collision_to_others = (
                self.curr["collision_vehicles"] + self.curr["collision_other"] + self.curr["collision_pedestrians"]
                - self.prev["collision_vehicles"] - self.prev["collision_other"] - self.prev["collision_pedestrians"]
        )
        if new_collision_to_others:
            self.reward -= 20.0

        # # ============================================================================
        # # 🎯 一次性惩罚 (One-off Penalty) - 替代原来的轨迹误差惩罚
        # # ============================================================================
        # # 原理：
        # #   - REPLAY_MODE下车辆严格按NGSIM数据回放，轨迹误差为0
        # #   - 只有切换到FREE_MODE（被SUMO接管）时才产生惩罚
        # #   - 惩罚只在"切换的那一瞬间"发生，之后不再重复惩罚
        # #
        # # 物理含义：
        # #   - 你的攻击行为做出了"危险动作"
        # #   - 导致一辆背景车被迫接管（脱离原始轨迹）
        # #   - 这一瞬间扣你一次分，代表你制造了一次"交通扰动事件"
        # #   - 之后车辆怎么跑、误差多少，都跟你无关了，避免了雪球式惩罚
        # # ============================================================================
        # ngsim_new_detached = self.curr.get("ngsim_new_detached", 0)
        #
        # # 用new_collision_to_ego判断本帧是否攻击成功（而非累计值）
        # # 只有在攻击未成功时才惩罚（攻击成功说明策略有效，不应惩罚）
        # if not new_collision_to_ego and ngsim_new_detached > 0:
        #     # 扣5分
        #     self.reward -= 5.0

        # 计算本帧ego新撞到其他物体
        new_ego_collision_other = (
                self.ego_curr["collision_other"] - self.ego_prev["collision_other"]
        )
        # ego本帧撞到其他物体（非NPC攻击导致）给予一次性惩罚
        if new_ego_collision_other > 0 and not new_collision_to_ego:
            self.reward -= 20.0

        # 速度范围惩罚 (Velocity Penalty)
        # 限制NPC速度在 40-60 km/h 范围内，超出范围给予惩罚
        # 公式：
        #   若 V_t ≤ V_min: R = (V_t - V_min) / (V_max - V_min)  → 负值（太慢惩罚）
        #   若 V_t ≥ V_max: R = (V_t - V_max) / (V_max - V_min)  → 正值（太快惩罚，需减去）
        warmup_steps = 6
        curr_step = self.curr.get("step", 0)
        if curr_step >= warmup_steps and not has_collision:
            v_min = 30.0 / 3.6  # 30 km/h → 8.33 m/s
            v_max = 60.0 / 3.6  # 60 km/h → 16.67 m/s
            cur_speed = self.curr["forward_speed"]
            # print("cur_speed:", cur_speed)
            if cur_speed < v_min:
                # 速度太低，公式结果为负值，直接加即为惩罚
                r_velocity = (cur_speed - v_min) / (v_max - v_min)
                self.reward += 0.4 * r_velocity

            elif cur_speed > v_max:
                # 速度太高，公式结果为正值，需要减去作为惩罚
                r_velocity = (cur_speed - v_max) / (v_max - v_min)
                self.reward -= 0.4 * r_velocity

        return self.reward

    def advCollisionRewardv5(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m
        goal_reward = np.clip(prev_dist - cur_dist, -1.0, 1.0)
        ego_distance_reward = np.clip(self.prev["distance_to_ego"] - self.curr["distance_to_ego"], -1.0, 1.0)

        self.reward += 0.2 * goal_reward
        # if(ego_distance_reward > 0):
        #     self.reward += ego_distance_reward
        # else:
        #     self.reward += goal_reward

        cur_speed = self.curr["forward_speed"]
        if (cur_speed < 1):
            self.reward += 0.5 * (cur_speed - 1) / (10 - 1)
        elif (cur_speed > 10):
            self.reward += 0.5 * (10 - cur_speed) / (10 - 1)

        relative_speed = self.curr["forward_speed"] - self.ego_curr["forward_speed"]
        self.reward += 0.1 * np.clip(relative_speed, 0.0, 5.0)

        new_collision_to_ego = (
                self.curr["collision_to_ego"] - self.prev["collision_to_ego"]
        )
        new_collision_to_others = (
                self.curr["collision_vehicles"] + self.curr["collision_other"] + self.curr["collision_pedestrians"]
                - self.prev["collision_vehicles"] - self.prev["collision_other"] - self.prev["collision_pedestrians"]
        )
        new_own_offlane = (  # offline和offroad区分开
                self.curr["intersection_otherlane"]
                - self.prev["intersection_otherlane"]
        )
        new_ego_offlane = (
                self.ego_curr["intersection_otherlane"]
                - self.ego_prev["intersection_otherlane"]
        )
        new_ego_collision_to_others = (
                self.ego_curr["collision_vehicles"] + self.ego_curr["collision_other"] + self.ego_curr[
            "collision_pedestrians"]
                - self.ego_prev["collision_vehicles"] - self.ego_prev["collision_other"] - self.ego_prev[
                    "collision_pedestrians"]
        )
        if new_collision_to_ego:
            self.reward += new_collision_to_ego * 100
        # if new_ego_offlane + new_ego_collision_to_others:
        #     self.reward += new_ego_offlane * 0.1 + new_ego_collision_to_others*1
        if new_collision_to_others:
            self.reward -= new_collision_to_others * 10
        # if new_own_offlane:
        #     self.reward -= new_own_offlane*0.5
        if new_ego_offlane:
            self.reward += new_ego_offlane * 0.5

        return self.reward

    def advCollisionRewardv4(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m
        goal_reward = np.clip(prev_dist - cur_dist, -1.0, 1.0)
        ego_distance_reward = np.clip(self.prev["distance_to_ego"] - self.curr["distance_to_ego"], -1.0, 1.0)

        self.reward += goal_reward
        # if(ego_distance_reward > 0):
        #     self.reward += ego_distance_reward
        # else:
        #     self.reward += goal_reward

        cur_speed = self.curr["forward_speed"]
        if (cur_speed < 1):
            self.reward += cur_speed - 1
        else:
            self.reward += 0.1 * cur_speed

        new_collision_to_ego = (
                self.curr["collision_to_ego"] - self.prev["collision_to_ego"]
        )
        new_collision_to_others = (
                self.curr["collision_vehicles"] + self.curr["collision_other"] + self.curr["collision_pedestrians"]
                - self.prev["collision_vehicles"] - self.prev["collision_other"] - self.prev["collision_pedestrians"]
        )
        new_own_offlane = (  # offline和offroad区分开
                self.curr["intersection_otherlane"]
                - self.prev["intersection_otherlane"]
        )
        new_ego_offlane = (
                self.ego_curr["intersection_otherlane"]
                - self.ego_prev["intersection_otherlane"]
        )
        new_ego_collision_to_others = (
                self.ego_curr["collision_vehicles"] + self.ego_curr["collision_other"] + self.ego_curr[
            "collision_pedestrians"]
                - self.ego_prev["collision_vehicles"] - self.ego_prev["collision_other"] - self.ego_prev[
                    "collision_pedestrians"]
        )
        if new_collision_to_ego:
            self.reward += new_collision_to_ego * 10
        # if new_ego_offlane + new_ego_collision_to_others:
        #     self.reward += new_ego_offlane * 0.1 + new_ego_collision_to_others*1
        if new_collision_to_others:
            self.reward -= new_collision_to_others * 5
        # if new_own_offlane:
        #     self.reward -= new_own_offlane*0.5

        return self.reward

    def advCollisionRewardv3(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        self.reward += 0.5 * np.clip(prev_dist - cur_dist, -10.0, 10.0)

        # 加速奖励
        self.reward += 0.5 * (
                self.curr["forward_speed"] - self.prev["forward_speed"])

        new_collision_to_ego = (
                self.curr["collision_to_ego"] - self.prev["collision_to_ego"]
        )
        new_collision_to_others = (
                self.curr["collision_vehicles"] + self.curr["collision_other"] + self.curr["collision_pedestrians"]
                - self.prev["collision_vehicles"] - self.prev["collision_other"] - self.prev["collision_pedestrians"]
        )
        new_own_offlane = (  # offline和offroad区分开
                self.curr["intersection_otherlane"]
                - self.prev["intersection_otherlane"]
        )
        new_ego_offlane = (
                self.ego_curr["intersection_otherlane"]
                - self.ego_prev["intersection_otherlane"]
        )
        new_ego_collision_to_others = (
                self.ego_curr["collision_vehicles"] + self.ego_curr["collision_other"] + self.ego_curr[
            "collision_pedestrians"]
                - self.ego_prev["collision_vehicles"] - self.ego_prev["collision_other"] - self.ego_prev[
                    "collision_pedestrians"]
        )
        if new_collision_to_ego:
            self.reward += new_collision_to_ego * 100
        # if new_ego_offlane + new_ego_collision_to_others:
        #     self.reward += new_ego_offlane * 0.1 + new_ego_collision_to_others*1
        if new_collision_to_others:
            self.reward -= new_collision_to_others * 10
        if new_own_offlane:
            self.reward += new_own_offlane * 0.5

        return self.reward

    def advCollisionRewardv2(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        self.reward += np.clip(prev_dist - cur_dist, -1.0, 1.0)
        self.reward += 0.1 * self.curr["forward_speed"]
        # if(self.curr["forward_speed"] < 0.5):
        #     self.reward -= (0.5-self.curr["forward_speed"])
        # elif self.curr["forward_speed"] < 5:
        #     self.reward += np.clip(self.curr["forward_speed"], 0.0, 10.0) * 0.15
        # else:
        #     self.reward -= 0.1
        if (self.prev["distance_to_ego"] - self.curr["distance_to_ego"]) > 0:
            if self.curr["distance_to_ego"] < 2 and self.curr["forward_speed"] > 0:
                self.reward += (2 - self.curr["distance_to_ego"])
            elif self.curr["distance_to_ego"] < 5 and self.curr["forward_speed"] > 0.5:
                self.reward += 0.2
            elif self.curr["distance_to_ego"] < 8 and self.curr["forward_speed"] > 1:
                self.reward += 0.1
        elif (prev_dist - cur_dist) < 0:
            self.reward += np.clip(self.prev["distance_to_ego"] - self.curr["distance_to_ego"], -0.5, 0)

        new_collision_to_ego = (
                self.curr["collision_to_ego"] - self.prev["collision_to_ego"]
        )
        new_collision_to_others = (
                self.curr["collision_vehicles"] + self.curr["collision_other"] + self.curr["collision_pedestrians"]
                - self.prev["collision_vehicles"] - self.prev["collision_other"] - self.prev["collision_pedestrians"]
        )
        new_own_offlane = (  # offline和offroad区分开
                self.curr["intersection_otherlane"]
                - self.prev["intersection_otherlane"]
        )
        new_ego_offlane = (
                self.ego_curr["intersection_otherlane"]
                - self.ego_prev["intersection_otherlane"]
        )
        new_ego_collision_to_others = (
                self.ego_curr["collision_vehicles"] + self.ego_curr["collision_other"] + self.ego_curr[
            "collision_pedestrians"]
                - self.ego_prev["collision_vehicles"] - self.ego_prev["collision_other"] - self.ego_prev[
                    "collision_pedestrians"]
        )
        if new_collision_to_ego:
            self.reward += new_collision_to_ego * 100
        if new_ego_offlane + new_ego_collision_to_others:
            self.reward += new_ego_offlane * 0.1 + new_ego_collision_to_others * 1
        if new_collision_to_others:
            self.reward -= new_collision_to_others * 10
        if new_own_offlane:
            self.reward -= new_own_offlane * 0.2

        return self.reward

    def egoReward(self):
        self.reward = 0.0
        # self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) * 0.01
        new_ego_offlane = (
                self.ego_curr["intersection_offroad"] + self.ego_curr["intersection_otherlane"]
                - self.ego_prev["intersection_offroad"] - self.ego_prev["intersection_otherlane"]
        )
        new_ego_collision_to_others = (
                self.ego_curr["collision_other"] + self.ego_curr["collision_vehicles"] + self.ego_curr[
            "collision_pedestrians"]
                - self.ego_prev["collision_other"] - self.ego_prev["collision_vehicles"] - self.ego_prev[
                    "collision_pedestrians"]
        )
        if (new_ego_collision_to_others):
            self.reward -= new_ego_collision_to_others * 10

        self.reward -= new_ego_offlane * 0.5

        self.reward += 0.1

        # add a neg-rwd to get the ego faster to collision, but only suitable for centralized training
        # self.reward -= 0.01

        return self.reward

    def MADARL(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

        self.reward += 0.1 * self.curr["forward_speed"]
        # New collision damage
        collision_times = (
                self.curr["collision_vehicles"] +
                self.curr["collision_pedestrians"] + self.curr["collision_other"] -
                self.prev["collision_vehicles"] -
                self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if collision_times > 0:
            self.reward += 5
        # New sidewalk intersection
        offroad = (self.curr["intersection_offroad"] -
                   self.prev["intersection_offroad"])

        # New opposite lane intersection
        offlane = (self.curr["intersection_otherlane"] -
                   self.prev["intersection_otherlane"])
        if offroad > 0:
            self.reward += 0.05
        if offlane > 0:
            self.reward += 0.05
        return self.reward

    def compute_reward_corl2017(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        # Change in speed (km/h)
        self.reward += 0.5 * (
                self.curr["forward_speed"] - self.prev["forward_speed"])
        # New collision damage
        self.reward -= 5 * (
                self.curr["collision_vehicles"] +
                self.curr["collision_pedestrians"] + self.curr["collision_other"] -
                self.prev["collision_vehicles"] -
                self.prev["collision_pedestrians"] - self.prev["collision_other"])

        # New sidewalk intersection
        self.reward -= 0.5 * (self.curr["intersection_offroad"] -
                              self.prev["intersection_offroad"])

        # New opposite lane intersection
        self.reward -= 0.5 * (self.curr["intersection_otherlane"] -
                              self.prev["intersection_otherlane"])

        return self.reward

    def compute_reward_lane_keep(self):
        self.reward = 0.0
        # Speed reward, up 30.0 (km/h)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        # New collision damage
        new_damage = (
                self.curr["collision_vehicles"] +
                self.curr["collision_pedestrians"] + self.curr["collision_other"] -
                self.prev["collision_vehicles"] -
                self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0
        # Sidewalk intersection
        self.reward -= self.curr["intersection_offroad"]
        # Opposite lane intersection
        self.reward -= self.curr["intersection_otherlane"]

        return self.reward

    '''
    advCollisionProbability()
    * distance_to_ego ---> probability of collision to ego(PoCTE),  as reward of NPC
    '''

    def advCollisionProbability(self):
        self.reward = 0.0

        return self.reward

    def advrs(self):
        self.reward = 0.0
        # distance and speed
        x = self.curr["x"]
        y = self.curr["y"]
        z = self.curr["z"]
        ego_x = self.ego_curr["x"]
        ego_y = self.ego_curr["y"]
        ego_z = self.ego_curr["z"]
        disance = calculate_distance(x, y, z, ego_x, ego_y, ego_z)
        if disance < 10:
            self.reward += np.clip(1 / disance, 0, 1)
        else:
            self.reward -= np.clip(disance / 20, 0, 1) * 0.5
        if self.curr["forward_speed"] < 2:
            self.reward -= self.curr["forward_speed"] * 0.1
        else:
            self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) * 0.01

        new_collision_to_ego = (
                self.curr["collision_to_ego"] - self.prev["collision_to_ego"]
        )
        new_ego_offlane = (
                self.ego_curr["intersection_offroad"] + self.ego_curr["intersection_otherlane"]
        )
        new_ego_collision_to_others = (
                self.ego_curr["collision_other"] + self.ego_curr["collision_vehicles"] + self.ego_curr[
            "collision_pedestrians"]
                - self.ego_prev["collision_other"] - self.ego_prev["collision_vehicles"] - self.ego_prev[
                    "collision_pedestrians"]
        )

        if new_collision_to_ego:
            self.reward += new_collision_to_ego * 10
        if new_ego_offlane + new_ego_collision_to_others:
            self.reward += new_ego_offlane * 0.5 + new_ego_collision_to_others

        self.reward += self.curr["intersection_offroad"] * 0.5
        self.reward += self.curr["intersection_otherlane"] * 0.5

        return self.reward

    def compute_reward_custom(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        new_damage = (
                self.curr["collision_vehicles"]  # + self.curr["collision_pedestrians"]
                + self.curr["collision_other"] -
                self.prev["collision_vehicles"] -  # - self.prev["collision_pedestrians"]
                self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0

        self.reward -= self.curr["intersection_offroad"] * 0.5
        self.reward -= self.curr["intersection_otherlane"] * 0.5

        if self.curr["next_command"] == "REACH_GOAL":
            self.reward += 100
        if self.curr["next_command"] == "LANE_FOLLOW":
            self.reward += 0.5
        return self.reward

    # def compute_reward_corl2017(self):
    #     self.reward = 0.0
    #     cur_dist = self.curr["distance_to_goal"]
    #     prev_dist = self.prev["distance_to_goal"]
    #     # Distance travelled toward the goal in m
    #     self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
    #     # Change in speed (km/h)
    #     self.reward += 0.05 * (
    #         self.curr["forward_speed"] - self.prev["forward_speed"])
    #     # New collision damage
    #     self.reward -= .00002 * (
    #         self.curr["collision_vehicles"] +
    #         self.curr["collision_pedestrians"] + self.curr["collision_other"] -
    #         self.prev["collision_vehicles"] -
    #         self.prev["collision_pedestrians"] - self.prev["collision_other"])

    #     # New sidewalk intersection
    #     self.reward -= 2 * (self.curr["intersection_offroad"] -
    #                         self.prev["intersection_offroad"])

    #     # New opposite lane intersection
    #     self.reward -= 2 * (self.curr["intersection_otherlane"] -
    #                         self.prev["intersection_otherlane"])

    #     return self.reward

    # def compute_reward_lane_keep(self):
    #     self.reward = 0.0
    #     # Speed reward, up 30.0 (km/h)
    #     self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
    #     # New collision damage
    #     new_damage = (
    #         self.curr["collision_vehicles"] +
    #         self.curr["collision_pedestrians"] + self.curr["collision_other"] -
    #         self.prev["collision_vehicles"] -
    #         self.prev["collision_pedestrians"] - self.prev["collision_other"])
    #     if new_damage:
    #         self.reward -= 100.0
    #     # Sidewalk intersection
    #     self.reward -= self.curr["intersection_offroad"]
    #     # Opposite lane intersection
    #     self.reward -= self.curr["intersection_otherlane"]

    #     return self.reward

    # def destory(self):
    #     pass
