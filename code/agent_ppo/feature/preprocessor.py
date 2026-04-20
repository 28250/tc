#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright æ¼ 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery feature preprocessor.
负责将环境原始 observation 转成 PPO 使用的特征与即时奖励。
"""


import numpy as np
from agent_ppo.conf.conf import Config


def norm(v, max_v, min_v=0):
    """将数值裁剪后归一化到 [0, 1]。"""
    v = np.clip(v, min_v, max_v)
    return (v - min_v) / (max_v - min_v)


def _get_pos_feature(found, cur_pos, target_pos, is_target=False):
    """构造目标点相对当前位置的 7 维位置特征。"""
    relative_pos = (target_pos[0] - cur_pos[0], target_pos[1] - cur_pos[1])
    dist = np.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2)
    abs_norm = norm(np.array(target_pos), 128, -128)
    return np.array(
        [
            float(found),
            norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
            norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
            abs_norm[0],
            abs_norm[1],
            norm(dist, 1.41 * 128),
            1.0 if is_target else 0.0,
        ]
    )


class Preprocessor:
    """无人机配送任务的特征预处理器。"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置局内状态。"""
        self.cur_pos = (0, 0)

        # 局内状态
        self.battery = 100
        self.battery_max = 100
        self.packages = []
        self.local_map = []
        self.delivered = 0
        self.last_delivered = 0
        self.step_no = 0

        # 地图实体
        self.warehouses = []
        self.chargers = []
        self.stations = []
        self.force_charging = False  # 是否处于强制补能模式
        self.locked_energy_target_key = None  # 充电模式下锁定的补能目标

    def _parse_obs(self, env_obs):
        """从环境 observation 中提取当前位置、资源状态和地图实体。"""
        obs = env_obs["observation"]
        frame_state = obs["frame_state"]

        hero = frame_state["heroes"]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        map_info = obs.get("map_info", [])
        if isinstance(map_info, list):
            local_map = map_info
        else:
            local_map = []
        self.local_map = local_map

        self.battery = hero.get("battery", self.battery_max)
        self.battery_max = hero.get("battery_max", 100)
        self.packages = hero.get("packages", [])

        self.last_delivered = self.delivered
        self.delivered = hero.get("delivered", 0)
        self.step_no = obs.get("step_no", 0)

        # sub_type: 1=仓库, 2=充电桩, 3=驿站
        self.warehouses = []
        self.chargers = []
        self.stations = []
        for organ in frame_state.get("organs", []):
            st = organ.get("sub_type", 0)
            if st == 1:
                self.warehouses.append(organ)
            elif st == 2:
                self.chargers.append(organ)
            elif st == 3:
                self.stations.append(organ)

        self.legal_act = obs.get("legal_act", obs.get("legal_action", [1] * 8))
        self.legal_action = self.legal_act

    def feature_process(self, env_obs, last_action):
        """提取特征，返回 (feature, legal_action, reward)。"""
        self._parse_obs(env_obs)

        # 1. 自身状态特征（4维）
        battery_ratio = norm(self.battery, self.battery_max)
        package_count_norm = norm(len(self.packages), 3)
        cur_pos_norm = norm(np.array(self.cur_pos, dtype=float), 128, -128)
        hero_feat = np.array(
            [
                battery_ratio,
                package_count_norm,
                cur_pos_norm[0],
                cur_pos_norm[1],
            ]
        )

        # 2. 导航相关目标
        # nav_goal: 当前真正执行的目标
        # active_goal: 正常送货/取货目标
        # energy_goal: 最近或锁定的补能目标
        nav_goal, active_goal, energy_goal, target_visible = self._select_navigation_goal()

        if nav_goal is not None:
            station_feat = _get_pos_feature(
                True,
                self.cur_pos,
                (nav_goal["pos"]["x"], nav_goal["pos"]["z"]),
                is_target=bool(target_visible),
            )
        else:
            station_feat = _get_pos_feature(False, self.cur_pos, self.cur_pos, is_target=False)
            target_visible = 0.0

        # 3. 合法动作掩码（8维）
        legal_action = self._get_legal_action()

        # 4. 二值状态特征（3维）
        has_package = 1.0 if len(self.packages) > 0 else 0.0
        battery_low = 1.0 if (self.battery / max(self.battery_max, 1)) < 0.3 else 0.0
        indicators = np.array([has_package, battery_low, target_visible])
        target_hint_feat = self._extract_target_hint_feat(nav_goal)
        energy_hint_feat = self._extract_energy_hint_feat(energy_goal)

        # 5. 局部地图 patch（9x9 -> 81维）
        local_patch_feat = self._extract_local_patch_feat(self.local_map)

        feature = np.concatenate(
            [
                hero_feat,
                station_feat,
                np.array(legal_action, dtype=float),
                indicators,
                target_hint_feat,
                energy_hint_feat,
                local_patch_feat,
            ]
        )

        reward = self._reward_process()
        return feature, legal_action, reward

    def _act_to_delta(self, act):
        """将 8 个离散方向动作映射为 (dx, dz)。"""
        act_to_delta = {
            0: (1, 0),
            1: (1, -1),
            2: (0, -1),
            3: (-1, -1),
            4: (-1, 0),
            5: (-1, 1),
            6: (0, 1),
            7: (1, 1),
        }
        return act_to_delta.get(act)

    def _extract_local_passable_feat(self, local_map):
        """提取当前位置周围 1/2 步内各方向是否可通行。"""
        if not isinstance(local_map, list):
            return np.zeros(2 * Config.ACTION_NUM, dtype=float)

        def is_passable(act, step):
            delta = self._act_to_delta(act)
            if delta is None:
                return 0.0

            dx, dz = delta
            row_idx = 10 + dz * step
            col_idx = 10 + dx * step
            if row_idx < 0 or row_idx >= len(local_map):
                return 0.0

            row = local_map[row_idx]
            if not isinstance(row, list):
                return 0.0
            if col_idx < 0 or col_idx >= len(row):
                return 0.0

            cell = row[col_idx]
            return 1.0 if isinstance(cell, (int, float)) and int(cell) == 1 else 0.0

        passable_1 = [is_passable(act, 1) for act in range(Config.ACTION_NUM)]
        passable_2 = [is_passable(act, 2) for act in range(Config.ACTION_NUM)]
        return np.array(passable_1 + passable_2, dtype=float)

    def _extract_local_patch_feat(self, local_map):
        """提取局部 9x9 可通行 patch，作为地图观测。"""
        if not isinstance(local_map, list):
            return np.zeros(Config.LOCAL_PATCH_DIM, dtype=float)

        patch_feat = []
        for row_idx in range(6, 15):
            for col_idx in range(6, 15):
                value = 0.0

                if 0 <= row_idx < len(local_map):
                    row = local_map[row_idx]
                    if isinstance(row, list) and 0 <= col_idx < len(row):
                        cell = row[col_idx]
                        if isinstance(cell, (int, float)) and int(cell) == 1:
                            value = 1.0

                patch_feat.append(value)

        return np.array(patch_feat, dtype=float)

    def _get_entity_key(self, entity):
        """为实体生成稳定 key，用于补能目标锁定。"""
        if entity is None:
            return None

        sub_type = entity.get("sub_type", 0)
        if "config_id" in entity:
            return (sub_type, entity.get("config_id"))

        pos = entity.get("pos", {})
        return (sub_type, pos.get("x", 0), pos.get("z", 0))

    def _get_entity_dist(self, entity):
        """计算当前位置到实体的欧氏距离。"""
        dx = entity["pos"]["x"] - self.cur_pos[0]
        dz = entity["pos"]["z"] - self.cur_pos[1]
        return np.sqrt(dx * dx + dz * dz)

    def _select_nearest_warehouse(self):
        """选择最近仓库。"""
        if len(self.warehouses) <= 0:
            return None

        return min(
            self.warehouses,
            key=lambda s: (s["pos"]["x"] - self.cur_pos[0]) ** 2
            + (s["pos"]["z"] - self.cur_pos[1]) ** 2,
        )

    def _select_nearest_station(self):
        """选择最近驿站。"""
        if len(self.stations) <= 0:
            return None

        return min(
            self.stations,
            key=lambda s: (s["pos"]["x"] - self.cur_pos[0]) ** 2
            + (s["pos"]["z"] - self.cur_pos[1]) ** 2,
        )

    def _select_nearest_energy_goal(self):
        """选择最近补能点。

        补能点包括充电桩和仓库。
        仓库选择时带有轻微距离偏置。
        """
        energy_points = self.chargers + self.warehouses
        if len(energy_points) <= 0:
            return None

        return min(
            energy_points,
            key=lambda s: self._get_entity_dist(s) - (5.0 if s.get("sub_type", 0) == 1 else 0.0),
        )

    def _find_locked_energy_goal(self):
        """根据锁定 key 查找当前补能目标。"""
        if self.locked_energy_target_key is None:
            return None

        for energy_point in self.chargers + self.warehouses:
            if self._get_entity_key(energy_point) == self.locked_energy_target_key:
                return energy_point

        self.locked_energy_target_key = None
        return None

    def _select_navigation_goal(self):
        """统一选择当前导航目标。

        规则：
        1. 无包裹时去最近仓库，并退出充电模式
        2. 有包裹时默认去 active_goal
        3. 低电时进入充电模式，优先去锁定或最近补能点
        4. 电量恢复后退出充电模式
        """
        battery_ratio = self.battery / max(self.battery_max, 1)

        if len(self.packages) <= 0:
            self.force_charging = False
            self.locked_energy_target_key = None
            nav_goal = self._select_nearest_warehouse()
            return nav_goal, nav_goal, nav_goal, 0.0

        active_goal, active_is_target = self._select_active_goal()
        if active_goal is None:
            active_goal = self._select_nearest_station()
            active_is_target = False

        locked_energy_goal = self._find_locked_energy_goal()
        nearest_energy_goal = self._select_nearest_energy_goal()
        energy_goal = locked_energy_goal if locked_energy_goal is not None else nearest_energy_goal

        if self.force_charging:
            if battery_ratio > 0.75 or energy_goal is None:
                self.force_charging = False
                self.locked_energy_target_key = None
        else:
            active_goal_dist = self._get_entity_dist(active_goal) if active_goal is not None else None
            energy_goal_dist = self._get_entity_dist(energy_goal) if energy_goal is not None else None

            should_enter = False
            if energy_goal is not None:
                if battery_ratio < 0.20:
                    should_enter = True
                elif battery_ratio < 0.35 and active_goal_dist is not None:
                    should_enter = (energy_goal_dist + 8.0) < active_goal_dist

            if should_enter:
                self.force_charging = True
                self.locked_energy_target_key = self._get_entity_key(energy_goal)

        if self.force_charging:
            locked_energy_goal = self._find_locked_energy_goal()
            energy_goal = locked_energy_goal if locked_energy_goal is not None else nearest_energy_goal
            if energy_goal is None:
                self.force_charging = False
                self.locked_energy_target_key = None

        nav_goal = energy_goal if self.force_charging and energy_goal is not None else active_goal
        target_visible = float(
            nav_goal is not None
            and active_goal is not None
            and self._get_entity_key(nav_goal) == self._get_entity_key(active_goal)
            and active_is_target
        )
        return nav_goal, active_goal, energy_goal, target_visible

    def _extract_target_hint_feat(self, target):
        """基于当前导航目标生成 3 维 target hint。"""
        if target is None:
            return np.zeros(Config.TARGET_HINT_DIM, dtype=float)

        dx = target["pos"]["x"] - self.cur_pos[0]
        dz = target["pos"]["z"] - self.cur_pos[1]
        dist = np.sqrt(dx * dx + dz * dz)

        target_dir_x = dx / dist if dist > 1e-6 else 0.0
        target_dir_z = dz / dist if dist > 1e-6 else 0.0
        target_dist_norm = min(dist / 181.0, 1.0)
        return np.array([target_dir_x, target_dir_z, target_dist_norm], dtype=float)

    def _extract_energy_hint_feat(self, energy_goal):
        """基于补能目标生成 4 维 energy hint。"""
        if energy_goal is None:
            return np.zeros(Config.ENERGY_HINT_DIM, dtype=float)

        dx = energy_goal["pos"]["x"] - self.cur_pos[0]
        dz = energy_goal["pos"]["z"] - self.cur_pos[1]
        dist = self._get_entity_dist(energy_goal)
        denom = max(dist, 1e-6)

        return np.array(
            [
                1.0 if self.force_charging else 0.0,
                dx / denom,
                dz / denom,
                min(dist / 181.0, 1.0),
            ],
            dtype=float,
        )

    def _get_legal_action(self):
        """返回长度为 8 的合法动作掩码。"""
        legal_act = None
        if hasattr(self, "legal_act") and self.legal_act:
            legal_act = self.legal_act
        elif hasattr(self, "legal_action") and self.legal_action:
            legal_act = self.legal_action

        if legal_act:
            legal_action = [int(x) for x in legal_act[:8]]
        else:
            legal_action = [1] * 8

        if len(legal_action) < 8:
            legal_action = legal_action + [1] * (8 - len(legal_action))

        if sum(legal_action) == 0:
            return [1] * 8

        return legal_action

    def _select_active_goal(self):
        """选择正常任务目标。

        有包裹时选择对应投递驿站；
        无包裹时选择最近仓库。
        """
        if len(self.packages) > 0:
            target_ids = set(self.packages)
            target_stations = [s for s in self.stations if s.get("config_id", 0) in target_ids]
            if len(target_stations) > 0:
                return min(
                    target_stations,
                    key=lambda s: (s["pos"]["x"] - self.cur_pos[0]) ** 2 + (s["pos"]["z"] - self.cur_pos[1]) ** 2,
                ), True
            return None, False

        if len(self.warehouses) > 0:
            return min(
                self.warehouses,
                key=lambda s: (s["pos"]["x"] - self.cur_pos[0]) ** 2 + (s["pos"]["z"] - self.cur_pos[1]) ** 2,
            ), False

        return None, False

    def _reward_process(self):
        """计算单步奖励。"""
        reward = 0.0

        # 1. 新增投递奖励
        newly_delivered = max(0, self.delivered - self.last_delivered)
        if newly_delivered > 0:
            reward += 1.0 * newly_delivered

        # 2. 每步固定惩罚
        reward -= 0.001

        return [reward]
