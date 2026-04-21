#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery feature preprocessor.
智运无人机特征预处理器。
"""


import numpy as np
from agent_ppo.conf.conf import Config


def norm(v, max_v, min_v=0):
    """Normalize v to [0, 1].

    将 v 归一化到 [0, 1]。
    """
    v = np.clip(v, min_v, max_v)
    return (v - min_v) / (max_v - min_v)


def _get_pos_feature(found, cur_pos, target_pos, is_target=False):
    """Compute 7D position feature for a target relative to current position.

    计算目标位置相对于当前位置的 7 维特征。
    """
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
    """feature preprocessor for Drone Delivery.

    智运无人机预处理器，仅保留最少信息。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state.

        重置所有状态。
        """
        self.cur_pos = (0, 0)
        self.last_pos = None

        # Game state / 游戏状态
        self.battery = 100
        self.battery_max = 100
        self.packages = []
        self.last_package_count = 0
        self.local_map = []
        self.delivered = 0
        self.last_delivered = 0
        self.step_no = 0
        self.prev_goal_key = None
        self.prev_goal_dist = None

        # Entities / 实体
        self.warehouses = []
        self.stations = []

    def _parse_obs(self, env_obs):
        """Parse essential fields from observation dict.

        从 observation 字典中解析必要字段。
        """
        obs = env_obs["observation"]
        # if self.step_no < 3:
        #     print("obs_keys =", list(obs.keys()))
        #     print("obs_map_info =", obs.get("map_info"))
        #     if self.step_no < 3 and isinstance(obs.get("map_info"), dict):
        #         print("map_info_keys =", list(obs["map_info"].keys()))
        frame_state = obs["frame_state"]

        hero = frame_state["heroes"]
        new_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.last_pos = self.cur_pos
        self.cur_pos = new_pos

        map_info = obs.get("map_info", [])
        if isinstance(map_info, list):
            local_map = map_info
        else:
            local_map = []
        self.local_map = local_map

        self.battery = hero.get("battery", self.battery_max)
        self.battery_max = hero.get("battery_max", 100)
        self.last_package_count = len(self.packages)
        self.packages = hero.get("packages", [])
        self.last_delivered = self.delivered
        self.delivered = hero.get("delivered", 0)
        self.step_no = obs.get("step_no", 0)

        self.warehouses = []
        self.stations = []
        for organ in frame_state.get("organs", []):
            st = organ.get("sub_type", 0)
            if st == 1:
                self.warehouses.append(organ)
            elif st == 3:
                self.stations.append(organ)

        self.legal_act = obs.get("legal_act", obs.get("legal_action", [1] * 8))
        self.legal_action = self.legal_act

    def feature_process(self, env_obs, last_action):
        """Core feature extraction. Returns (feature_38d, legal_action, reward).

        核心特征提取方法，返回 22 维特征向量、合法动作掩码和奖励。
        """
        self._parse_obs(env_obs)

        # 1. Hero state features (4D) / 英雄状态特征（4D）
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

        # 2. Nearest 1 station feature (7D) / 最近 1 个驿站特征（7D）
        # Target stations first, then by distance
        # 目标驿站优先，然后按距离排序
        s, is_target = self._select_active_goal()

        if s is not None:
            station_feat = _get_pos_feature(
                True,
                self.cur_pos,
                (s["pos"]["x"], s["pos"]["z"]),
                is_target=is_target,
            )
            target_visible = float(is_target)
        else:
            station_feat = _get_pos_feature(False, self.cur_pos, self.cur_pos, is_target=False)
            target_visible = 0.0

        # 3. Legal action mask (8D) / 合法动作掩码（8D）
        legal_action = self._get_legal_action()

        # 4. Binary indicators (3D) / 二值指示器（3D）
        has_package = 1.0 if len(self.packages) > 0 else 0.0
        battery_low = 1.0 if (self.battery / max(self.battery_max, 1)) < 0.3 else 0.0
        indicators = np.array([has_package, battery_low, target_visible])
        last_act_feat = np.zeros(Config.LAST_ACT_DIM, dtype=float)
        if last_action is not None and 0 <= int(last_action) < Config.LAST_ACT_DIM:
            last_act_feat[int(last_action)] = 1.0
        moved_status = 1.0
        if self.last_pos is not None and self.cur_pos == self.last_pos:
            moved_status = 0.0
        moved_feat = np.array([moved_status], dtype=float)
        # Concatenate features (Total 22D / 合计 22D)
        local_patch_feat = self._extract_local_patch_feat(self.local_map)

        feature = np.concatenate(
            [
                hero_feat,
                station_feat,
                np.array(legal_action, dtype=float),
                indicators,
                last_act_feat,
                moved_feat,
                local_patch_feat,
            ]
        )

        reward = self._reward_process()

        # if self.step_no < 3:
        #     print("feature_len =", len(feature))
        #     print("local_map_type =", type(self.local_map))
        #     print("local_map_len =", len(self.local_map) if isinstance(self.local_map, list) else "not_list")

        #     if isinstance(self.local_map, list) and len(self.local_map) > 0:
        #         print("local_map_first_type =", type(self.local_map[0]))
        #         print("local_map_first =", self.local_map[0])

        #         if isinstance(self.local_map[0], list):
        #             print("local_map_shape =", len(self.local_map), len(self.local_map[0]))
        #             print("local_map_center_row =", self.local_map[10] if len(self.local_map) > 10 else "no_row_10")
        #             if len(self.local_map) > 10 and isinstance(self.local_map[10], list) and len(self.local_map[10]) > 10:
        #                 print("local_map_center_cell =", self.local_map[10][10])

        #     print("last16 =", feature[-16:])

        return feature, legal_action, reward

    def _act_to_delta(self, act):
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

    def _get_legal_action(self):
        """Get legal action mask.

        获取合法动作掩码。
        """
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

    def _get_goal_key(self, goal):
        if goal is None:
            return None

        if "sub_type" in goal and "config_id" in goal:
            return (goal["sub_type"], goal["config_id"])

        pos = goal.get("pos", {})
        return (
            goal.get("sub_type", 0),
            pos.get("x", 0),
            pos.get("z", 0),
        )

    def _get_goal_dist(self, goal):
        if goal is None:
            return None

        pos = goal.get("pos", {})
        dx = pos.get("x", 0) - self.cur_pos[0]
        dz = pos.get("z", 0) - self.cur_pos[1]
        return np.sqrt(dx ** 2 + dz ** 2)

    def _reward_process(self):
        """Reward function.

        奖励函数。
        """
        reward = 0.0

        # 1. Delivery reward / 投递奖励
        newly_delivered = max(0, self.delivered - self.last_delivered)
        if newly_delivered > 0:
            reward += 1.0 * newly_delivered
        if self.last_package_count == 0 and len(self.packages) > 0:
            reward += 0.1
        cur_goal = self._select_active_goal()[0]
        cur_goal_key = self._get_goal_key(cur_goal)
        cur_goal_dist = self._get_goal_dist(cur_goal)
        if (
            cur_goal is not None
            and self.prev_goal_key == cur_goal_key
            and self.prev_goal_dist is not None
        ):
            progress = self.prev_goal_dist - cur_goal_dist
            reward += 0.005 * np.clip(progress, -1.0, 1.0)

        if self.last_pos is not None and self.cur_pos == self.last_pos:
            reward -= 0.002


        # 2. Step penalty / 步数惩罚
        reward -= 0.001

        self.prev_goal_key = cur_goal_key
        self.prev_goal_dist = cur_goal_dist

        return [reward]
