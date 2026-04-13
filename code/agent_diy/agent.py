#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery DIY Agent class based on kaiwudrl BaseAgent interface.
智运无人机 DIY Agent 主类，基于 kaiwudrl BaseAgent 接口。
"""


import torch
from kaiwudrl.interface.agent import BaseAgent
from agent_diy.model.model import Model
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ObsData, ActData

import os
import pickle

class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        """Initialize the agent.

        初始化 Agent。
        """
        super().__init__(agent_type, device, logger, monitor)
        self._last_debug_target = None
        self._last_debug_reason = None
        self._last_debug_final_act = None
        self._debug_interval = 50
        self._last_pos = None
        self._stuck_count = 0
        self._escape_index = 0
        self._stuck_threshold = 3
        if logger is not None:
            logger.info("DIY Agent __init__ called")

    def _format_act(self, act):
        act_name = {
            0: "right",
            1: "right_up",
            2: "up",
            3: "left_up",
            4: "left",
            5: "left_down",
            6: "down",
            7: "right_down",
        }
        if isinstance(act, int):
            return f"{act}({act_name.get(act, 'unknown')})"
        return str(act)

    def _log_decision(
        self,
        step_no,
        feature,
        battery,
        battery_max,
        packages,
        target_desc,
        target_reason,
        preferred_act,
        final_act,
        dx,
        dz,
        legal_act,
    ):
        log_target_changed = target_desc != self._last_debug_target
        log_reason_changed = target_reason != self._last_debug_reason
        log_act_changed = final_act != self._last_debug_final_act
        log_act_mismatch = preferred_act != final_act
        log_periodic = (step_no <= 10) or (step_no % self._debug_interval == 0)
        should_log = (
            log_target_changed
            or log_reason_changed
            or log_act_changed
            or log_act_mismatch
            or log_periodic
        )

        if should_log and self.logger is not None:
            self.logger.info(
                f"[DECISION] step={step_no}, pos={feature['cur_pos']}, "
                f"battery={battery}/{battery_max}, packages={packages}, "
                f"target={target_desc}, reason={target_reason}, "
                f"preferred_act={self._format_act(preferred_act)}, "
                f"final_act={self._format_act(final_act)}, "
                f"same_act={preferred_act == final_act}, "
                f"dx={dx if dx is not None else 'NA'}, "
                f"dz={dz if dz is not None else 'NA'}, "
                f"legal_act={legal_act}"
            )

        self._last_debug_target = target_desc
        self._last_debug_reason = target_reason
        self._last_debug_final_act = final_act

    def _update_stuck_state(self, cur_pos):
        """Track whether the agent stays on the same tile."""
        if self._last_pos == cur_pos:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
            self._escape_index = 0

        self._last_pos = cur_pos

    def _get_escape_candidates(self, preferred_act):
        """Try nearby directions around the preferred action first."""
        if preferred_act is None or not isinstance(preferred_act, int):
            return list(range(8))

        offsets = [1, -1, 2, -2, 3, -3, 4]
        candidates = [preferred_act]
        for offset in offsets:
            candidates.append((preferred_act + offset) % 8)
        return candidates

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

    def _is_local_act_passable(self, local_map, act, step=1):
        if not isinstance(local_map, list):
            return False
        if not isinstance(act, int):
            return False
        if not isinstance(step, int) or step <= 0:
            return False

        delta = self._act_to_delta(act)
        if delta is None:
            return False

        dx, dz = delta
        row_idx = 10 + dz * step
        col_idx = 10 + dx * step
        if row_idx < 0 or row_idx >= len(local_map):
            return False
        row = local_map[row_idx]
        if not isinstance(row, list):
            return False
        if col_idx < 0 or col_idx >= len(row):
            return False
        cell = row[col_idx]
        return isinstance(cell, (int, float)) and int(cell) == 1

    def _select_best_avoid_act(self, local_map, legal_act, candidates, target_dx, target_dz, reference_act, log_mode):
        best_act = None
        best_score = None
        log_items = []
        current_dist2 = None
        if isinstance(target_dx, (int, float)) and isinstance(target_dz, (int, float)):
            current_dist2 = target_dx * target_dx + target_dz * target_dz

        for candidate_act in candidates:
            legal_ok = 0 <= candidate_act < len(legal_act) and legal_act[candidate_act]
            pass1 = legal_ok and self._is_local_act_passable(local_map, candidate_act, step=1)
            pass2 = pass1 and self._is_local_act_passable(local_map, candidate_act, step=2)
            closer = False
            turn_penalty = 0.0
            score = None

            if pass1:
                score = 0.0
                if pass2:
                    score += 1.0

                delta = self._act_to_delta(candidate_act)
                if delta is not None and current_dist2 is not None:
                    next_dx = target_dx - delta[0]
                    next_dz = target_dz - delta[1]
                    if next_dx * next_dx + next_dz * next_dz < current_dist2:
                        closer = True
                        score += 1.0

                if isinstance(reference_act, int):
                    turn_distance = min(
                        (candidate_act - reference_act) % 8,
                        (reference_act - candidate_act) % 8,
                    )
                    turn_penalty = 0.05 * turn_distance
                    score -= turn_penalty

                if best_score is None or score > best_score:
                    best_score = score
                    best_act = candidate_act

            score_text = "NA" if score is None else f"{score:.2f}"
            log_items.append(
                f"{self._format_act(candidate_act)}:legal={int(legal_ok)},pass1={int(pass1)},"
                f"pass2={int(pass2)},closer={int(closer)},turn_penalty={turn_penalty:.2f},score={score_text}"
            )

        if self.logger is not None:
            self.logger.info(
                f"[AVOID] mode={log_mode}, "
                f"reference_act={self._format_act(reference_act)}, "
                f"selected={self._format_act(best_act)}, "
                f"candidates={log_items}"
            )

        return best_act

    def _finalize_action(self, feature, legal_act, preferred_act, local_map=None, target_dx=None, target_dz=None):
        """Rotate through nearby legal actions when the agent is stuck."""
        cur_pos = feature["cur_pos"]
        self._update_stuck_state(cur_pos)

        # Base action is the preferred direction if legal, otherwise the first legal move.
        if isinstance(preferred_act, int) and 0 <= preferred_act < len(legal_act) and legal_act[preferred_act]:
            base_act = preferred_act
        else:
            base_act = 0
            for act, ok in enumerate(legal_act):
                if ok:
                    base_act = act
                    break

        if self._stuck_count < self._stuck_threshold:
            return base_act

        candidates = self._get_escape_candidates(base_act)
        if candidates:
            rotate = self._escape_index % len(candidates)
            candidates = candidates[rotate:] + candidates[:rotate]
        final_act = self._select_best_avoid_act(
            local_map if local_map is not None else [],
            legal_act,
            candidates,
            target_dx,
            target_dz,
            base_act,
            "stuck",
        )
        legal_candidates = [act for act in candidates if 0 <= act < len(legal_act) and legal_act[act]]
        if final_act is None:
            final_act = base_act
        elif legal_candidates:
            self._escape_index += 1

        if self.logger is not None:
            self.logger.info(
                f"[STUCK] pos={cur_pos}, "
                f"stuck_count={self._stuck_count}, "
                f"base_act={self._format_act(base_act)}, "
                f"final_act={self._format_act(final_act)}, "
                f"escape_index={self._escape_index}, "
                f"legal_candidates={[self._format_act(act) for act in legal_candidates]}, "
                f"legal_act={legal_act}"
            )

        return final_act

    def predict(self, list_obs_data):
        """Predict action from observation data.

        根据观测数据推理动作。
        """
        act = self.exploit(list_obs_data)
        return [ActData(act=act)]


    def exploit(self, list_obs_data):
        """Evaluation mode inference (greedy)."""
        # 1. Extract feature and legal action mask from the latest observation.
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_act = obs_data.legal_act

        cur_x, cur_z = feature["cur_pos"]
        battery = feature["battery"]
        battery_max = max(feature["battery_max"], 1)
        packages = feature["packages"]
        warehouses = feature["warehouses"]
        chargers = feature["chargers"]
        stations = feature["stations"]
        step_no = feature.get("step_no", 0)
        local_map = feature.get("local_map", [])

        # 2. Decide the current target according to battery and package status.
        target = None
        target_reason = "none"
        if battery / battery_max < 0.25 and len(chargers) > 0:
            target = min(
                chargers,
                key=lambda o: (o["pos"]["x"] - cur_x) ** 2 + (o["pos"]["z"] - cur_z) ** 2,
            )
            target_reason = "low_battery_go_charger"
        elif len(packages) > 0:
            package_set = set(packages)
            target_stations = [s for s in stations if s.get("config_id", 0) in package_set]

            if len(target_stations) > 0:
                target = min(
                    target_stations,
                    key=lambda o: (o["pos"]["x"] - cur_x) ** 2 + (o["pos"]["z"] - cur_z) ** 2,
                )
                target_reason = "has_package_go_target_station"
            elif len(stations) > 0:
                target = min(
                    stations,
                    key=lambda o: (o["pos"]["x"] - cur_x) ** 2 + (o["pos"]["z"] - cur_z) ** 2,
                )
                target_reason = "has_package_go_nearest_station"
        elif len(warehouses) > 0:
            target = min(
                warehouses,
                key=lambda o: (o["pos"]["x"] - cur_x) ** 2 + (o["pos"]["z"] - cur_z) ** 2,
            )
            target_reason = "no_package_go_warehouse"

        if target is None:
            target_desc = "None"
        else:
            target_desc = (
                f"sub_type={target.get('sub_type')}, "
                f"config_id={target.get('config_id')}, "
                f"pos=({target['pos']['x']},{target['pos']['z']})"
            )

        preferred_act = None
        preferred_act_for_log = None
        dx = None
        dz = None

        if target is not None:
            # Convert the target offset into one of the 8 movement actions.
            target_x = target["pos"]["x"]
            target_z = target["pos"]["z"]
            dx = target_x - cur_x
            dz = target_z - cur_z

            if dx == 0 and dz == 0:
                preferred_act_for_log = "at_target"
            else:
                sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
                sz = 1 if dz > 0 else (-1 if dz < 0 else 0)

                dir_to_act = {
                    (1, 0): 0,
                    (1, -1): 1,
                    (0, -1): 2,
                    (-1, -1): 3,
                    (-1, 0): 4,
                    (-1, 1): 5,
                    (0, 1): 6,
                    (1, 1): 7,
                }
                preferred_act = dir_to_act.get((sx, sz), None)
                if preferred_act is not None and not self._is_local_act_passable(local_map, preferred_act):
                    selected_act = self._select_best_avoid_act(
                        local_map,
                        legal_act,
                        self._get_escape_candidates(preferred_act),
                        dx,
                        dz,
                        preferred_act,
                        "preferred_blocked",
                    )
                    if selected_act is not None:
                        preferred_act = selected_act
                preferred_act_for_log = preferred_act

        final_act = self._finalize_action(feature, legal_act, preferred_act, local_map=local_map, target_dx=dx, target_dz=dz)
        self._log_decision(
            step_no=step_no,
            feature=feature,
            battery=battery,
            battery_max=battery_max,
            packages=packages,
            target_desc=target_desc,
            target_reason=target_reason,
            preferred_act=preferred_act_for_log,
            final_act=final_act,
            dx=dx,
            dz=dz,
            legal_act=legal_act,
        )
        return final_act

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        pass

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        if path is None:
            return

        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        state = {
            "agent_type": "diy_rule_agent",
            "version": 1,
        }

        with open(model_file_path, "wb") as f:
            pickle.dump(state, f)

        if self.logger is not None:
            self.logger.info(f"DIY save_model success: {model_file_path}")


    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        if path is None:
            return

        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        if not os.path.exists(model_file_path):
            if self.logger is not None:
                self.logger.info(f"DIY load_model skipped, file not found: {model_file_path}")
            return

        with open(model_file_path, "rb") as f:
            state = pickle.load(f)

        if self.logger is not None:
            self.logger.info(f"DIY load_model success: {model_file_path}, state={state}")

    def observation_process(self, obs, preprocessor=None, extra_info=None):
        """This function is an important feature processing function, mainly responsible for:
            - Parsing information in the raw data
            - Parsing preprocessed feature data
            - Processing the features and returning the processed feature vector
            - Concatenation of features
            - Annotation of legal actions
        """
        if self.logger is not None:
            pass
        # 1. 兼容两种输入：
        #    - obs 本身就是 observation
        #    - obs 外面还包了一层 {"observation": ...}
        raw_obs = obs.get("observation", obs)

        # 2. 取 frame_state
        frame_state = raw_obs.get("frame_state", {})

        # 3. 取 hero 信息
        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})

        cur_x = hero_pos.get("x", 0)
        cur_z = hero_pos.get("z", 0)

        battery = hero.get("battery", 0)
        battery_max = hero.get("battery_max", 100)
        packages = hero.get("packages", [])
        delivered = hero.get("delivered", 0)
        map_info = raw_obs.get("map_info", {})
        local_map = []
        if isinstance(map_info, dict):
            local_map = map_info.get("map_info", [])
        if not isinstance(local_map, list):
            local_map = []

        # 4. 取 organs，并按类型分类
        organs = frame_state.get("organs", [])

        warehouses = []
        chargers = []
        stations = []

        for organ in organs:
            sub_type = organ.get("sub_type", 0)
            if sub_type == 1:
                warehouses.append(organ)
            elif sub_type == 2:
                chargers.append(organ)
            elif sub_type == 3:
                stations.append(organ)

        # 5. 取合法动作
        #    有的地方可能叫 legal_act，有的地方可能叫 legal_action
        legal_act = raw_obs.get("legal_act", raw_obs.get("legal_action", [1] * 8))
        legal_act = [int(x) for x in legal_act[:8]]

        # 防御性处理：长度不够补 1，全 0 就默认全可走
        if len(legal_act) < 8:
            legal_act = legal_act + [1] * (8 - len(legal_act))
        if sum(legal_act) == 0:
            legal_act = [1] * 8

        # 6. 先把 feature 做成字典，方便规则策略直接使用
        feature = {
            "cur_pos": (cur_x, cur_z),
            "battery": battery,
            "battery_max": battery_max,
            "packages": packages,
            "delivered": delivered,
            "warehouses": warehouses,
            "chargers": chargers,
            "stations": stations,
            "local_map": local_map,
            "step_no": raw_obs.get("step_no", 0),
        }

        # 7. remain_info 先简单留一些可能后面会用到的信息
        remain_info = {
            "delivered": delivered,
            "battery": battery,
            "step_no": raw_obs.get("step_no", 0),
        }

        return ObsData(feature=feature, legal_act=legal_act), remain_info
    
    def action_process(self, act_data):
        """Process action data.

        处理动作数据。
        """
        return int(act_data.act)
    
    def reset(self, env_obs=None):
        self._last_pos = None
        self._stuck_count = 0
        self._escape_index = 0
        self._last_debug_target = None
        self._last_debug_reason = None
        self._last_debug_final_act = None
        if self.logger is not None:
            self.logger.info("DIY reset called")
