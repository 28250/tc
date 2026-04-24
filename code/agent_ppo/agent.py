#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery Agent class. Inherits BaseAgent, implements PPO inference, training, and model save/load.
智运无人机 Agent 主类。继承 BaseAgent，实现 PPO 推理、训练、存取模型。
"""

import os
from collections import deque
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1

        # 几何陷阱短时接管用的最小状态
        self.recent_positions = deque(maxlen=6)
        self.recent_goal_dists = deque(maxlen=4)
        self.escape_mode = False
        self.escape_steps_left = 0
        self.geom_stuck_count = 0
        self.enemy_cooldown = 0

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1
        self.recent_positions.clear()
        self.recent_goal_dists.clear()
        self.escape_mode = False
        self.escape_steps_left = 0
        self.geom_stuck_count = 0
        self.enemy_cooldown = 0

    def _forward(self, feature, legal_action):
        """Gradient-free forward pass, returns (logits, value).

        无梯度前向推理，返回 (logits, value)。
        """
        self.model.set_eval_mode()
        obs_t = (
            torch.tensor(np.array([feature]), dtype=torch.float32).view(1, Config.DIM_OF_OBSERVATION).to(self.device)
        )
        with torch.no_grad():
            logits, value = self.model(obs_t, inference=True)
        return logits.cpu().numpy()[0], value.cpu().numpy()[0]

    def predict(self, list_obs_data):
        """Training inference: sample action from probability distribution.

        训练时推理：按概率采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value = self._forward(feature, legal_action)
        legal_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_np)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [ActData(action=[action], d_action=[d_action], prob=list(prob), value=value)]

    def exploit(self, env_obs):
        """Evaluation inference: greedy action selection.

        评估时推理：贪心选最大概率动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        if obs_data is None:
            return 0
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Delegate to Algorithm for PPO update.

        委托给 Algorithm 执行 PPO 更新。
        """
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, env_obs):
        """Convert raw env observation to ObsData + remain_info.

        将原始环境观测转换为 ObsData + remain_info。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)

        # 先记录当前位置和当前目标距离，供 escape_mode 判断
        self.recent_positions.append(tuple(self.preprocessor.cur_pos))
        cur_goal, _ = self.preprocessor._select_active_goal()
        cur_goal_dist = self.preprocessor._get_goal_dist(cur_goal)
        self.recent_goal_dists.append(cur_goal_dist)

        # 先更新状态，再过滤动作
        self._update_escape_mode(legal_action)
        legal_action = self._filter_safe_legal_action(legal_action)

        # feature 里的 legal_action 8 维同步改成过滤后的 mask
        # layout: hero(4) + station(7) + legal(8) + ...
        feature = np.array(feature, dtype=float)
        feature[11:19] = np.array(legal_action, dtype=float)

        remain_info = {"reward": reward}
        return (
            ObsData(feature=list(feature), legal_action=legal_action),
            remain_info,
        )
    
    def _filter_wall_only(self, legal_action):
        original = list(legal_action)
        wall_filtered = list(legal_action)

        local_map = self.preprocessor.local_map
        if isinstance(local_map, list) and len(local_map) > 0:
            for act in range(Config.ACTION_NUM):
                if int(wall_filtered[act]) != 1:
                    continue

                dx, dz = self.preprocessor._act_to_delta(act)
                row = 10 + dz
                col = 10 + dx

                if not self.preprocessor._is_local_cell_passable(row, col):
                    wall_filtered[act] = 0

        if sum(wall_filtered) == 0:
            return original
        return wall_filtered

    def _filter_enemy_only(self, wall_filtered):
        enemy_filtered = list(wall_filtered)

        hero_x, hero_z = self.preprocessor.cur_pos
        visible_npcs = getattr(self.preprocessor, "visible_npcs", [])

        if len(visible_npcs) > 0:
            danger_r2 = 9
            for act in range(Config.ACTION_NUM):
                if int(enemy_filtered[act]) != 1:
                    continue

                dx, dz = self.preprocessor._act_to_delta(act)
                next_x = hero_x + dx
                next_z = hero_z + dz

                for npc_x, npc_z in visible_npcs:
                    dist2 = (next_x - npc_x) ** 2 + (next_z - npc_z) ** 2
                    if dist2 <= danger_r2:
                        enemy_filtered[act] = 0
                        break

        return enemy_filtered

    def _has_enemy_pressure(self, wall_filtered, enemy_filtered):
        hero_x, hero_z = self.preprocessor.cur_pos

        # 敌机很近，说明当前更像正常规避，不要误触 escape_mode
        for npc_x, npc_z in getattr(self.preprocessor, "visible_npcs", []):
            dist2 = (hero_x - npc_x) ** 2 + (hero_z - npc_z) ** 2
            if dist2 <= 25:
                return True

        # 敌机过滤确实裁掉了动作，也认为当前有 enemy pressure
        return sum(enemy_filtered) < sum(wall_filtered)

    def _is_position_concentrated(self):
        return len(self.recent_positions) >= 6 and len(set(self.recent_positions)) <= 2

    def _is_goal_progress_stalled(self):
        if len(self.recent_goal_dists) < 4:
            return False
        if any(d is None for d in self.recent_goal_dists):
            return False

        # 最近几步目标距离几乎没下降，才算停滞
        return (self.recent_goal_dists[0] - self.recent_goal_dists[-1]) < 1.0

    def _count_forward_second_steps(self, hero_x, hero_z, next_x, next_z):
        count = 0
        recent_tail = set(list(self.recent_positions)[-2:])

        for act in range(Config.ACTION_NUM):
            dx2, dz2 = self.preprocessor._act_to_delta(act)
            second_x = next_x + dx2
            second_z = next_z + dz2

            row = 10 + (second_z - hero_z)
            col = 10 + (second_x - hero_x)

            if not self.preprocessor._is_local_cell_passable(row, col):
                continue

            second_pos = (second_x, second_z)

            # 直接退回当前点，不算前向延伸
            if second_pos == (hero_x, hero_z):
                continue

            # 只排除最近两步，避免误伤必要回退和窄通道
            if second_pos in recent_tail:
                continue

            count += 1

        return count
    
    def _is_corridor_like(self, base_filtered, cur_goal, cur_goal_dist):
        # 狭窄通道保护：
        # 只要存在一个动作同时满足：
        # 1) 两步后还有前向连续性
        # 2) 走这一步后更接近当前目标
        # 就认为当前更像“狭窄但可通”的通道，而不是口袋陷阱
        if cur_goal is None or cur_goal_dist is None:
            return False

        hero_x, hero_z = self.preprocessor.cur_pos
        goal_pos = (cur_goal["pos"]["x"], cur_goal["pos"]["z"])

        for act in range(Config.ACTION_NUM):
            if int(base_filtered[act]) != 1:
                continue

            dx, dz = self.preprocessor._act_to_delta(act)
            next_x = hero_x + dx
            next_z = hero_z + dz

            forward_count = self._count_forward_second_steps(hero_x, hero_z, next_x, next_z)
            if forward_count <= 0:
                continue

            next_goal_dist = np.sqrt((goal_pos[0] - next_x) ** 2 + (goal_pos[1] - next_z) ** 2)
            if next_goal_dist < cur_goal_dist:
                return True

        return False

    def _update_escape_mode(self, legal_action):
        wall_filtered = self._filter_wall_only(legal_action)
        enemy_filtered = self._filter_enemy_only(wall_filtered)

        enemy_pressure = self._has_enemy_pressure(wall_filtered, enemy_filtered)
        geom_like = self._is_position_concentrated() and self._is_goal_progress_stalled()
        wall_trap_like = sum(wall_filtered) <= 3

        base_filtered = enemy_filtered if sum(enemy_filtered) > 0 else wall_filtered
        cur_goal, _ = self.preprocessor._select_active_goal()
        cur_goal_dist = self.preprocessor._get_goal_dist(cur_goal)
        corridor_like = self._is_corridor_like(base_filtered, cur_goal, cur_goal_dist)

        # 敌机刚离开时，先给两帧冷却，避免把正常规避残留误判成几何陷阱
        if enemy_pressure:
            self.enemy_cooldown = 2
        else:
            self.enemy_cooldown = max(0, self.enemy_cooldown - 1)

        # 正常状态 -> 进入 escape_mode：要求更严格
        if not self.escape_mode:
            if geom_like and wall_trap_like and (not corridor_like) and (not enemy_pressure) and self.enemy_cooldown == 0:
                self.geom_stuck_count += 1
            else:
                self.geom_stuck_count = 0

            if self.geom_stuck_count >= 3:
                self.escape_mode = True
                self.escape_steps_left = 3
                self.geom_stuck_count = 0
            return

        # escape_mode -> 退出：脱离小圈子 / 目标重新推进 / 敌机施压 / 步数到
        should_exit = False

        if enemy_pressure:
            should_exit = True
        elif not self._is_position_concentrated():
            should_exit = True
        if len(self.recent_goal_dists) >= 2:
            prev_dist = self.recent_goal_dists[-2]
            cur_dist = self.recent_goal_dists[-1]
            if prev_dist is not None and cur_dist is not None and cur_dist < prev_dist:
                should_exit = True
        if self.escape_steps_left <= 0:
            should_exit = True

        if should_exit:
            self.escape_mode = False
            self.escape_steps_left = 0
        else:
            self.escape_steps_left -= 1

    def _filter_escape_mode(self, enemy_filtered, wall_filtered, original):
        base = enemy_filtered if sum(enemy_filtered) > 0 else wall_filtered

        hero_x, hero_z = self.preprocessor.cur_pos
        recent_set = set(self.recent_positions)
        recent_tail = set(list(self.recent_positions)[-2:])

        cur_goal, _ = self.preprocessor._select_active_goal()
        cur_goal_dist = self.preprocessor._get_goal_dist(cur_goal)

        scored = []

        for act in range(Config.ACTION_NUM):
            if int(base[act]) != 1:
                continue

            dx, dz = self.preprocessor._act_to_delta(act)
            next_x = hero_x + dx
            next_z = hero_z + dz
            next_pos = (next_x, next_z)

            score = 0

            # 尽量离开当前小圈子
            if next_pos not in recent_set:
                score += 2

            # 两步后还能继续延伸，更像窄通道，不像口袋
            forward_count = self._count_forward_second_steps(hero_x, hero_z, next_x, next_z)
            if forward_count > 0:
                score += 2

            # 仍在朝目标改善，给一点偏好
            if cur_goal is not None and cur_goal_dist is not None:
                goal_pos = (cur_goal["pos"]["x"], cur_goal["pos"]["z"])
                next_goal_dist = np.sqrt((goal_pos[0] - next_x) ** 2 + (goal_pos[1] - next_z) ** 2)
                if next_goal_dist < cur_goal_dist:
                    score += 1

            # 立刻回到最近两步，减分
            if next_pos in recent_tail:
                score -= 2

            scored.append((score, act))

        if not scored:
            if sum(enemy_filtered) > 0:
                return enemy_filtered
            if sum(wall_filtered) > 0:
                return wall_filtered
            return list(original)

        scored.sort(reverse=True)
        best_score = scored[0][0]

        # 只保留 top-1 / top-2 动作，让 PPO 在少量更像脱困的动作里选
        keep = [act for score, act in scored if score >= best_score - 1][:2]

        filtered = [0] * Config.ACTION_NUM
        for act in keep:
            filtered[act] = 1

        return filtered

    def _filter_safe_legal_action(self, legal_action):
        original = list(legal_action)
        wall_filtered = self._filter_wall_only(legal_action)
        enemy_filtered = self._filter_enemy_only(wall_filtered)

        # 正常状态：尽量少干预，保持原来更直线的风格
        if not self.escape_mode:
            if sum(enemy_filtered) > 0:
                return enemy_filtered
            if sum(wall_filtered) > 0:
                return wall_filtered
            return original

        # 陷阱态：短时启发式接管
        return self._filter_escape_mode(enemy_filtered, wall_filtered, original)

    def action_process(self, act_data, is_stochastic=True):
        """Extract int action from ActData and update last_action.

        从 ActData 提取整数动作，并更新 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        if path is None:
            return

        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if not os.path.exists(model_file_path):
            if self.logger is not None:
                self.logger.info(f"skip load model, file not found: {model_file_path}")
            return

        state_dict = torch.load(model_file_path, map_location=self.device)
        model_state_dict = self.model.state_dict()
        mismatched = []
        for key, value in state_dict.items():
            if key in model_state_dict and hasattr(value, "shape") and model_state_dict[key].shape != value.shape:
                mismatched.append(f"{key}:{tuple(value.shape)}->{tuple(model_state_dict[key].shape)}")

        if mismatched:
            if self.logger is not None:
                self.logger.warning(
                    f"skip load model due to incompatible checkpoint: {model_file_path}, mismatched={mismatched}"
                )
            return

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            if self.logger is not None:
                self.logger.warning(
                    f"skip load model due to invalid checkpoint format: {model_file_path}, err={exc}"
                )
            return

        if self.logger is not None:
            self.logger.info(f"load model {model_file_path} successfully")

    def _legal_soft_max(self, logits, legal_action):
        """Apply legal action mask and compute normalized probabilities.

        对 logits 应用合法动作掩码并计算归一化概率。
        """
        _w, _e = 1e20, 1e-5
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample from probability distribution (or argmax).

        从概率分布中采样（或取最大值）。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
