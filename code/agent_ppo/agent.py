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
        self.recent_positions = deque(maxlen=6)
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1
        self.recent_positions.clear()

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
        self.recent_positions.append(tuple(self.preprocessor.cur_pos))
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

    def _is_geom_stuck(self):
        # 只把 recent_positions 当成“几何卡住触发器”
        return len(self.recent_positions) >= 6 and len(set(self.recent_positions)) <= 3

    def _count_forward_second_steps(self, hero_x, hero_z, next_x, next_z):
        # 统计一步后位置还能不能在两步内继续“往前延伸”
        # 这里不看 enemy，只看局部几何连续性
        count = 0
        recent_set = set(self.recent_positions)

        for act in range(Config.ACTION_NUM):
            dx2, dz2 = self.preprocessor._act_to_delta(act)
            second_x = next_x + dx2
            second_z = next_z + dz2

            row = 10 + (second_z - hero_z)
            col = 10 + (second_x - hero_x)

            if not self.preprocessor._is_local_cell_passable(row, col):
                continue

            second_pos = (second_x, second_z)

            # 直接退回当前点，不算前向连续性
            if second_pos == (hero_x, hero_z):
                continue

            # 又回到最近小圈子里，也不算真正往前延伸
            if second_pos in recent_set:
                continue

            count += 1

        return count
    def _filter_safe_legal_action(self, legal_action):
        """用局部地图过滤明显撞墙动作，并避开可见敌机的危险半径。"""
        original = list(legal_action)
        wall_filtered = list(legal_action)

        local_map = self.preprocessor.local_map
        hero_x, hero_z = self.preprocessor.cur_pos
        visible_npcs = getattr(self.preprocessor, "visible_npcs", [])

        # 第一层：过滤一步后撞墙
        if isinstance(local_map, list) and len(local_map) > 0:
            for act in range(Config.ACTION_NUM):
                if int(wall_filtered[act]) != 1:
                    continue

                dx, dz = self.preprocessor._act_to_delta(act)
                row = 10 + dz
                col = 10 + dx

                if not self.preprocessor._is_local_cell_passable(row, col):
                    wall_filtered[act] = 0

        # 如果墙过滤后已经全没了，只能回退到原始 legal_action
        if sum(wall_filtered) == 0:
            return original

        # 第二层：过滤一步后进入敌机危险区
        enemy_filtered = list(wall_filtered)
        if len(visible_npcs) > 0:
            danger_r2 = 9  # 半径 3 格，先用保守值
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

        base_filtered = enemy_filtered if sum(enemy_filtered) > 0 else wall_filtered
        geometry_filtered = list(base_filtered)

        # 只有“疑似几何卡住”时，才启用两步几何判断
        if self._is_geom_stuck():
            for act in range(Config.ACTION_NUM):
                if int(geometry_filtered[act]) != 1:
                    continue

                dx, dz = self.preprocessor._act_to_delta(act)
                next_x = hero_x + dx
                next_z = hero_z + dz

                # 两步后仍有前向连续性：更像狭窄通道，放行
                # 两步后没有前向连续性：更像U形/口袋，屏蔽
                forward_count = self._count_forward_second_steps(hero_x, hero_z, next_x, next_z)
                if forward_count == 0:
                    geometry_filtered[act] = 0

        if sum(geometry_filtered) > 0:
            return geometry_filtered
        if sum(enemy_filtered) > 0:
            return enemy_filtered
        if sum(wall_filtered) > 0:
            return wall_filtered
        return original

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
