#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery PPO algorithm implementation.
Standard PPO clip + value clip + entropy.
智运无人机 PPO 算法实现。标准 PPO clip + value clip + entropy。
"""


import os
import time

import torch
from agent_ppo.conf.conf import Config


class Algorithm:
    """Standard PPO algorithm : policy loss + value loss (clipped) + entropy bonus.

    标准 PPO 算法：策略损失 + 价值损失（裁剪）+ 熵奖励。
    """

    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.var_beta = Config.BETA_START

        self.last_report_monitor_time = 0

    def learn(self, list_sample_data):
        """Receive a list of SampleData and perform one PPO gradient step.

        接收 SampleData 列表，执行一步 PPO 梯度更新。
        """
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()

        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP_RANGE)

        self.optimizer.step()

        # Periodic monitoring report / 定期上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info["value_loss"], 4),
                "policy_loss": round(info["policy_loss"], 4),
                "entropy_loss": round(info["entropy_loss"], 4),
                "reward": round(info["reward_mean"], 4),
            }
            if self.logger:
                self.logger.info(
                    f"[LEARN] policy_loss={results['policy_loss']} "
                    f"value_loss={results['value_loss']} "
                    f"entropy={results['entropy_loss']} "
                    f"reward={results['reward']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

        return {"total_loss": total_loss.item()}

    def _compute_loss(
        self, logits, value_pred, legal_action, old_action, old_prob, advantage, old_value, reward_sum, reward
    ):
        """PPO loss computation.

        PPO 损失计算。

        total_loss = policy_loss + vf_coef * value_loss - beta * entropy_loss
        """
        # 1. Masked softmax → action probability distribution
        # 1. 掩码 softmax → 动作概率分布
        prob_dist = self._masked_softmax(logits, legal_action)

        # 2. Policy loss (PPO clip) / 策略损失（PPO clip）
        entropy_loss = -(prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_act_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_act_prob

        adv = advantage
        if adv.dim() > 1:
            adv = adv.squeeze(-1)
        adv = adv.unsqueeze(-1)

        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # 3. Value loss (clipped) / 价值损失（裁剪）
        v = value_pred
        ov = old_value
        tgt = reward_sum
        if v.dim() > 1:
            v = v.squeeze(-1)
        if ov.dim() > 1:
            ov = ov.squeeze(-1)
        if tgt.dim() > 1:
            tgt = tgt.squeeze(-1)

        v_clip = ov + (v - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = 0.5 * torch.maximum((tgt - v) ** 2, (tgt - v_clip) ** 2).mean()

        # 4. Combine / 合并
        total_loss = policy_loss + self.vf_coef * value_loss - self.var_beta * entropy_loss

        info = {
            "value_loss": value_loss.detach().cpu().item(),
            "policy_loss": policy_loss.detach().cpu().item(),
            "entropy_loss": entropy_loss.detach().cpu().item(),
            "reward_mean": reward.mean().detach().cpu().item(),
        }
        return total_loss, info

    def _masked_softmax(self, logits, legal_action):
        """Apply legal action mask then compute softmax.

        应用合法动作掩码后计算 softmax。
        """
        lmax, _ = torch.max(logits * legal_action, 1, True)
        logits = logits - lmax
        logits = logits * legal_action
        logits = logits + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(logits, dim=1)
