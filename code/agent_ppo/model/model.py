#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery policy network.
智运无人机策略网络。
"""


import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features: int, out_features: int):
    """Create and initialize a linear layer with orthogonal init.

    创建并初始化线性层（正交初始化）。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight)
    nn.init.zeros_(fc.bias)
    return fc


class MLP(nn.Module):
    """Multi-layer perceptron.

    多层感知器。
    """

    def __init__(
        self,
        fc_feat_dim_list: list,
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i + 1}", fc)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_relu{i + 1}", non_linearity())

    def forward(self, x):
        return self.fc_layers(x)


class Model(nn.Module):
    """Actor-Critic model for Drone Delivery.

    智运无人机 Actor-Critic 模型。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "drone_delivery"
        self.device = device

        feature_len = Config.FEATURE_LEN
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM
        hidden_dim = 64

        # Backbone network / 主干网络
        self.backbone = MLP(
            [feature_len, hidden_dim, hidden_dim],
            "backbone",
            non_linearity_last=True,
        )

        # Actor head (direct projection, no hidden layer) / Actor 输出头（直接投影，无隐藏层）
        self.actor_head = make_fc_layer(hidden_dim, action_num)

        # Critic head (direct projection, small init) / Critic 输出头（直接投影，小初始化）
        self.critic_head = make_fc_layer(hidden_dim, value_num)
        nn.init.orthogonal_(self.critic_head.weight, gain=0.01)

    def forward(self, s, inference=False):
        """Forward pass.

        前向传播。
        """
        feat = s.to(torch.float32)
        hidden = self.backbone(feat)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
