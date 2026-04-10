#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery data class definitions.
智运无人机数据类定义。
"""


import numpy as np
from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


# ObsData: feature vector + legal action mask
# 观测数据：feature 为特征向量，legal_action 为合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: sampled action, greedy action, action probabilities, state value
# 动作数据：action 为采样动作，d_action 为贪心动作，prob 为动作概率，value 为状态价值
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData: int values are treated as dimensions by the framework
# 样本数据：字段值为 int 时框架自动按维度处理
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
    reward_sum=Config.VALUE_NUM,
)


def sample_process(list_sample_data):
    """Sample post-processing: fill next_value and compute GAE advantage.

    填充 next_value 并用 GAE 计算 advantage / reward_sum。
    """
    # Fill next_value: each step's next_value = next step's value
    # 填充 next_value：每步的 next_value = 下一步的 value
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Generalized Advantage Estimation (GAE).

    广义优势估计。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = sample.reward - sample.value + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value
