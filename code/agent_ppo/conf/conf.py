#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery PPO configuration.
智运无人机 PPO 配置。
"""


class Config:

    # Feature dimensions / 特征维度
    HERO_STATE_DIM = 4
    STATION_DIM = 7 * 1
    LEGAL_ACT_DIM = 8
    INDICATOR_DIM = 3

    FEATURES = [
        HERO_STATE_DIM,
        STATION_DIM,
        LEGAL_ACT_DIM,
        INDICATOR_DIM,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间
    ACTION_NUM = 8
    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    # Value head (single) / 价值头（单头）
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    # Moderate entropy coeff / 中等熵系数
    BETA_START = 0.005
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    GRAD_CLIP_RANGE = 0.5
    USE_GRAD_CLIP = True

    NUMB_HEAD = 1
