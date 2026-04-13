#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
from common_python.utils.common_func import Frame
from agent_diy.feature.definition import (
    sample_process,
    reward_shaping,
)
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery



def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    if logger is not None:
        logger.info("===== ENTER workflow() =====")
    env, agent = envs[0], agents[0]
    last_save_model_time = time.time()
    episode_cnt = 0

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        if logger is not None:
            logger.error(f"usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    # Please write your DIY training process below.
    # 请在下方写你DIY的训练流程
    while True:
        # 定时保存一次
        now = time.time()
        if now - last_save_model_time >= 1800:
            agent.save_model()
            last_save_model_time = now

        # 1. reset 环境
        env_obs = env.reset(usr_conf)
        if handle_disaster_recovery(env_obs, logger):
            continue

        # 2. reset agent，并尝试加载最新“模型”
        agent.reset(env_obs)
        agent.load_model(id="latest")

        # 3. 第一次处理观测
        obs_data, remain_info = agent.observation_process(
            env_obs,
            preprocessor=None,
            extra_info=env_obs.get("extra_info", None),
        )

        episode_cnt += 1
        step = 0
        done = False
        if logger is not None:
            logger.info(f"DIY Episode {episode_cnt} start")

        # 4. 主循环：predict -> action_process -> env.step -> next obs
        while not done:
            act_data = agent.predict([obs_data])[0]
            act = agent.action_process(act_data)

            env_reward, env_obs = env.step(act)
            if handle_disaster_recovery(env_obs, logger):
                break

            step += 1
            terminated = env_obs.get("terminated", False)
            truncated = env_obs.get("truncated", False)
            done = terminated or truncated

            if done:
                if logger is not None:
                    logger.info(f"DIY Episode {episode_cnt} done, total_steps={step}")
                break

            obs_data, remain_info = agent.observation_process(
                env_obs,
                preprocessor=None,
                extra_info=env_obs.get("extra_info", None),
            )
        
    # At the start of each game, support loading the latest model file
    # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
    # agent.load_model(id="latest")

    # model saving
    # 保存模型
    # agent.save_model()

    # return
