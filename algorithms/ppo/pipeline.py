# -*- coding:utf-8 -*-
###
# File:  pipeline.py
# Created Date: 24/05/2026 02:01:33
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 25/05/2026 01:27:19
# Modified By: Lucas de Jesus 
# -----
# Copyright (c) 2026
# 
# This file is subject to the terms and conditions defined in
# the 'LICENSE.txt' file found in the root of this source tree.
# Please read LICENSE.txt for full copyright and licensing details.
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###

import jax
from agent import get_action, get_ctrl, shape_return
from robot import mujoco_step_monad, read_sensors_monad, as_array_monad
from task import calculate_task_errors, reward_and_termination_pipeline
from protocol import StepFunction


def create_step(rsd, network_settings, network_parameters, reward_config, action_scale, n_substeps, progress: jax.Array)->StepFunction:
    
    def step_fn(state):
        
        # escolhe a ação com base na observação anterior
        pl = get_action(network_settings, network_parameters)
        
        # transforma ação em comando de junta (ctrl)
        pl = pl.bind(lambda pdata: get_ctrl(rsd, pdata["action"], state["last_action"], action_scale))
        
        # aplica a ação física e roda o MuJoCo
        pl = pl.bind(lambda pdata: mujoco_step_monad(rsd, pdata["ctrl"], n_substeps))
        
        # faz a leitura dos sensores brutos após o movimento
        pl = pl.bind(lambda _: read_sensors_monad(rsd))
        
        # calcula os erros em relação ao target
        pl = pl.bind(lambda pdata: calculate_task_errors(state, pdata))
        
        # transformações extras
        pl = pl.bind(lambda pdata: as_array_monad(pdata))
        #pl = pl.bind(lambda pdata: normalize_obs(pdata, runpar.obs_stat))
        
        # função de recompensa e critério de término
        pl = pl.bind(lambda pdata: reward_and_termination_pipeline(progress, reward_config, pdata))
        
        # dá uma limpada na saida
        pl = pl.bind(shape_return)
        
        # roda o pipeline 
        return pl.run(state)

    return step_fn