# -*- coding:utf-8 -*-
###
# File:  mujoco.py
# Created Date: 26/05/2026 06:04:50
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 30/05/2026 11:17:08
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
import jax.numpy as jnp
from typing import Protocol, runtime_checkable, cast
from mujoco import mjx
from flax import struct
from algorithms.ppo.utils.canonical_space import CanonicalSpace, transform_to_cs, transform_vel_to_cs
from algorithms.ppo.utils.monads import State

class EnvState(Protocol):
    def failed(self)->jax.Array:
        ...
    def reached(self)->jax.Array:
        ...

@runtime_checkable
class MujocoEnv(Protocol):
    @property
    def n_substeps(self)->int:
        ...
    @property
    def mjx_model(self)->mjx.Model:
        ...
    @property
    def world_space(self)->CanonicalSpace:
        ...
    @property
    def joint_space(self)->CanonicalSpace:
        ...

    @property
    def def_qpos(self)->jax.Array:
        ...
        
    @property
    def def_qvel(self)->jax.Array:
        ...

    def sensor_data(self, mjx_data: mjx.Data, sensor_name: str)->jax.Array:
        ...

    def failed(self, mjx_data: mjx.Data)->jax.Array:
        ...


########################## Métodos aplicáveis a toda classe que segue MujocoEnv#####################################      
def mujoco_step(
    env: MujocoEnv,
    mjx_data:mjx.Data,
    ctrl_action: jax.Array
)->mjx.Data:
    
    def single_step(data, _):
        data = data.replace(ctrl=ctrl_action)
        data = mjx.step(env.mjx_model, data)
        return data, None
    
    return jax.lax.scan(single_step, mjx_data, (), env.n_substeps)[0]

def mujoco_reset(
    env: MujocoEnv,
    mjx_data: mjx.Data,
    def_qpos: jax.Array,
)->mjx.Data:
    
    #adiciona uma dimensão de batch caso não tenha
    qpos_ready = def_qpos.reshape(-1, def_qpos.shape[-1])

    # qpos_ready.shape ou é (1, N) ou (batch_sz, N) test
    # qpos.shape ou é (N, ) ou é (batch_sz, N)
    num_repeats = mjx_data.qpos.shape[0] // qpos_ready.shape[0]
    # faz um broadcast de env.def_qpos para o mesmo shape de mjx_data.qpos, para lidar com o caso de mjx_data em batch
    broadcasted_qpos = jnp.broadcast_to(qpos_ready, mjx_data.qpos.shape)
    print(f"qpos.shape:{def_qpos.shape}, mjx_data.qpos.shape: {mjx_data.qpos.shape}, broadcast_def_qpos.shape{broadcasted_qpos.shape}")
    new_mjx_data = mjx_data.replace(
        qpos=broadcasted_qpos,
        qvel=jnp.zeros_like(mjx_data.qvel),
        qacc=jnp.zeros_like(mjx_data.qacc)
    )
    return mjx.forward(env.mjx_model, new_mjx_data)
#####################################################################################################################