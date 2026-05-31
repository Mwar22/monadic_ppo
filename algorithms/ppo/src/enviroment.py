# -*- coding:utf-8 -*-
###
# File:  mujoco.py
# Created Date: 26/05/2026 06:04:50
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 31/05/2026 09:53:39
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

    def sensor_data(self, sensor_name: str, mjx_data: mjx.Data)->jax.Array:
        ...

    def failed(self, mjx_data: mjx.Data)->jax.Array:
        ...


########################## Métodos aplicáveis a toda classe que segue MujocoEnv#####################################      
def mujoco_step(
    env: MujocoEnv,
    ctrl_action: jax.Array,
    mjx_data:mjx.Data,
)->mjx.Data:
    
    def single_step(data, _):
        data = data.replace(ctrl=ctrl_action)
        data = mjx.step(env.mjx_model, data)
        return data, None
    
    return jax.lax.scan(single_step, mjx_data, (), env.n_substeps)[0]

def mujoco_reset(
    env: MujocoEnv,
    def_qpos: jax.Array,
    mjx_data: mjx.Data,
)->mjx.Data:
    new_mjx_data = mjx_data.replace(
        qpos=def_qpos,
        qvel=jnp.zeros_like(mjx_data.qvel),
        qacc=jnp.zeros_like(mjx_data.qacc)
    )
    return mjx.forward(env.mjx_model, new_mjx_data)
#####################################################################################################################