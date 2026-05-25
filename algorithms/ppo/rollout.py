# -*- coding:utf-8 -*-
###
# File:  rollout.py
# Created Date: 25/05/2026 11:54:02
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 25/05/2026 02:16:52
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
from functools import partial
from flax import struct
from enviroment.step import StepFunction, StepData
from enviroment.state import EnviromentState

class RolloutBuffer(struct.PyTreeNode):
    observations: jax.Array       # (rollout_steps +1, num_envs, *obs_shape)
    actions: jax.Array    # (rollout_steps +1, num_envs, *action_shape)
    rewards: jax.Array    # (rollout_steps +1, num_envs,)
    logprobs: jax.Array   # (rollout_steps +1, num_envs,)
  
def new_buffer(
    num_envs: int,
    rollout_steps: int,
    obs_size:int,
    action_size: int
)->RolloutBuffer:
    """
    Cria um novo buffer para armazenar informações de um rollout.

    Parameters
    ----------
    num_envs : int
    rollout_steps : int
    obs_size : int
    action_size : int

    Returns
    -------
    RolloutBuffer
    """
    _num_envs = jnp.maximum(jnp.array(num_envs), 1.0)
    _rollout_steps = jnp.maximum(jnp.array(rollout_steps), 1.0) + 1
    _action_size = jnp.maximum(jnp.array(action_size), 1.0)
    _observation_size = jnp.maximum(jnp.array(obs_size), 1.0)

    return RolloutBuffer(
        jnp.zeros((_rollout_steps, _num_envs, _observation_size), dtype=jnp.float32),
        jnp.zeros((_rollout_steps, _num_envs, _action_size), dtype=jnp.float32),
        jnp.zeros((_rollout_steps, _num_envs),  dtype=jnp.float32),
        jnp.zeros((_rollout_steps, _num_envs), dtype=jnp.float32),
    )

def add_on_buffer(
    buffer: RolloutBuffer, 
    index: jax.Array,
    data: StepData
)->RolloutBuffer:
    """
    Adiciona dados no buffer em uma determinada posição

    Parameters
    ----------
    buffer : RolloutBuffer
    step : int
    batch_obs : jax.Array
    batch_action : jax.Array
    batch_reward : jax.Array
    batch_logprob : jax.Array

    Returns
    -------
    RolloutBuffer atualizado
    """
    return buffer.replace(
        observations = buffer.observations.at[index].set(data.observation),
        actions = buffer.actions.at[index].set(data.action),
        rewards = buffer.rewards.at[index].set(data.reward),
        logprobs = buffer.logprobs.at[index].set(data.logprob),
    )

def rollout(
    num_envs, rollout_steps, obs_size, action_size, step_fn:StepFunction, init_state: EnviromentState
):
    def rollout_step(
        state: EnviromentState,
        index: jax.Array,
        buffer: RolloutBuffer,
    ):
        new_state, data = step_fn(state)
        new_buffer = add_on_buffer(buffer, index, data)
        return new_state, new_buffer

    def scan_fn(carry, index):
        state, buffer = carry
        new_state, new_buffer = rollout_step(state, index, buffer)
        return (new_state, new_buffer), None

    # cria um buffer vazio
    buffer  = new_buffer(num_envs, rollout_steps, obs_size, action_size)

    (final_state, final_buffer), _ = jax.lax.scan(
        scan_fn, (init_state, buffer), None, length=rollout_steps
    )
    return final_state, final_buffer