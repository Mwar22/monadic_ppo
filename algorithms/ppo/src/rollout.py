# -*- coding:utf-8 -*-
###
# File:  rollout.py
# Created Date: 25/05/2026 11:54:02
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 04/06/2026 11:05:37
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
from mujoco import mjx
from flax import struct, nnx
from .agent import Agent, MujocoEnv

class RolloutBuffer(struct.PyTreeNode):
    policy_obs: jax.Array       # (rollout_steps +1, num_enviroments, *obs_shape)
    value_obs: jax.Array
    actions: jax.Array    # (rollout_steps +1, num_enviroments, *action_shape)
    values: jax.Array    # (rollout_steps +1, num_enviroments,)
    rewards: jax.Array    # (rollout_steps +1, num_enviroments,)
    logprobs: jax.Array   # (rollout_steps +1, num_enviroments,)
    dones: jax.Array       #(rollout_steps +1, num_enviroments,)
  
def new_buffer(
    num_enviroments: int,
    rollout_steps: int,
    policy_obs_size:int,
    value_obs_size:int,
    action_size: int
)->RolloutBuffer:
    """
    Cria um novo buffer para armazenar informações de um rollout.

    Parameters
    ----------
    num_enviroments : int
    rollout_steps : int
    obs_size : int
    action_size : int

    Returns
    -------
    RolloutBuffer
    """
    _num_enviroments = int(jnp.maximum(jnp.array(num_enviroments), 1.0))
    _rollout_steps = int(jnp.maximum(jnp.array(rollout_steps), 1.0) + 1)
    _action_size = int(jnp.maximum(jnp.array(action_size), 1.0))
    _policy_obs_size = int(jnp.maximum(jnp.array(policy_obs_size), 1.0))
    _value_obs_size = int(jnp.maximum(jnp.array(value_obs_size), 1.0))

    return RolloutBuffer(
        jnp.zeros((_rollout_steps, _num_enviroments, _policy_obs_size), dtype=jnp.float32),
        jnp.zeros((_rollout_steps, _num_enviroments, _value_obs_size), dtype=jnp.float32),
        jnp.zeros((_rollout_steps, _num_enviroments, _action_size), dtype=jnp.float32),
        jnp.zeros((_rollout_steps, _num_enviroments),  dtype=jnp.float32),
        jnp.zeros((_rollout_steps, _num_enviroments),  dtype=jnp.float32),
        jnp.zeros((_rollout_steps, _num_enviroments), dtype=jnp.float32),
        jnp.zeros((_rollout_steps, _num_enviroments), dtype=jnp.bool),
    )

def add_on_buffer(
    buffer: RolloutBuffer, 
    index: int,
    policy_obs: jax.Array,
    value_obs: jax.Array,
    action: jax.Array,
    value: jax.Array,
    reward: jax.Array,
    logprob: jax.Array,
    done: jax.Array,
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
        policy_obs = buffer.policy_obs.at[index].set(policy_obs),
        value_obs = buffer.value_obs.at[index].set(value_obs),
        actions = buffer.actions.at[index].set(action),
        values = buffer.values.at[index].set(value),
        rewards = buffer.rewards.at[index].set(reward),
        logprobs = buffer.logprobs.at[index].set(logprob),
        dones = buffer.dones.at[index].set(done),
    )

##################################################### TEST#############################################



@nnx.jit(static_argnames=['rollout_steps'])
def rollout(
    agent: Agent,
    enviroment: MujocoEnv,
    rngs: nnx.Rngs,
    mjx_data: mjx.Data,
    target: jax.Array,
    rollout_steps: int,
    buffer: RolloutBuffer,
):
    #cria vmaps para para funcionar com dados em batch
    vmap_agent_reset = jax.vmap(agent.reset, in_axes=(None, None, 0))
    vmap_agent_step = jax.vmap(agent.step, in_axes = (None, 0, 0, 0))
    vmap_agent_compose_obs = jax.vmap(agent.compose_obs, in_axes=(None, 0))

    #reseta o agente e coleta as primeiras observações do ambiente
    _, mjx_data = vmap_agent_reset(enviroment, rngs, mjx_data)

    policy_obs, value_obs = vmap_agent_compose_obs(enviroment, mjx_data)


    def rollout_step(carry, step):
        policy_obs, value_obs, mjx_data, buffer, rngs = carry

        #obtem a ação, e avalia logprob e entropia relativas
        actions = agent.policy.sample(policy_obs, rngs)
        values = agent.value(value_obs)

        #logprob segundo a politica atual
        logprob, _ = agent.policy.evaluate_actions(policy_obs, actions)
        
        #avança o agente
        step_data, mjx_data = vmap_agent_step(enviroment, actions, target, mjx_data)
        
        # guarda no buffer
        buffer = add_on_buffer(buffer, step, policy_obs, value_obs, actions, values, step_data.reward, logprob, step_data.done)
        
        #obtem a proxima observação
        policy_obs, value_obs = vmap_agent_compose_obs(enviroment, mjx_data)
        return (policy_obs, value_obs, mjx_data, buffer, rngs), step_data
    
    (policy_obs, value_obs, mjx_data, buffer, rngs), data = jax.lax.scan(rollout_step, (policy_obs, value_obs, mjx_data, buffer, rngs), jnp.arange(rollout_steps+1))

    return buffer, data, mjx_data
