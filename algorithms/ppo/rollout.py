# -*- coding:utf-8 -*-
###
# File:  rollout.py
# Created Date: 25/05/2026 11:54:02
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 26/05/2026 10:22:33
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
from enviroment.step import StepData
from networks.actor import Actor
from enviroment.mujoco import MujocoEnv, mujoco_step, mujoco_reset, mujoco_sensors

class RolloutBuffer(struct.PyTreeNode):
    observations: jax.Array       # (rollout_steps +1, num_envs, *obs_shape)
    tool_pos: jax.Array  # (rollout_steps +1, num_envs, 3)
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
    _num_envs = int(jnp.maximum(jnp.array(num_envs), 1.0))
    _rollout_steps = int(jnp.maximum(jnp.array(rollout_steps), 1.0) + 1)
    _action_size = int(jnp.maximum(jnp.array(action_size), 1.0))
    _observation_size = int(jnp.maximum(jnp.array(obs_size), 1.0))

    return RolloutBuffer(
        jnp.zeros((_rollout_steps, _num_envs, _observation_size), dtype=jnp.float32),
        jnp.zeros((_rollout_steps, _num_envs, 3), dtype=jnp.float32),
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
        tool_pos = buffer.tool_pos.at[index].set(data.tool_pos),
        actions = buffer.actions.at[index].set(data.action),
        rewards = buffer.rewards.at[index].set(data.reward),
        logprobs = buffer.logprobs.at[index].set(data.logprob),
    )

####################################################################################################
def update_obs(rng: jax.Array, obs:jax.Array, obs_noise=0.0):
    rng, rng1 = jax.random.split(rng)
    obs_processed = jnp.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    noise = jax.random.uniform(rng, obs_processed.shape, minval=-1.0, maxval=1.0)
    obs_processed = obs_processed + (obs_noise * noise)
    return rng1, obs_processed

def action_sample_betadist(
    rng: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
):
    actions = jax.random.beta(rng, alpha, beta)
    logprobs = jax.scipy.stats.beta.logpdf(actions, alpha, beta)
    return actions, jnp.sum(logprobs, axis=-1)
    
def get_action(
    rng: jax.Array,
    obs: jax.Array,
    actor: Actor,
):
    alpha, beta = actor(obs)
    action, logprob = action_sample_betadist(rng, alpha, beta)
    return action, logprob
    
def get_ctrl(mjx_data: mjx.Data, action: jax.Array, max_step_rads = 0.05 ):
   
    #muda a escala das ações de [0, 1] para [-max_step_rad, max_step_rad]
    delta = (2*action - 1) * max_step_rads  
    return mjx_data.ctrl + delta


def get_reward(tool_pos: jax.Array, qpos: jax.Array, qvel:jax.Array, target:jax.Array)->jax.Array:
    error = target - tool_pos
    return -jnp.linalg.norm(error, ord=2)

####################################################################################################
def batched_step_wrapper(
    rng: jax.Array,
    mjx_data: mjx.Data,
    target: jax.Array,
    fail_flag: jax.Array,
    index: jax.Array,
    env: MujocoEnv,       
    actor_graph: nnx.GraphDef, # Replaces Actor
    actor_state: nnx.State     # Replaces Actor
):
    #reconstroi o ator
    actor = nnx.merge(actor_graph, actor_state)


    fail_flag |= env.failed(mjx_data)

    def finished(carry):
        rng, mjx_data, fail_flag= carry
        new_mjx_data = mujoco_reset(env, mjx_data, env.def_qpos)
        return (rng, new_mjx_data, jnp.zeros(env.observation_size), jnp.zeros(3), 0.5*jnp.ones(env.action_size), jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32), fail_flag, index)
    
    def not_finished(carry):
        rng, mjx_data, fail_flag = carry

        #lê o sensor e compôe o tensor de observação
        tool_pos, qpos, qvel = mujoco_sensors(env, mjx_data)
        obs = jnp.concatenate((qpos, qvel, target))

        #obtem uma nova ação
        rng, rng1 = jax.random.split(rng)
        action, logprob = get_action(rng1, obs, actor)

        #novo valor de controle para a dada ação
        ctrl_action  = get_ctrl(mjx_data, action)

        #avança a fisica, lê os sensores novamente e calcula as recompensas
        mjx_data = mujoco_step(env, mjx_data, ctrl_action)
        tool_pos, qpos, qvel = mujoco_sensors(env, mjx_data)
        reward = get_reward(tool_pos, qpos, qvel, target)

        return rng, mjx_data, obs, tool_pos, action, reward, logprob, fail_flag, index +1

    return jax.lax.cond(fail_flag, finished, not_finished, (rng, mjx_data, fail_flag))

def rollout_step(
        rng: jax.Array,
        mjx_data: mjx.Data,
        env: MujocoEnv,
        targets: jax.Array,
        fail_flag: jax.Array,
        index: jax.Array,
        buffer: RolloutBuffer,
        actor_graph: nnx.GraphDef, 
        actor_state: nnx.State   
    ):
        vmapped_step = jax.vmap(
            batched_step_wrapper,
            in_axes=(
                0,     # rng: diferente para cada ambiente
                0,     # mjx_data: cada ambiente tem o seu estado unico
                0,     # target: cada ambiente tem um alvo diferente
                0,     # fail_flag: cada ambiente pode falhar em momentos distintos
                0,     # index: aponta para prox timestep
                None,  # MujocoEnv é um dataclass que é compartilhado
                None,  # actor_graph: compartilhado (arquitetura estática)
                None   # actor_state: compartilhado (todos os ambientes usam os mesmos pesos para a política)
            )
        )
        rng, mjx_data, obs, tool_pos, action, reward, logprob, fail_flag, index = vmapped_step(
            rng, mjx_data, targets, fail_flag, index , env, actor_graph, actor_state
        )
        new_buffer = add_on_buffer(buffer, index, StepData(obs, tool_pos, action, reward, logprob))
        return rng, mjx_data, new_buffer, fail_flag, index


##################################################### TEST#############################################
