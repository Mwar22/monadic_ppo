# -*- coding:utf-8 -*-
###
# File:  dataclassutils.py
# Created Date: 24/05/2026 08:50:47
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 26/05/2026 09:14:39
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
import optax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from typing import Any, cast, Tuple, Callable, Self
from algorithms.ppo.agent.robot import RobotSharedData, obs_pipeline
from mujoco import mjx
from functools import partial
from utils2 import conv2jax_quat
from algorithms.ppo.agent.config import RangeConfig
from  enviroment import StateMonad

#faz um casting, para evitar o pylance reclamar de coisas como mjData, que vem do c/c++
mujoco: Any

@struct.dataclass
class RunningAvg:
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def init(cls, shape) -> Self:
        # Inicializamos com uma contagem pequena para evitar divisões por zero
        return cls(
            mean=jnp.zeros(shape),
            var=jnp.ones(shape),
            count=jnp.array(0.0)
        )
    
    @jax.jit
    def update(self, batch_obs: jax.Array):
        """
        Atualiza média e variância usando o Algoritmo de Welford vetorizado.
        batch_obs esperado: (num_envs, num_steps, obs_dim)
        """
        # achata as dimensões de batch (envs * steps) pra ficar mais facil
        obs_dim = batch_obs.shape[-1]
        batch_obs = batch_obs.reshape(-1, obs_dim)
        
        # obtem as estatisticas do batch atual
        batch_mean = jnp.mean(batch_obs, axis=0)
        batch_var = jnp.var(batch_obs, axis=0)
        batch_count = jnp.array(batch_obs.shape[0], dtype=jnp.float32)

        # lógica do algoritmo de welford
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * (self.count * batch_count / total_count)
        new_var = M2 / (total_count + 1e-6)

        # evita que gradientes sejam calculados
        return RunningAvg(mean=new_mean, var=new_var, count=total_count)

@struct.dataclass   
class RunningExponentialAvg:
    ema_value: jax.Array
    alpha: float

    @classmethod
    def init(cls, initial_ema_value: jax.Array = jnp.array(0), alpha: float =0.9) -> Self:
        return cls(initial_ema_value, alpha)
    
    def update(self, update_value:jax.Array):
        new_ema = (self.alpha * self.ema_value) + ((1 - self.alpha) * update_value)
        return RunningExponentialAvg(new_ema, self.alpha)
    

@struct.dataclass
class RunningParameters:
    obs_stat: RunningAvg
    progress: jax.Array
    update_threshold: float
    update_increment: float
    
    @classmethod
    def init(cls, obs_shape, update_threshold: float = 0.6, update_increment: float = 0.01) -> Self:
        # Inicializamos com uma contagem pequena para evitar divisões por zero
        return cls(
            RunningAvg.init(obs_shape),
            jnp.array(0.0),
            update_threshold,
            update_increment
        )
    
    def update(self, batch_obs: jax.Array, success_rate: jax.Array):
        new_obs_stat = self.obs_stat.update(batch_obs)

        progress = self.progress + jnp.where(success_rate > self.update_threshold, self.update_increment, 0.0)
        return RunningParameters(
            new_obs_stat,
            progress,
            self.update_threshold,
            self.update_increment,
        )
        
    

@struct.dataclass
class NetworkParameters:
    actor: Any
    critic: Any

    def update(self, actor_params, critic_params):
        return NetworkParameters(actor_params, critic_params)
    
@struct.dataclass
class NetworksSettings:
    obs_size: int
    action_size: int
    actor: nn.Module
    critic: nn.Module

    @classmethod
    def init(cls, 
        obs_size: int,
        action_size: int,
        actor: nn.Module,
        critic: nn.Module,
        params: Tuple[Any, Any],
    ):
        return cls(
            obs_size,
            action_size,
            actor,
            critic,
        )
    

@struct.dataclass
class TrainingSettings:
    network_settings: NetworksSettings
    epochs: int
    num_envs: int
    rollout_steps: int
    gamma: float
    gae_lambda: float

    robot_shared_data: RobotSharedData
    optimizer: optax.GradientTransformationExtraArgs
    optimizer_state: optax.OptState
    step_fn_creator: Callable
    target_success: float

    action_scale: float = 0.007
    obs_noise_scale: float = 0.001
    
    numberof_goals: int = 100
    early_stop: int = -1

    @property
    def active_numberof_goals(self)-> int:
        return self.early_stop if self.early_stop > 0 else self.numberof_goals


    @classmethod
    def init(
        cls,
        network_settings: NetworksSettings,
        network_params: NetworkParameters,
        robot_shared_settings: RobotSharedData,
        optimizer_creator: Callable[[int], optax.GradientTransformationExtraArgs],
        step_fn_creator:Callable[[Self, NetworkParameters], Callable],
        num_envs: int = 1,
        epochs: int = 1,
        action_scale:float = 0.001,
        obs_noise_scale:float = 0.01,
        numberof_goals: int = 10,
        early_stop:int = -1,
        rollout_steps: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        target_success: float = 0.6,
    ):
        # numero de passos para o escalonador de LR baseado na configuração
        total_steps = numberof_goals * epochs

        optimizer = optimizer_creator(total_steps)
        optimizer_params = optimizer.init(cast(optax.Params, network_params))

        
        return cls(
            network_settings,
            epochs,
            num_envs,
            rollout_steps,
            gamma,
            gae_lambda,
            robot_shared_settings,
            optimizer,
            optimizer_params,
            step_fn_creator,
            target_success,
            action_scale,
            obs_noise_scale,
            numberof_goals,
            early_stop
        )
    

@struct.dataclass
class BatchedBuffer:
    obs_buffer: jax.Array       # (num_envs, rollout_steps +1, *obs_shape)
    action_buffer: jax.Array    # (num_envs, rollout_steps +1, *action_shape)
    reward_buffer: jax.Array    # (num_envs, rollout_steps +1)
    logprob_buffer: jax.Array   # (num_envs, rollout_steps +1)
    ptr: jax.Array  # (num_envs,)
    done_flag: jax.Array # (num_envs,)  se atingiu o alvo ou colidiu/ultrapassou os limites
    stop_flag: jax.Array # (num_envs,)  para parar com o rollout. True se done_flag estiver true ou o buffer estiver cheio

    def __str__(self):
        return f"obs_buffer: {self.obs_buffer}\n \
        action_buffer: {self.action_buffer}\n \
        reward_buffer: {self.reward_buffer}\n \
        logprob_buffer: {self.logprob_buffer}\n \
        ptr: {self.ptr}\n \
        done_flag: {self.done_flag} \
        stop_flag: {self.done_flag}"
    
    @property
    def num_steps(self):
        return self.reward_buffer.shape[1]
    
    @property
    def is_full(self):
        return self.ptr >= self.num_steps
    
    @classmethod
    def init(cls, settings: TrainingSettings):
        num_envs = settings.num_envs
        rollout_steps = settings.rollout_steps +1
    
        return cls(
            jnp.zeros((num_envs, rollout_steps, settings.network_settings.obs_size), dtype=jnp.float32),
            jnp.zeros((num_envs, rollout_steps, settings.network_settings.action_size), dtype=jnp.float32),
            jnp.zeros((num_envs, rollout_steps), dtype=jnp.float32),
            jnp.zeros((num_envs, rollout_steps), dtype=jnp.float32),
            jnp.zeros((num_envs,), dtype=jnp.uint16),
            jnp.zeros((num_envs,), dtype=jnp.bool_),
            jnp.zeros((num_envs,), dtype=jnp.bool_),
        )
    
@struct.dataclass
class Goals:
    batched_pos: jax.Array
    batched_vel: jax.Array

    get_goal: Callable[[float, jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]

    @classmethod
    def init(cls, batched_rng: jax.Array, range_config: RangeConfig, progress: float = 0.0)->Tuple[jax.Array, Self]:

        def get_goal(progress: float, rng: jax.Array):
            rng1, position = range_config.position.sample_normal(rng, progress)
            rng2, position_velocities = range_config.position_velocities.sample_normal(rng1, progress)
            return rng2, position, position_velocities
        
        vmapped_get_goal = jax.vmap(partial(get_goal, progress))
        batched_rng, batched_pos, batched_vel = vmapped_get_goal(batched_rng)

        return batched_rng, cls(batched_pos, batched_vel, get_goal)

    def update(self, batched_rng: jax.Array, progress: float):
        vmapped_get_goal = jax.vmap(partial(self.get_goal, progress))
        batched_rng, batched_pos, batched_vel = vmapped_get_goal(batched_rng)
        return batched_rng, Goals(batched_pos, batched_vel, self.get_goal)
        

@struct.dataclass
class EnviromentsState:
    num_envs: int
    batched_rng: jax.Array
    batched_mjx_data: Any
    batched_goals: Goals
    batched_obs: jax.Array  #observação no tempo t
    batched_current_step: jax.Array
    batched_success_count: jax.Array
    batched_last_action: jax.Array

    @classmethod
    def init(cls, settings: TrainingSettings, rng: jax.Array, progress: float = 0.0)->Self:
        # (num_envs, features_dim)
        num_envs = settings.num_envs
        batched_rng = jax.random.split(rng, num_envs)

        mjx_data = mjx.make_data(settings.robot_shared_data._mjx_model)
        batched_mjx_data = jax.tree_util.tree_map(
            lambda x: jax.numpy.repeat(x[None], num_envs, axis=0),
            mjx_data
        )

        batched_rng, batched_goals = Goals.init(batched_rng, settings.robot_shared_data.range_config)

        temp_state = EnviromentsState(
            num_envs,
            batched_rng,
            batched_mjx_data,
            batched_goals,
            jnp.zeros((num_envs, settings.network_settings.obs_size)),
            jnp.zeros((num_envs,)),
            jnp.zeros((num_envs,)),
            jnp.zeros((num_envs, settings.network_settings.action_size)),
        )

        # Rode apenas o pipeline de observação para obter o estado REAL inicial
        # Isso garante que a primeira obs que o agente vê não seja zero
        runpar_init = RunningParameters.init((settings.network_settings.obs_size,), settings.numberof_goals)
        
        def get_single_obs(s):
            # StateMonad.pure({}) inicia o pdata como um dict vazio
            pipe = obs_pipeline(settings.robot_shared_data, runpar_init.obs_stat, StateMonad.pure({}), settings.obs_noise_scale)
            _, out_data = pipe.run(s)
            return out_data["obs"]

        # vmap mapeia 'get_single_obs' sobre a primeira dimensão de todos os arrays no temp_state
        initial_obs = jax.vmap(get_single_obs)(temp_state)

        return cls(
            num_envs,
            batched_rng,
            batched_mjx_data,
            batched_goals,
            initial_obs,
            jnp.zeros((num_envs,)),
            jnp.zeros((num_envs,)),
            jnp.zeros((num_envs, settings.network_settings.action_size)),
        )
    # Adicione essa assinatura apenas para o Pylance calar a boca:
    def replace(self, **kwargs: Any) -> "EnviromentsState":
        ...
       