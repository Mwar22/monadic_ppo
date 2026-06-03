# -*- coding:utf-8 -*-
###
# File:  test_loss.py
# Created Date: 30/05/2026 04:12:14
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 31/05/2026 12:27:23
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

"""import pytest
import jax
import jax.numpy as jnp
from etils import epath
from flax import nnx
from mujoco import mjx
from algorithms.ppo.utils.gae import general_advantage_estimator
from algorithms.ppo.task.thor import ThorEnv, ThorAgent
from algorithms.ppo.utils.canonical_space import new_cs
from algorithms.ppo.src.agent import Agent
from algorithms.ppo.utils.loss import ppo_loss

model_path = "/home/lucas/Documentos/MLProjects/monadic_ppo"

@pytest.fixture(scope="session")
def enviroment():
    # Training code here (runs only once for the entire test session)
    env = ThorEnv.init(
        epath.Path(model_path + "/model/joystick_env.xml"),
        epath.Path(model_path + "/model"),
        epath.Path(model_path + "/model/meshes"),
        new_cs(jnp.array([-0.468, -0.468, 0]), jnp.array([0.468, 0.468, 0.664])),
        "thor",
        ["tool_position"],
        ctrl_dt=1.0/50,
        sim_dt= 1.0/1000
    )
    assert env is not None, "O ambiente não foi carregado corretamente!"
    
    return env

@pytest.fixture(scope="session")
def agent_par(enviroment):
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)
    agent = ThorAgent(enviroment, rngs)

    assert agent is not None, "O agente não foi carregado corretamente!"
    assert isinstance(agent, Agent), " Agente deve ser reconhecido/implementar os métodos da classe Agent"
    
    return agent, rngs

@pytest.fixture(scope="session")
def batch_and_mjx_data(enviroment):
    num_envs = 2

    #cria um mjx_data inicial e reseta um dado ambiente
    initial_mjx_data = mjx.make_data(enviroment.mjx_model)
    
    batched_mjx_data = jax.tree_util.tree_map(
        lambda x: jax.numpy.repeat(x[None], num_envs, axis=0), initial_mjx_data
    )
    return num_envs, batched_mjx_data

def test_loss_shape(agent_par, batch_and_mjx_data):
    agent, rngs = agent_par
    num_envs, batched_mjx_data = batch_and_mjx_data
    
    rollout_steps = 15

    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)

    vmap_agent_step = jax.vmap(agent.step, in_axes = (None, 0, 0, 0))
   
    advantages, returns = general_advantage_estimator(rewards, dones, values, gamma=0.01, lam=0.01)

    policy_obs_size = agent.policy().obs_size
    value_obs_size = agent.value().obs_size

    policy_obs = jax.random.uniform(rngs(), (rollout_steps +1, policy_obs_size))
    value_obs = jax.random.uniform(rngs(), (rollout_steps +1, value_obs_size))

    action = agent.policy().sample(policy_obs, rngs)

    x = ppo_loss(agent, policy_obs, value_obs, action, advantages, )

    assert advantages.shape == (rollout_steps, num_envs)
    assert returns.shape == (rollout_steps, num_envs)"""