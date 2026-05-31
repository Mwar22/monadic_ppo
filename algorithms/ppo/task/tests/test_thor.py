# -*- coding:utf-8 -*-
###
# File:  test_enviroment.py
# Created Date: 30/05/2026 03:33:57
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 30/05/2026 10:52:51
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
import pytest
import jax
import jax.numpy as jnp
from etils import epath
from flax import nnx
from mujoco import mjx
from algorithms.ppo.utils.gae import general_advantage_estimator
from algorithms.ppo.task.thor import ThorEnv, ThorAgent
from algorithms.ppo.utils.canonical_space import new_cs
from algorithms.ppo.src.enviroment import MujocoEnv
from algorithms.ppo.src.agent import Agent

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


def test_instance(enviroment):
    assert isinstance(enviroment, MujocoEnv), "Ambiente deve ser reconhecido/implementar os métodos da classe MujocoEnv"

def test_thor_sizes(agent_par):
    agent, _ = agent_par

    assert agent.policy().obs_size == 12, "Policy obs_size deve ser 12, pois são 6 juntas, e cada uma com um angulo e velocidade"
    assert agent.policy().action_size == 6, "Policy action_size deve ser 6, pois são 6 juntas "
    assert agent.value().obs_size== 15, "Value obs_size deve ser 15, uma vez que contem uma observação extra se comparado à politica: tool_position"



import jax.numpy as jnp
from jax.tree_util import tree_map

def are_data_equal(data1, data2, atol=1e-6):
    # tree_map applies the comparison to every leaf of the mjx.Data struct
    # (qpos, qvel, ctrl, etc.)
    comparisons = tree_map(lambda x, y: jnp.isclose(x, y, atol=atol), data1, data2)
    
    # Check if all comparisons across the entire tree are True
    return jnp.all(jax.tree_util.tree_leaves(comparisons))

def test_thor_step_and_reset(enviroment, agent_par):
    agent, rngs = agent_par

    num_envs = 2

    #cria um mjx_data inicial e reseta um dado ambiente
    initial_mjx_data = mjx.make_data(enviroment.mjx_model)
    batched_mjx_data = jax.tree_util.tree_map(
        lambda x: jax.numpy.repeat(x[None], num_envs, axis=0), initial_mjx_data
    )

    reset_mjx_data, reset_data = agent.reset(enviroment, batched_mjx_data, rngs)

    #cria uma observação qualquer (batch de 1), e obtem a ação relativa
    dummy_policy_obs = jax.random.normal(rngs(), (1, agent.policy().obs_size))
    action = agent.policy().sample(dummy_policy_obs, rngs)

    assert action.shape == (1, agent.policy().action_size)

    dummy_target = jax.random.normal(rngs(), (3,))
    step_mjx_data, step_data = agent.step(enviroment, reset_mjx_data, action, dummy_target)

   # assert are_data_equal(reset_mjx_data, reset_mjx_data), "mjx_data após tomada de ação deve ser diferente"