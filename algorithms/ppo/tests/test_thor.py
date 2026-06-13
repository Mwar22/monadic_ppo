# -*- coding:utf-8 -*-
###
# File:  test_enviroment.py
# Created Date: 30/05/2026 03:33:57
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 06/06/2026 08:22:50
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
from usr.thor import ThorEnv, ThorAgent
from src.canonical_space import new_cs
from src.enviroment import MujocoEnv
from src.agent import Agent

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
        ctrl_dt=1.0 / 50,
        sim_dt=1.0 / 1000,
    )
    assert env is not None, "O ambiente não foi carregado corretamente!"

    return env


@pytest.fixture(scope="session")
def agent_par(enviroment):
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)
    agent = ThorAgent(enviroment, rngs)

    assert agent is not None, "O agente não foi carregado corretamente!"
    assert isinstance(agent, Agent), (
        " Agente deve ser reconhecido/implementar os métodos da classe Agent"
    )

    return agent, rngs


def test_instance(enviroment):
    assert isinstance(enviroment, MujocoEnv), (
        "Ambiente deve ser reconhecido/implementar os métodos da classe MujocoEnv"
    )


def test_thor_sizes(agent_par):
    agent, _ = agent_par

    assert agent.policy.obs_size == 15, (
        "Policy obs_size deve ser 12, pois são 6 juntas, e cada uma com um angulo e velocidade, além do alvo"
    )
    assert agent.policy.action_size == 6, (
        "Policy action_size deve ser 6, pois são 6 juntas "
    )
    assert agent.value.obs_size == 18, (
        "Value obs_size deve ser 15, uma vez que contem uma observação extra se comparado à politica: tool_position"
    )


@pytest.fixture(scope="session")
def batch_and_mjx_data(enviroment):
    num_envs = 2

    # cria um mjx_data inicial e reseta um dado ambiente
    initial_mjx_data = mjx.make_data(enviroment.mjx_model)

    batched_mjx_data = jax.tree_util.tree_map(
        lambda x: jax.numpy.repeat(x[None], num_envs, axis=0), initial_mjx_data
    )
    return num_envs, batched_mjx_data


def test_thor_reset(enviroment, agent_par, batch_and_mjx_data):
    agent, rngs = agent_par
    num_envs, batched_mjx_data = batch_and_mjx_data

    target = jax.random.uniform(rngs(), shape=(num_envs, 3))

    # cria mapas vetoriais para as funções step e reset, para funcinar com mjx_data em batch
    vmap_agent_reset = jax.vmap(agent.reset, in_axes=(None, 0, 0))
    reset_error, reset_mjx_data = vmap_agent_reset(enviroment, batched_mjx_data, target)

    # testa se a dimensão de batch vai aparecer
    assert reset_error.shape == (num_envs,)

    # testa se o mjx_data resultante também terá o shape de batch, como inicial
    assert reset_mjx_data.qpos.shape == batched_mjx_data.qpos.shape


def test_thor_step(enviroment, agent_par, batch_and_mjx_data):
    agent, rngs = agent_par
    num_envs, batched_mjx_data = batch_and_mjx_data

    # cria mapas vetoriais para as funções step e reset, para funcinar com mjx_data em batch
    vmap_agent_step = jax.vmap(agent.step, in_axes=(None, 0, 0, 0, 0, None))

    # cria uma observação qualquer  e obtem a ação relativa
    dummy_policy_obs = jax.random.normal(rngs(), (num_envs, agent.policy.obs_size))
    action = agent.policy.sample(dummy_policy_obs, rngs)

    assert action.shape == (num_envs, agent.policy.action_size)

    dummy_target = jax.random.uniform(rngs(), (num_envs, 3))
    last_error = jax.random.uniform(rngs(), (num_envs,))

    step_data, step_mjx_data = vmap_agent_step(
        enviroment, batched_mjx_data, dummy_target, action, last_error, 0
    )

    # testa se o shape das recompensas vai bater com o batch
    assert step_data.reward.shape == (num_envs,)

    # testa se o mjx_data resultante também terá o shape de batch, como inicial
    assert step_mjx_data.qpos.shape == batched_mjx_data.qpos.shape

