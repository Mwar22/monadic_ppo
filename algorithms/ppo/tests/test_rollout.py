# -*- coding:utf-8 -*-
###
# File:  test_rollout.py
# Created Date: 31/05/2026 12:47:24
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 07/06/2026 06:08:35
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
from src.agent import Agent
from src.rollout import *

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


@pytest.fixture(scope="session")
def batch_and_mjx_data(enviroment):
    num_envs = 5

    # cria um mjx_data inicial e reseta um dado ambiente
    initial_mjx_data = mjx.make_data(enviroment.mjx_model)

    batched_mjx_data = jax.tree_util.tree_map(
        lambda x: jax.numpy.repeat(x[None], num_envs, axis=0), initial_mjx_data
    )
    return num_envs, batched_mjx_data


def test_rollout(enviroment, agent_par, batch_and_mjx_data):
    agent, rngs = agent_par
    num_envs, batched_mjx_data = batch_and_mjx_data

    buffer_length = 25

    dummy_target = jax.random.normal(rngs(), (num_envs, 3))
    buffer = new_buffer(
        num_envs,
        buffer_length,
        agent.policy.obs_size,
        agent.value.obs_size,
        agent.policy.action_size,
    )

    buffer, data, mjx_data = rollout(
        agent,
        enviroment,
        rngs,
        batched_mjx_data,
        dummy_target,
        buffer_length,
        0.2,
        buffer,
    )
    assert data.info["error"].shape == (buffer_length + 1, num_envs)
