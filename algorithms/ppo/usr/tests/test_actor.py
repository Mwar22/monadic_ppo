# -*- coding:utf-8 -*-
###
# File:  test_actor.py
# Created Date: 30/05/2026 09:59:47
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 04/06/2026 08:54:03
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
from flax import nnx
from algorithms.ppo.usr.actor import Actor 

obs_size = 10
action_size = 2
batch_size = 32

@pytest.fixture(scope="session")
def shared_parameters():
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)
    model = Actor(obs_size=obs_size, action_size=action_size, rngs=rngs)
    obs = jax.random.normal(rngs(), (batch_size, obs_size))

    return model, obs, rngs

def test_output_shape(shared_parameters):
    model, obs, _ = shared_parameters
    alpha, beta = model(obs)
    
    assert alpha.shape == (batch_size, action_size)
    assert beta.shape == (batch_size, action_size)

def test_sample(shared_parameters):
    model, obs, rngs = shared_parameters

    action1 = model.sample(obs, rngs)
    action2 = model.sample(obs, rngs)

    #verifica se o formato da ação tomada é condizente com o definido na classe e contem dimensão de batch
    assert action1.shape == (batch_size, action_size)

    #verifica se a amostragem está randômica (se existir ao menos 1 das ações diferentes para duas amostras sobre a mesma observação)
    assert jnp.any(action1 != action2)

def test_evaluate_actions(shared_parameters):
    model, obs, rngs = shared_parameters

    action = model.sample(obs, rngs)
    logprob, entropy = model.evaluate_actions(obs, action)

    assert logprob.shape == (batch_size, )
    assert entropy.shape == (batch_size, )