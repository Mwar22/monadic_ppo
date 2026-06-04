# -*- coding:utf-8 -*-
###
# File:  test_actor.py
# Created Date: 30/05/2026 09:59:47
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 04/06/2026 08:54:05
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
from algorithms.ppo.usr.critic import Critic

obs_size = 10
batch_size = 32

def test_output_shape():
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)
    model = Critic(obs_size=obs_size, rngs=rngs)
    obs = jax.random.normal(key, (batch_size, obs_size))
    
    value = model(obs)
    
    # Assert
    assert value.shape == (batch_size, )
