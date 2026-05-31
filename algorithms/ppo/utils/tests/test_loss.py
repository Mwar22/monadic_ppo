# -*- coding:utf-8 -*-
###
# File:  test_loss.py
# Created Date: 30/05/2026 04:12:14
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 30/05/2026 07:34:43
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
from flax import nnx
from algorithms.ppo.utils.gae import general_advantage_estimator
from algorithms.ppo.utils.loss import ppo_loss

rollout_steps = 10
num_envs = 5

def test_loss_shape():
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)

    rewards = jax.random.normal(rngs(), (rollout_steps + 1, num_envs))
    dones = jax.random.bernoulli(rngs(), p=0.1, shape=(rollout_steps + 1, num_envs)).astype(int)
    values = jax.random.normal(rngs(), (rollout_steps+1, num_envs))
   
    advantages, returns = general_advantage_estimator(rewards, dones, values, gamma=0.01, lam=0.01)

    x = ppo_loss()

    assert advantages.shape == (rollout_steps, num_envs)
    assert returns.shape == (rollout_steps, num_envs)"""