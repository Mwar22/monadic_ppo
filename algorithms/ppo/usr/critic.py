# -*- coding:utf-8 -*-
###
# File:  critic.py
# Created Date: 25/05/2026 10:47:38
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 25/05/2026 10:52:34
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
from flax import nnx
from typing import Callable
from .networks import ReluLinear, CauchyLinear

MIN_ALPHA_BETA = 1.1
MAX_ALPHA_BETA = 100.0

critic_init = nnx.initializers.orthogonal(0.01)


class Critic(nnx.Module):
    def __init__(
        self,
        obs_size: int,
        rngs: nnx.Rngs,
        layer: Callable[[int, int, nnx.Rngs], nnx.Module] = ReluLinear,
    ):
        self.obs_size = obs_size

        self.linear1 = layer(obs_size, 256, rngs)
        self.linear2 = layer(256, 256, rngs)
        self.linear3 = layer(256, 64, rngs)

        self.out_layer = nnx.Linear(
            64,
            1,
            kernel_init=critic_init,
            rngs=rngs,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        x1 = self.linear1(obs)
        x2 = self.linear2(x1)
        x3 = self.linear3(x2)
        out = self.out_layer(x3)
        return out.squeeze(-1)
