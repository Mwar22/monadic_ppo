# -*- coding:utf-8 -*-
###
# File:  networks.py
# Created Date: 16/06/2026 10:41:42
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 28/07/2026 12:30:00
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
from .cauchy import CauchyActivationModule


class ReluLinear(nnx.Module):
    def __init__(self, in_dim, out_dim, rngs: nnx.Rngs):
        self.layer = nnx.Linear(
            in_dim,
            out_dim,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            rngs=rngs,
            dtype=jnp.float16,
            param_dtype=jnp.float32,
        )
        self.norm = nnx.LayerNorm(out_dim, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.layer(x)
        x = self.norm(x)
        return jax.nn.relu(x)


class CauchyLinear(nnx.Module):
    def __init__(self, in_dim, out_dim, rngs: nnx.Rngs):
        self.layer = nnx.Linear(
            in_dim,
            out_dim,
            kernel_init=nnx.initializers.orthogonal(1.0),
            rngs=rngs,
            dtype=jnp.float16,
            param_dtype=jnp.float32,
        )
        self.norm = nnx.LayerNorm(out_dim, rngs=rngs)
        self.activation = CauchyActivationModule()

    def __call__(self, x: jax.Array):
        x = self.layer(x)
        x = self.norm(x)
        return self.activation(x)
