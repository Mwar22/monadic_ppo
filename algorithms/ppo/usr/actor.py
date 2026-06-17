# -*- coding:utf-8 -*-
###
# File:  actor.py
# Created Date: 25/05/2026 09:38:11
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 16/06/2026 02:01:25
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
import jax.scipy.special as jsp
from flax import nnx
from typing import Callable, Tuple
from .networks import ReluLinear, CauchyLinear

MIN_ALPHA_BETA = 1.05
MAX_ALPHA_BETA = 100.0

hidden_init = nnx.initializers.orthogonal(jnp.sqrt(2))
actor_init = nnx.initializers.orthogonal(0.005)
activation = jax.nn.leaky_relu


class Actor(nnx.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        rngs: nnx.Rngs,
        layer: Callable[[int, int, nnx.Rngs], nnx.Module] = ReluLinear,
        ab_max=1e4,
        ab_min=1.001,
        eps=1e-4,
    ):
        self.ab_max = ab_max
        self.ab_min = ab_min
        self._obs_size = obs_size
        self._action_size = action_size
        self.eps = eps

        self.linear1 = layer(obs_size, 256, rngs)
        self.linear2 = layer(256, 256, rngs)
        self.linear3 = layer(256, 64, rngs)

        self.alinear1 = layer(64, 64, rngs)
        self.alinear2 = layer(64, 64, rngs)

        self.blinear1 = layer(64, 64, rngs)
        self.blinear2 = layer(64, 64, rngs)

        self.alpha_layer = nnx.Linear(
            64,
            action_size,
            kernel_init=actor_init,
            rngs=rngs,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
        self.beta_layer = nnx.Linear(
            64,
            action_size,
            kernel_init=actor_init,
            rngs=rngs,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x1 = self.linear1(obs)
        x2 = self.linear2(x1)
        x3 = self.linear3(x2)

        # alpha section
        a1 = self.alinear1(x3)
        a2 = self.alinear2(a1)
        raw_alpha = self.alpha_layer(a2)

        # beta section
        b1 = self.blinear1(x3)
        b2 = self.blinear2(b1)
        raw_beta = self.beta_layer(b2)

        alpha = jnp.clip(
            jax.nn.softplus(raw_alpha) + self.ab_min, min=self.ab_min, max=self.ab_max
        )
        beta = jnp.clip(
            jax.nn.softplus(raw_beta) + self.ab_min, min=self.ab_min, max=self.ab_max
        )
        return alpha, beta

    @property
    def obs_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._action_size

    def sample(self, obs: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        alpha, beta = self(obs)

        # retira as amostras da distribuição beta
        actions = jax.random.beta(rngs(), alpha, beta)

        # Retorna a ação com clip de segurança para mandar para o ambiente
        safe_actions = jnp.clip(actions, self.eps, 1.0 - self.eps)

        return safe_actions

    def evaluate_actions(
        self, obs: jax.Array, actions: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        alpha, beta = self(obs)
        actions = jnp.clip(actions, self.eps, 1.0 - self.eps)

        logprobs_per_dim = jax.scipy.stats.beta.logpdf(actions, alpha, beta)
        entropy_per_dim = (
            jsp.betaln(alpha, beta)
            - (alpha - 1.0) * jsp.digamma(alpha)
            - (beta - 1.0) * jsp.digamma(beta)
            + (alpha + beta - 2.0) * jsp.digamma(alpha + beta)
        )
        logprobs = jnp.sum(logprobs_per_dim, axis=-1)
        entropy = jnp.sum(entropy_per_dim, axis=-1)

        return logprobs, entropy
