# -*- coding:utf-8 -*-
###
# File:  cauchy.py
# Created Date: 25/05/2026 10:55:30
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 16/06/2026 04:18:27
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

"""
Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------
Redes que representam actor/critic
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import jit, custom_jvp

@custom_jvp
def cauchy_activation(x, lambda1= 0.01, lambda2= 0.01, d=1.0):
    #para estabilidade numerica
    eps = 1e-12
    
    x2_d2 = x**2 + d**2 + eps
    return (lambda1 * x + lambda2) / x2_d2

# gradiente customizado
@cauchy_activation.defjvp
def cauchy_activation_jvp(primals, tangents):
    x, lambda1, lambda2, d = primals
    x_dot, lambda1_dot, lambda2_dot, d_dot = tangents

    #para estabilidade numerica
    eps = 1e-12

    #cria uma versão "clipada" de d, diferenciavel, e evitando valores muito pequenos (explosão de gradiente)
    d_safe = jnp.sqrt(d**2 + 1e-6) # se |d|  << 1e-3, d_safe ≃ 1e-3. Caso contrário d_safe ≃ d
    
    #calcula valores que se repetem
    x2 = x**2
    d2 = d_safe**2
    x2_d2 = x2 + d2 + eps
    x2_d2_sq = x2_d2 ** 2
    inv = 1 / x2_d2_sq

    
    y = (lambda1 * x + lambda2) / x2_d2

    #derivadas parciais
    dy_dx = ((d2 - x2) * lambda1  - 2 * x * lambda2)*inv
    dy_dlambda1 = x / x2_d2
    dy_dlambda2 = 1 / x2_d2
    dy_dd = -2 * d_safe * (lambda1 *x + lambda2) * inv * (d / d_safe)

    tangent_out = dy_dx * x_dot + dy_dlambda1 * lambda1_dot + dy_dlambda2 * lambda2_dot + dy_dd * d_dot
    return y, tangent_out

class CauchyActivationModule(nnx.Module):
    def __init__(self, init_lambda1: float = 1.0, init_lambda2: float = 0.0, init_d: float = 1.0):
        self.lambda1 = nnx.Param(jnp.array(init_lambda1))
        self.lambda2 = nnx.Param(jnp.array(init_lambda2))
        self.d = nnx.Param(jnp.array(init_d))

    def __call__(self, x: jax.Array)->jax.Array:
        return cauchy_activation(x, self.lambda1, self.lambda2, self.d)





