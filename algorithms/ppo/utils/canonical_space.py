# -*- coding:utf-8 -*-
###
# File:  canonical_space.py
# Created Date: 25/05/2026 02:27:14
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 26/05/2026 05:40:25
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
from flax import struct

class CanonicalSpace(struct.PyTreeNode):
    """ Permite transformar o espaço de observação dentro do intervalo [-1, 1]"""
    max_extent: jax.Array
    center: jax.Array


def new_cs(min: jax.Array, max: jax.Array)->CanonicalSpace:
    max_extent = jnp.max(max - min)
    center = (max + min)/2.0
    return CanonicalSpace(max_extent, center)

def transform_to_cs(space: CanonicalSpace, position: jax.Array)->jax.Array:
    return 2*(position - space.center)/space.max_extent

def transform_vel_to_cs(space: CanonicalSpace, velocity: jax.Array)->jax.Array:
    return 2*velocity/space.max_extent