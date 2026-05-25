# -*- coding:utf-8 -*-
###
# File:  protocol.py
# Created Date: 25/05/2026 09:53:25
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 25/05/2026 11:06:03
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
from typing import Tuple, Protocol, runtime_checkable

@runtime_checkable
class Policy(Protocol):
    def __call__(self, obs: jax.Array)->Tuple[jax.Array, jax.Array]:
        ...

class Value(Protocol):
    def __call__(self, obs: jax.Array)->jax.Array:
        ...