# -*- coding:utf-8 -*-
###
# File:  step.py
# Created Date: 25/05/2026 01:33:38
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 25/05/2026 01:55:03
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
from typing import Protocol, Tuple, Any
from flax import struct
from state import EnviromentState

class StepData(struct.PyTreeNode):
    observation: jax.Array       # (num_envs, *obs_shape)
    action: jax.Array    # (num_envs, *action_shape)
    reward: jax.Array    # (num_envs,)
    logprob: jax.Array   # (num_envs,)

class StepFunction(Protocol):
    def __call__(self, state:EnviromentState)->Tuple[EnviromentState, StepData]:
        ...