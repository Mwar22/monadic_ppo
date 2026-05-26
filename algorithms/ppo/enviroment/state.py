# -*- coding:utf-8 -*-
###
# File:  state.py
# Created Date: 25/05/2026 01:46:39
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 25/05/2026 02:22:43
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
from typing import Protocol

class EnviromentState(Protocol):
    def failed(self)->jax.Array:
        ...
    def reached(self)->jax.Array:
        ...