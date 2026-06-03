# -*- coding:utf-8 -*-
###
# File:  main.py
# Created Date: 26/05/2026 09:58:32
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 03/06/2026 05:42:26
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
Arquivo com o código principal de treinamento
"""

import os
import sys

sys.stdout.flush()

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")

xla_flags += (
    " --xla_gpu_triton_gemm_any=True --xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
)
os.environ["XLA_FLAGS"] = xla_flags

# alocação dinamica
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# evita do jax prealocar a gpu inteira
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# limite, pois tbm precisamos de um pouco de vram para o sistema
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60"

import jax
from jax import config

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", False)
print(f"jax_enable_x64: {jax.config.read('jax_enable_x64')}")


import jax.numpy as jnp
from jax import config
from etils import epath
from flax import nnx
from mujoco import mjx
from reach_target.actor import Actor
from algorithms.ppo.src.rollout import new_buffer, rollout_step

from algorithms.ppo.reach_target.thor import ThorEnv
from algorithms.ppo.src.canonical_space import new_cs

world_space = new_cs(jnp.array([-0.468, -0.468, 0]), jnp.array([0.468, 0.468, 0.664]))
env =  ThorEnv.init(
    epath.Path("../model/joystick_env.xml"),
    epath.Path("../model"),
    epath.Path("../model/meshes"),
    world_space,
    name="thor",
    sensor_names=["tool_position"]
)
num_envs = 15
obs_size = env.observation_size
buff = new_buffer(num_envs, 10, obs_size, 6)


rng = jax.random.PRNGKey(777)
rng1, rng2 = jax.random.split(rng)


batched_rng = jax.random.split(rng1, num_envs)

mjx_data = mjx.make_data(env.mjx_model)
batched_mjx_data = jax.tree_util.tree_map(
    lambda x: jax.numpy.repeat(x[None], num_envs, axis=0), mjx_data
)

def sample_single_env(key):
    # Shape is just (3,) for one environment
    return jax.random.truncated_normal(key, lower=-1.0, upper=1.0, shape=(3,))

batched_sample = jax.vmap(sample_single_env)

target_rng = jax.random.split(rng2, num_envs)
targets = batched_sample(target_rng)

done = jnp.zeros((num_envs,), dtype=jnp.bool)
index = jnp.zeros((num_envs,), dtype=jnp.int32)

rngs = nnx.Rngs(0)
model = Actor(env.observation_size, env.action_size, rngs)

graph, state = nnx.split(model)
   
jit_rollout_step = jax.jit(rollout_step, static_argnums=(7,))
rng, mjx_data, new_buffer, done, index = jit_rollout_step(batched_rng, batched_mjx_data, env, targets, done, index, buff, graph, state)

#print(f"rng:{rng}")
print(f"logprobs:{new_buffer.logprobs} | logprobs rollout[0]:{new_buffer.logprobs[0, :]}")
print(f"done shape:{done.shape}")
print(f"index shape:{index.shape}")
