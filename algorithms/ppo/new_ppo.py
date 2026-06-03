# -*- coding:utf-8 -*-
###
# File:  new_ppo.py
# Created Date: 31/05/2026 01:29:51
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 03/06/2026 07:42:34
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

from os import wait
import jax
import jax.numpy as jnp
import optax
from etils import epath
from flax import nnx
from mujoco import mjx
from usr.thor import ThorEnv, ThorAgent
from src.canonical_space import new_cs
from src.loss import ppo_loss
from src.rollout import new_buffer, rollout
from src.gae import general_advantage_estimator


model_path = "/home/lucas/Documentos/MLProjects/monadic_ppo"
K_epochs = 5
num_envs = 5
rollout_steps = 25
total_updates = 100


env = ThorEnv.init(
    epath.Path(model_path + "/model/joystick_env.xml"),
    epath.Path(model_path + "/model"),
    epath.Path(model_path + "/model/meshes"),
    new_cs(jnp.array([-0.468, -0.468, 0]), jnp.array([0.468, 0.468, 0.664])),
    "thor",
    ["tool_position"],
    ctrl_dt=1.0/50,
    sim_dt= 1.0/1000
)
key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(key)
model = ThorAgent(env, rngs)



optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


#cria um mjx_data inicial e reseta um dado ambiente
initial_mjx_data = mjx.make_data(env.mjx_model)

batched_mjx_data = jax.tree_util.tree_map(
    lambda x: jax.numpy.repeat(x[None], num_envs, axis=0), initial_mjx_data
)


dummy_target = jax.random.normal(rngs(), (num_envs, 3))
buffer = new_buffer(num_envs, rollout_steps, model.policy.obs_size, model.value.obs_size, model.policy.action_size)

@nnx.jit(static_argnames=['k_epochs'])
def train_epochs(model, optimizer, buffer, k_epochs: int):

    advantages, returns = general_advantage_estimator(buffer.rewards, buffer.dones, buffer.values, gamma=0.01, lam=0.01)
    
    # This is the function that runs for a single epoch
    def epoch_step(carry, _):
        agent, opt = carry
        
        # Use your exact ppo_loss function here
        def loss_fn(agent):
            return ppo_loss(
                agent, 
                buffer.policy_obs, 
                buffer.value_obs, 
                buffer.actions, 
                advantages, 
                returns, 
                buffer.logprobs # Ensure this is from the OLD policy
            )

        # Calculate gradients and metrics
        (loss, aux_metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
        
        # Apply updates to the agent
        opt.update(agent, grads)
        
        # Return the updated state, and save the metrics for this epoch
        return (agent, opt), (loss, aux_metrics)

    # jax.lax.scan can act like a 'for' loop if you pass None for the array
    # and specify the length (k_epochs)
    (model, optimizer), (losses, metrics) = jax.lax.scan(
        epoch_step, 
        (model, optimizer), 
        None, 
        length=k_epochs
    )
    
    # 'losses' and 'metrics' will now be arrays containing the values for every epoch
    return losses, metrics

for update in range(total_updates):
    # 1. ROLLOUT: Collect experience with the CURRENT policy ONCE
    # This creates a fresh buffer and fills it.
    buffer = new_buffer(num_envs, rollout_steps, model.policy.obs_size, model.value.obs_size, model.policy.action_size)
    buffer, mjx_data = rollout(model, env, rngs, batched_mjx_data, dummy_target, rollout_steps, buffer)
    
    # Note: If your rollout doesn't calculate advantages/returns, 
    # you must calculate them right here before passing to train_epochs.


    # 2. OPTIMIZE: Train for K epochs on that SAME buffer
    # The JIT function handles the looping internally via scan
    losses, metrics = train_epochs(model, optimizer, buffer, k_epochs=K_epochs)
    
    # 3. LOGGING: Take the mean of the metrics across the K epochs for logging
    mean_loss = jnp.mean(losses)
    entropy, p_loss, v_loss, kl = [jnp.mean(m) for m in metrics]
    
    if update % 10 == 0:
        print(f"Update {update} | Loss: {mean_loss:.4f} | KL: {kl:.4f}")