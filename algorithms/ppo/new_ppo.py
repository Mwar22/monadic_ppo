# -*- coding:utf-8 -*-
###
# File:  new_ppo.py
# Created Date: 31/05/2026 01:29:51
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 03/06/2026 06:17:16
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


model_path = "/home/lucas/Documentos/MLProjects/monadic_ppo"
K_epochs = 5
num_envs = 5
rollout_steps = 25


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

buffer, mjx_data = rollout(model, env, rngs, batched_mjx_data, dummy_target, rollout_steps, buffer)


@nnx.jit
def train_step(agent, optimizer, buffer):
    # Use has_aux=True so the optimizer doesn't try to differentiate the metrics
    def loss_fn(agent):
        return ppo_loss(
            agent, 
            buffer.policy_obs, 
            buffer.value_obs, 
            buffer.actions, 
            buffer.advantages, 
            buffer.returns, 
            buffer.logprobs
        )

    # Calculate gradients and return auxiliary metrics
    # grads will only contain updates for the 'agent' parameters
    (loss, (entropy, p_loss, v_loss, kl)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
    
    # Apply updates
    optimizer.update(agent, grads)
    
    return loss, (entropy, p_loss, v_loss, kl)

# 3. Execution Loop
for epoch in range(K_epochs):
    buffer = new_buffer(num_envs, rollout_steps, model.policy.obs_size, model.value.obs_size, model.policy.action_size)
    
    buffer, mjx_data = rollout(model, env, rngs, batched_mjx_data, dummy_target, rollout_steps, buffer)
    loss, metrics = train_step(model, optimizer, buffer)