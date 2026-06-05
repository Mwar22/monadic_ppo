# -*- coding:utf-8 -*-
###
# File:  new_ppo.py
# Created Date: 31/05/2026 01:29:51
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 05/06/2026 05:56:59
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


@nnx.jit(static_argnums=(4, 5)) 
def train_epochs(model, optimizer, buffer, rngs: nnx.Rngs, k_epochs: int, minibatch_size: int):
     #calcula as vantagens e os retornos, como um tensor 2d (rollout_steps, num_envs)
    advantages, returns = general_advantage_estimator(
        buffer.rewards, buffer.dones, buffer.values, gamma=0.99, lam=0.95
    )
    
    # exclui os valores de bootstrap com [:-1], e aplica um flatten, 
    # onde batch_size = rollout_steps * num_envs. Então os tensores vão de (rollout_steps, num_envs, ...) -> (batch_size, ...)
    policy_obs = buffer.policy_obs[:-1].reshape(-1, buffer.policy_obs.shape[-1])
    value_obs = buffer.value_obs[:-1].reshape(-1, buffer.value_obs.shape[-1])
    actions = buffer.actions[:-1].reshape(-1, buffer.actions.shape[-1])
    old_logprobs = buffer.logprobs[:-1].reshape(-1)
    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)

    batch_size = policy_obs.shape[0]
    num_minibatches = batch_size // minibatch_size
    
    # separa o modelo, optimizer e rngs em estrutura e estado (para funcionar no scan)
    graphdef, state = nnx.split((model, optimizer, rngs))
    
    def epoch_step(epoch_state, _):
        #recostroi o agente, o optimizer e o rngs para o estado da epoca
        ep_agent, ep_opt, ep_rngs = nnx.merge(graphdef, epoch_state)
        
        # Calling ep_rngs() generates a new key AND advances its internal state automatically
        permutation = jax.random.permutation(ep_rngs(), batch_size)
        
        #avançamos o estado novamente, pois utilizamos  rngs
        _, pre_mb_state = nnx.split((ep_agent, ep_opt, ep_rngs))
        
        def minibatch_step(mb_state, mb_idx):
            mb_agent, mb_opt, mb_rngs = nnx.merge(graphdef, mb_state)
            
            # THE FIX: Use dynamic_slice_in_dim instead of standard python slicing
            start_index = mb_idx * minibatch_size
            idx = jax.lax.dynamic_slice_in_dim(
                permutation, 
                start_index, 
                minibatch_size  # The size is explicitly static!
            )
            
            def loss_fn(agent):
                return ppo_loss(
                    agent, 
                    policy_obs[idx], 
                    value_obs[idx], 
                    actions[idx], 
                    advantages[idx], 
                    returns[idx], 
                    old_logprobs[idx]
                )

            (loss, aux_metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(mb_agent)
            mb_opt.update(mb_agent, grads)
            
            # Pack it all back up
            _, updated_mb_state = nnx.split((mb_agent, mb_opt, mb_rngs))
            return updated_mb_state, (loss, aux_metrics)

        #executa os minibatches
        post_mb_state, (mb_losses, mb_metrics) = jax.lax.scan(
            minibatch_step, 
            pre_mb_state, 
            jnp.arange(num_minibatches)
        )
        
        epoch_loss = jnp.mean(mb_losses)
        epoch_metrics = tuple(jnp.mean(m) for m in mb_metrics)
        
        return post_mb_state, (epoch_loss, epoch_metrics)

    # executa as epocas
    final_state, (losses, metrics) = jax.lax.scan(
        epoch_step, 
        state, 
        None,             # No external arrays needed!
        length=k_epochs   # JAX knows exactly how many times to loop
    )
    
    # faz o udate seguro dos pesos, otimizer e estado do rngs de volta
    nnx.update((model, optimizer, rngs), final_state)
    return losses, metrics


def exp_mean(x: jax.Array, alpha=0.9):
    """ Gera uma média ponderada considerando os valores finais em especial"""
    N =x.shape[0]
    gain = (1-alpha)/(1-alpha**N)
    weights = alpha**jnp.arange(N)[::-1]    #utiliza uma sequencia inversa
    return gain*jnp.inner(weights, x)

@nnx.jit(static_argnums=(6, 7, 8, 9))
def run_multiple_updates(
    model, optimizer, rngs, mjx_data, buffer, dummy_target, 
    rollout_steps: int, k_epochs: int, minibatch_size: int, num_updates: int
):
    # separa o grafo dos estados (puramente funcional)
    graphdef, state = nnx.split((model, optimizer, rngs))

    def update_step(carry, _):
        state_carry, mjx_carry, buffer_carry = carry
        
        #reconstroi os modelos para este passo especifico 
        step_model, step_opt, step_rngs = nnx.merge(graphdef, state_carry)
        
        next_buffer, steps_data,  next_mjx_data = rollout(
            step_model, env, step_rngs, mjx_carry, dummy_target, rollout_steps, buffer_carry
        )
        
        # otimiza
        losses, metrics = train_epochs(
            step_model, step_opt, next_buffer, step_rngs, k_epochs, minibatch_size
        )
        
        #erro esperado entre os ambientes
        expected_error = jnp.mean(steps_data.info["error"], axis=1) #erro médio entre ambientes
        accumulated_error = exp_mean(expected_error)

        #cada rollout só termina ou em sucesso ou falha. Neste caso, contamos quantas falhas e quantos sucessos tivemos
        success_count = jnp.sum(steps_data.info["success"], axis=0)
        failure_count = jnp.sum(steps_data.info["failure"], axis=0)
        success_rate = success_count/(success_count+failure_count + 1e-6)
        success_rate = jnp.mean(success_rate)
        
        #adiciona às metricas os dados dos passos (como o erro: shape = (rollout_steps+1, num_envs))
        metrics = (*metrics, accumulated_error, success_rate)
        
        # separa o modelo novamente para a forma funcional com o estado
        _, next_state = nnx.split((step_model, step_opt, step_rngs))
        
        return (next_state, next_mjx_data, next_buffer), (losses, metrics)

   
    final_carry, (all_losses, all_metrics) = jax.lax.scan(
        update_step, 
        (state, mjx_data, buffer), 
        None, 
        length=num_updates
    )

    final_state, final_mjx_data, final_buffer = final_carry
    
    #aplica o estado final nas instancias que estão fora do loop
    nnx.update((model, optimizer, rngs), final_state)
    
    return final_mjx_data, final_buffer, all_losses, all_metrics


######################################################################################################################

model_path = "/home/lucas/Documentos/MLProjects/monadic_ppo"
EPOCHS = 150
NUM_ENVS =8192
ROLLOUT_STEPS = 256
UPDATES = 50
MINIBATCH_SIZE = 4096


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
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

#cria um mjx_data inicial e reseta um dado ambiente
initial_mjx_data = mjx.make_data(env.mjx_model)

batched_mjx_data = jax.tree_util.tree_map(
    lambda x: jax.numpy.repeat(x[None], NUM_ENVS, axis=0), initial_mjx_data
)

dummy_target = jax.random.uniform(rngs(), (NUM_ENVS, 3), minval=-1, maxval=1)
buffer = new_buffer(NUM_ENVS, ROLLOUT_STEPS, model.policy.obs_size, model.value.obs_size, model.policy.action_size)


#treino
mjx_data, buffer, losses, metrics = run_multiple_updates(
    model, optimizer, rngs, batched_mjx_data, buffer, dummy_target, 
    ROLLOUT_STEPS, EPOCHS, MINIBATCH_SIZE, UPDATES
)

entropy_loss, policy_loss, value_loss, kl_div, error, success_rate = metrics

print(f"losses shape: {losses.shape}")
print(f"entropy  loss shape: {entropy_loss.shape}")
print(f"policy loss shape: {policy_loss.shape}")
print(f"value loss shape: {value_loss.shape}")
print(f"kl_div shape: {kl_div.shape}")
print(f"error shape: {error.shape}")
print(f"success rate shape:{success_rate.shape}")

#(updates, epochs)
loss = jnp.mean(losses, axis=1)
entropy = jnp.mean(entropy_loss, axis=1)
kl_div = jnp.mean(kl_div, axis=1)


import matplotlib.pyplot as plt


fig, axs = plt.subplots(3, 2, figsize=(10, 8), tight_layout=True)
axs[0][0].plot(loss)
axs[0][0].set_title("Training Loss")
axs[0][0].set_xlabel("Updates")
axs[0][0].set_ylabel("Loss")
axs[0][0].grid(True)

axs[0][1].plot(entropy)
axs[0][1].set_title("Entropy")
axs[0][1].set_xlabel("Updates")
axs[0][1].grid(True)


axs[1][0].plot(kl_div)
axs[1][0].set_title("KL Divergence")
axs[1][0].set_xlabel("Updates")
axs[1][0].grid(True)

axs[1][1].plot(error)
axs[1][1].set_title("exp mean rollout error")
axs[1][1].set_xlabel("Updates")
axs[1][1].grid(True)

axs[2][0].plot(success_rate)
axs[2][0].set_title("Success rate")
axs[2][0].set_xlabel("Updates")
axs[2][0].grid(True)


plt.savefig(f"training_plots.png")
print("\nTraining plots saved to training_plots.png")