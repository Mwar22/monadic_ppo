# -*- coding:utf-8 -*-
###
# File:  new_ppo.py
# Created Date: 31/05/2026 01:29:51
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 07/06/2026 05:21:07
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
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# evita do jax prealocar a gpu inteira
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# limite, pois tbm precisamos de um pouco de vram para o sistema
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60"

import jax
from jax import config

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", False)
print(f"jax_enable_x64: {jax.config.read('jax_enable_x64')}")

import numpy as np
import matplotlib.pyplot as plt

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
def train_epochs(
    model, optimizer, buffer, rngs: nnx.Rngs, k_epochs: int, minibatch_size: int
):
    # calcula as vantagens e os retornos, como um tensor 2d (buffer_length, num_envs)
    advantages, returns = general_advantage_estimator(
        buffer.rewards, buffer.dones, buffer.values, gamma=0.99, lam=0.95
    )

    # exclui os valores de bootstrap com [:-1], e aplica um flatten,
    # onde batch_size = buffer_length * num_envs. Então os tensores vão de (buffer_length, num_envs, ...) -> (batch_size, ...)
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
        # recostroi o agente, o optimizer e o rngs para o estado da epoca
        ep_agent, ep_opt, ep_rngs = nnx.merge(graphdef, epoch_state)

        # Calling ep_rngs() generates a new key AND advances its internal state automatically
        permutation = jax.random.permutation(ep_rngs(), batch_size)

        # avançamos o estado novamente, pois utilizamos  rngs
        _, pre_mb_state = nnx.split((ep_agent, ep_opt, ep_rngs))

        def minibatch_step(mb_state, mb_idx):
            mb_agent, mb_opt, mb_rngs = nnx.merge(graphdef, mb_state)

            start_index = mb_idx * minibatch_size
            idx = jax.lax.dynamic_slice_in_dim(
                permutation,
                start_index,
                minibatch_size,  # The size is explicitly static!
            )

            def loss_fn(agent):
                return ppo_loss(
                    agent,
                    policy_obs[idx],
                    value_obs[idx],
                    actions[idx],
                    advantages[idx],
                    returns[idx],
                    old_logprobs[idx],
                )

            (loss, aux_metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
                mb_agent
            )
            mb_opt.update(mb_agent, grads)

            # Pack it all back up
            _, updated_mb_state = nnx.split((mb_agent, mb_opt, mb_rngs))
            return updated_mb_state, (loss, aux_metrics)

        # executa os minibatches
        post_mb_state, (mb_losses, mb_metrics) = jax.lax.scan(
            minibatch_step, pre_mb_state, jnp.arange(num_minibatches)
        )

        epoch_loss = jnp.mean(mb_losses)
        epoch_metrics = tuple(jnp.mean(m) for m in mb_metrics)

        return post_mb_state, (epoch_loss, epoch_metrics)

    # executa as epocas
    final_state, (losses, metrics) = jax.lax.scan(
        epoch_step,
        state,
        None,  # No external arrays needed!
        length=k_epochs,  # JAX knows exactly how many times to loop
    )

    # faz o udate seguro dos pesos, otimizer e estado do rngs de volta
    nnx.update((model, optimizer, rngs), final_state)
    return losses, metrics


def exp_mean(x: jax.Array, alpha=0.9):
    """Gera uma média ponderada considerando os valores finais em especial"""
    N = x.shape[0]
    gain = (1 - alpha) / (1 - alpha**N)
    weights = alpha ** jnp.arange(N)[::-1]  # utiliza uma sequencia inversa
    return gain * jnp.inner(weights, x)


@nnx.jit(static_argnums=(6, 7, 8, 9))
def run_multiple_updates(
    model,
    optimizer,
    rngs,
    mjx_data,
    buffer,
    target,
    buffer_length: int,
    k_epochs: int,
    minibatch_size: int,
    num_updates: int,
    start_error_tol: float = 0.2,
):
    # separa o grafo dos estados (puramente funcional)
    graphdef, state = nnx.split((model, optimizer, rngs))

    def update_step(carry, idx):
        state_carry, mjx_carry, buffer_carry, error_tol = carry

        # reconstroi os modelos para este passo especifico
        step_model, step_opt, step_rngs = nnx.merge(graphdef, state_carry)

        # progress = idx/(num_updates-1)
        # progress = jnp.clip(success_rate, 0.5, 1.0)

        next_buffer, steps_data, next_mjx_data = rollout(
            step_model,
            env,
            step_rngs,
            mjx_carry,
            target,
            buffer_length,
            error_tol,
            buffer_carry,
        )

        # otimiza
        losses, metrics = train_epochs(
            step_model, step_opt, next_buffer, step_rngs, k_epochs, minibatch_size
        )

        # erro esperado entre os ambientes
        # erro.shape (buffer_length+1, num_envs)
        expected_error = jnp.mean(
            steps_data.info["error"], axis=1
        )  # erro médio entre ambientes

        accumulated_error = exp_mean(expected_error)

        # cada rollout só termina ou em sucesso ou falha. Neste caso, contamos quantas falhas e quantos sucessos tivemos
        # (buffer_length+1, num_envs)
        success_count = jnp.sum(steps_data.info["success"], axis=0)
        failure_count = jnp.sum(steps_data.info["failure"], axis=0)

        # média entre ambientes
        success_count = jnp.mean(success_count)
        failure_count = jnp.mean(failure_count)

        expected_p_success = success_count / (success_count + failure_count + 1e-6)

        error_tol = jnp.where(
            expected_p_success > 0.8,
            error_tol * 0.8,  # torna 20% mais dificil
            error_tol,
        )

        # adiciona às metricas os dados dos passos (como o erro: shape = (buffer_length+1, num_envs))
        metrics = (
            *metrics,
            accumulated_error,
            success_count,
            failure_count,
        )

        # separa o modelo novamente para a forma funcional com o estado
        _, next_state = nnx.split((step_model, step_opt, step_rngs))

        return (next_state, next_mjx_data, next_buffer, error_tol), (
            losses,
            metrics,
        )

    final_carry, (all_losses, all_metrics) = jax.lax.scan(
        update_step, (state, mjx_data, buffer, start_error_tol), jnp.arange(num_updates)
    )

    final_state, final_mjx_data, final_buffer, final_error_tol = final_carry

    # aplica o estado final nas instancias que estão fora do loop
    nnx.update((model, optimizer, rngs), final_state)

    return final_mjx_data, final_buffer, all_losses, all_metrics


######################################################################################################################

model_path = "/home/lucas/Documentos/MLProjects/monadic_ppo"
EPOCHS = 8
NUM_ENVS = 10240
BUFFER_LENGTH = 512
UPDATES = 25
MINIBATCH_SIZE = 32768


env = ThorEnv.init(
    epath.Path(model_path + "/model/joystick_env.xml"),
    epath.Path(model_path + "/model"),
    epath.Path(model_path + "/model/meshes"),
    new_cs(jnp.array([-0.468, -0.468, 0]), jnp.array([0.468, 0.468, 0.664])),
    "thor",
    ["tool_position"],
    ctrl_dt=1.0 / 50,
    sim_dt=1.0 / 1000,
)

key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(key)
model = ThorAgent(env, rngs)
optimizer = nnx.Optimizer(model, optax.adam(5e-4), wrt=nnx.Param)

# cria um mjx_data inicial e reseta um dado ambiente
initial_mjx_data = mjx.make_data(env.mjx_model)

batched_mjx_data = jax.tree_util.tree_map(
    lambda x: jax.numpy.repeat(x[None], NUM_ENVS, axis=0), initial_mjx_data
)

dummy_target = jax.random.uniform(rngs(), (NUM_ENVS, 3), minval=-1, maxval=1)

buffer = new_buffer(
    NUM_ENVS,
    BUFFER_LENGTH,
    model.policy.obs_size,
    model.value.obs_size,
    model.policy.action_size,
)


# treino
mjx_data, buffer, losses, metrics = run_multiple_updates(
    model,
    optimizer,
    rngs,
    batched_mjx_data,
    buffer,
    dummy_target,
    BUFFER_LENGTH,
    EPOCHS,
    MINIBATCH_SIZE,
    UPDATES,
)

(entropy_loss, policy_loss, value_loss, kl_div, error, success, failure) = metrics

# (updates, epoch)
print(f"losses shape: {losses.shape}")
print(f"entropy  loss shape: {entropy_loss.shape}")
print(f"policy loss shape: {policy_loss.shape}")
print(f"value loss shape: {value_loss.shape}")
print(f"kl_div shape: {kl_div.shape}")
print(f"error shape: {error.shape}")
print(f"success rate shape:{success.shape}")
print(buffer.dones)

# 1. Convert JAX arrays to NumPy in one clean line
losses_np, entropy_np, kl_np = (
    np.asarray(losses),
    np.asarray(entropy_loss),
    np.asarray(kl_div),
)
error_np, success_np, failure_np = (
    np.asarray(error),
    np.asarray(success),
    np.asarray(failure),
)

error_np = np.asarray(error)

# We only average the 2D arrays (Loss, Entropy, KL)
plot_configs = [
    (losses_np, "Training Loss", "blue"),
    (entropy_np, "Entropy", "orange"),
    (kl_np, "KL Divergence", "red"),
]

fig = plt.figure(figsize=(12, 9), tight_layout=True)

# Plot the 2D data (Averaged across Epochs)
for i, (data, title, color) in enumerate(plot_configs, start=1):
    ax = fig.add_subplot(3, 2, i)

    updates = np.arange(data.shape[0])

    # Calculate Mean and Standard Deviation across the Epochs axis (axis=1)
    mean_val = np.mean(data, axis=1)
    std_val = np.std(data, axis=1)

    # Plot the solid mean line
    ax.plot(updates, mean_val, color=color, linewidth=2)

    # Plot the shaded region representing the variance between epochs
    ax.fill_between(
        updates, mean_val - std_val, mean_val + std_val, color=color, alpha=0.2
    )

    ax.set(xlabel="Updates", title=title)
    ax.grid(True, alpha=0.5)

# Plot the standard 1D Rollout metrics in the remaining slots
ax4 = fig.add_subplot(3, 2, 4)
ax4.plot(error_np, color="purple")
ax4.set(xlabel="Updates", title="Exp Mean Rollout Error")
ax4.grid(True, alpha=0.5)

ax5 = fig.add_subplot(3, 2, 5)
ax5.plot(success_np, color="green", label="Success")
ax5.plot(failure_np, color="red", label="Failure")
ax5.set(xlabel="Updates", title="Success and failure avg count")
ax5.grid(True, alpha=0.5)
ax5.legend()

ax6 = fig.add_subplot(3, 2, 6)
ax6.plot(error_np)
ax6.set(xlabel="Updates", title="Last error")
ax6.grid(True, alpha=0.5)

plt.savefig("training_plots.png")
print("\nTraining plots saved to training_plots.png")
