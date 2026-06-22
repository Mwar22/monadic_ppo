# -*- coding:utf-8 -*-
###
# File:  new_ppo.py
# Created Date: 31/05/2026 01:29:51
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 17/06/2026 10:12:17
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

xla_flags = os.environ.get("XLA_FLAGS", "")

# força o xla a testar varios algoritmos de multiplicaçao/convolução e escolhe o melhor para
# a gpu
xla_flags += " --xla_gpu_autotune_level=4"

# faz o xla usar Triton GEMM, o que melhora o desempenho em até 30% em algumas gpus
xla_flags += " --xla_gpu_triton_gemm_any=True"

# evita do jax prealocar a gpu inteira
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
from jax import config

config.update("jax_enable_x64", False)
print(f"jax_enable_x64: {jax.config.read('jax_enable_x64')}")

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import optax
from etils import epath
from flax import nnx, struct
from mujoco import mjx
from usr.thor import ThorEnv, ThorAgent
from src.canonical_space import new_cs
from src.loss import LossMetrics, ppo_loss
from src.rollout import RolloutBuffer, new_buffer, rollout
from src.gae import general_advantage_estimator
from src.agent import StepData, Agent
from typing import Self
import orbax.checkpoint as ocp


@nnx.jit(static_argnums=(4, 5))
def train_epochs(
    model,
    optimizer,
    buffer,
    rngs: nnx.Rngs,
    k_epochs: int,
    minibatch_size: int,
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

            (loss, loss_metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
                mb_agent
            )

            mb_opt.update(mb_agent, grads)

            # estado após update
            _, updated_mb_state = nnx.split((mb_agent, mb_opt, mb_rngs))

            mb_state = jax.lax.cond(
                loss_metrics.is_safe, lambda: updated_mb_state, lambda: mb_state
            )
            return mb_state, (loss, loss_metrics)

        # executa os minibatches
        epoch_state, (epoch_losses, epoch_metrics) = jax.lax.scan(
            minibatch_step, pre_mb_state, jnp.arange(num_minibatches)
        )

        epoch_avg_loss = jnp.mean(epoch_losses)
        epoch_avg_metrics = jax.tree.map(jnp.mean, epoch_metrics)

        return epoch_state, (epoch_avg_loss, epoch_avg_metrics)

    # executa as epocas
    final_state, (losses, metrics) = jax.lax.scan(
        epoch_step,
        state,
        None,
        length=k_epochs,
    )

    # faz o udate seguro dos pesos, otimizer e estado do rngs de volta
    nnx.update((model, optimizer, rngs), final_state)
    return losses, metrics


class TrainingMetrics(struct.PyTreeNode):
    loss_metrics: LossMetrics
    acumulated_error: jax.Array
    avg_success: jax.Array
    avg_failure: jax.Array
    avg_done: jax.Array

    @classmethod
    def init(cls, loss_metrics: LossMetrics, steps_data: StepData) -> Self:

        def exp_mean(x: jax.Array, alpha=0.95):
            """Gera uma média ponderada considerando os valores finais em especial"""
            N = x.shape[0]
            gain = (1 - alpha) / (1 - alpha**N)
            weights = alpha ** jnp.arange(N)[::-1]  # utiliza uma sequencia inversa
            return gain * jnp.inner(weights, x)

        # steps_data.error.shape = (buffer_length + 1, num_envs)
        acumulated_error = exp_mean(jnp.mean(steps_data.info["error"], axis=1))

        avg_success = jnp.mean(jnp.sum(steps_data.info["success"], axis=0))
        avg_failure = jnp.mean(jnp.sum(steps_data.info["failure"], axis=0))
        avg_done = jnp.mean(jnp.sum(steps_data.info["done"], axis=0))

        return cls(loss_metrics, acumulated_error, avg_success, avg_failure, avg_done)

    @property
    def success_rate(self):
        return self.avg_success / (self.avg_done + 1e-6)


@nnx.jit(static_argnums=(5, 6, 7, 8, 9))
def run_multiple_updates(
    model: Agent,
    optimizer,
    rngs: nnx.Rngs,
    mjx_data: mjx.Data,
    buffer: RolloutBuffer,
    buffer_length: int,
    k_epochs: int,
    minibatch_size: int,
    num_updates: int,
    num_envs: int,
    start_error_tol: float = 0.05,
):
    # separa o grafo dos estados (puramente funcional)
    graphdef, state = nnx.split((model, optimizer, rngs))

    def update_step(carry, _):
        state_carry, mjx_carry, buffer_carry, error_tol = carry

        # reconstroi os modelos para este passo especifico
        step_model, step_opt, step_rngs = nnx.merge(graphdef, state_carry)

        # cria um novo alvo
        target = jax.random.uniform(step_rngs(), (num_envs, 3), minval=-1, maxval=1)

        next_buffer, steps_data, next_mjx_data = rollout(
            step_model,
            env,
            step_rngs,
            mjx_carry,
            target,
            buffer_length,
            buffer_carry,
            error_tol,
        )

        # otimiza
        losses, loss_metrics = train_epochs(
            step_model, step_opt, next_buffer, step_rngs, k_epochs, minibatch_size
        )

        metrics = TrainingMetrics.init(loss_metrics, steps_data)

        # a partir de uma taxa de sucesso de 85%, aumenta a dificuldade em 2%
        error_tol = jnp.where(metrics.success_rate > 0.85, error_tol * 0.98, error_tol)

        MIN_TOLERANCE = 0.01
        error_tol = jnp.maximum(jnp.asarray(error_tol), MIN_TOLERANCE)

        # separa o modelo novamente para a forma funcional com o estado
        _, next_state = nnx.split((step_model, step_opt, step_rngs))

        carry_next = (next_state, next_mjx_data, next_buffer, error_tol)
        return carry_next, (losses, metrics, error_tol)

    final_carry, (all_losses, all_metrics, all_error_tol) = jax.lax.scan(
        update_step, (state, mjx_data, buffer, start_error_tol), jnp.arange(num_updates)
    )

    final_state, final_mjx_data, final_buffer, _ = final_carry

    # aplica o estado final nas instancias que estão fora do loop
    nnx.update((model, optimizer, rngs), final_state)

    return final_mjx_data, final_buffer, all_losses, all_metrics, all_error_tol


######################################################################################################################

model_path = "/home/lucas/Documentos/MLProjects/monadic_ppo"

EPOCHS = 2
NUM_ENVS = 4096
BUFFER_LENGTH = 512
UPDATES = 200
MINIBATCH_SIZE = 8192
START_ERROR_TOL = 0.06


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


def create_optimizer(epochs, num_envs, buffer_length, updates, minibatch_size):
    full_batch_size = num_envs * (buffer_length + 1)
    steps_per_epoch = epochs * (full_batch_size // minibatch_size)
    total_steps = steps_per_epoch * updates

    lr_scheduler = optax.schedules.cosine_onecycle_schedule(
        peak_value=5e-4,
        pct_start=0.3,  # 30% do treino subindo (warm-up), 70% descendo
        div_factor=5.0,  # LR inicial = peak_value / div_factor
        final_div_factor=10.0,  # LR final = LR inicial / final_div_factor para o ajuste fino,
        transition_steps=total_steps,
    )

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_scheduler),
    )


key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(key)
model = ThorAgent(env, rngs)
optimizer = nnx.Optimizer(
    model,
    create_optimizer(EPOCHS, NUM_ENVS, BUFFER_LENGTH, UPDATES, MINIBATCH_SIZE),
    wrt=nnx.Param,
)

# cria um mjx_data inicial e reseta um dado ambiente
initial_mjx_data = mjx.make_data(env.mjx_model)

batched_mjx_data = jax.tree_util.tree_map(
    lambda x: jax.numpy.repeat(x[None], NUM_ENVS, axis=0), initial_mjx_data
)

buffer = new_buffer(
    NUM_ENVS,
    BUFFER_LENGTH,
    model.policy.obs_size,
    model.value.obs_size,
    model.policy.action_size,
)


# treino
mjx_data, buffer, losses, metrics, error_tol = run_multiple_updates(
    model,
    optimizer,
    rngs,
    batched_mjx_data,
    buffer,
    BUFFER_LENGTH,
    EPOCHS,
    MINIBATCH_SIZE,
    UPDATES,
    NUM_ENVS,
    START_ERROR_TOL,
)

# (updates, epoch)
print(f"losses shape: {losses.shape}")
print(f"entropy  loss shape: {metrics.loss_metrics.entropy_loss.shape}")
print(f"policy loss shape: {metrics.loss_metrics.policy_loss.shape}")
print(f"value loss shape: {metrics.loss_metrics.value_loss.shape}")
print(f"kl_div shape: {metrics.loss_metrics.kl_div.shape}")
print(f"error shape: {metrics.acumulated_error.shape}")

# 1. Convert JAX arrays to NumPy in one clean line
losses_np, entropy_np, kl_np, is_safe_np = (
    np.asarray(losses),
    np.asarray(metrics.loss_metrics.entropy_loss),
    np.asarray(metrics.loss_metrics.kl_div),
    np.asarray(metrics.loss_metrics.is_safe),
)
error_np, success_np, failure_np, done_np = (
    np.asarray(metrics.acumulated_error),
    np.asarray(metrics.avg_success),
    np.asarray(metrics.avg_failure),
    np.asarray(metrics.avg_done),
)

# We only average the 2D arrays (Loss, Entropy, KL)
plot_configs = [
    (losses_np, "Training Loss", "blue"),
    (entropy_np, "Entropy", "orange"),
    (kl_np, "KL Divergence", "red"),
]

np.savez(
    "training_metrics",
    losses=losses_np,
    entropy=entropy_np,
    kl_div=kl_np,
    is_safe=is_safe_np,
    error=error_np,
    success=success_np,
    failure=failure_np,
    done=done_np,
)

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

success_rate_np = np.asarray(metrics.success_rate)
error_tol_np = np.asarray(error_tol)

ax3 = fig.add_subplot(3, 2, 4)
ax3b = ax3.twinx()

ax3.plot(success_rate_np, color="blue", label="Success rate")
ax3.set(xlabel="Updates", ylabel="SR", title="Success Rate/Error tol")
ax3.grid(True, alpha=0.5)
ax3.legend()

ax3b.plot(error_tol_np, color="green", label="Error tol")
ax3b.set(ylabel="ET")
ax3b.grid(True, alpha=0.5)
ax3b.legend()


# Plot the standard 1D Rollout metrics in the remaining slots
ax4 = fig.add_subplot(3, 2, 5)
ax4.plot(error_np, color="purple")
ax4.set(xlabel="Updates", title="Exp Mean Rollout Error")
ax4.grid(True, alpha=0.5)

ax5 = fig.add_subplot(3, 2, 6)
ax5.plot(success_np, color="green", label="Success")
ax5.plot(failure_np, color="red", label="Failure")
ax5.set(xlabel="Updates", title="Success and failure avg count", ylabel="Avg")
ax5.grid(True, alpha=0.5)
ax5.legend()

ax5b = ax5.twinx()
ax5b.plot(done_np, color="black", label="Done")
ax5b.set(ylabel="Avg Dones")
ax5b.grid(True, alpha=0.5)
ax5b.legend()


plt.savefig("training_plots.png")
print("\nTraining plots saved to training_plots.png")

# ==========================================
# SALVANDO OS PESOS DO MODELO
# ==========================================
# Define o caminho (usando o seu model_path existente)
ckpt_dir = os.path.abspath(f"{model_path}/checkpoints/ppo_thor_final")

# Garante que a pasta pai exista
os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)

# 1. Extrai apenas o estado (pesos) do modelo já treinado
_, model_state = nnx.split(model)

# 2. Salva usando o Orbax
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save(ckpt_dir, model_state, force=True)

print(f"✅ Pesos do modelo salvos com sucesso em: {ckpt_dir}")
