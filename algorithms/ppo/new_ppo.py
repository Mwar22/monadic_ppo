# -*- coding:utf-8 -*-
###
# File:  new_ppo.py
# Created Date: 31/05/2026 01:29:51
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 31/05/2026 02:14:41
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
import optax
import jax.numpy as jnp
from functools import partial
from mujoco import mjx
from algorithms.ppo.rollout import RolloutBuffer, rollout
from algorithms.ppo.src.agent import Agent
from algorithms.ppo.src.enviroment import MujocoEnv
from algorithms.ppo.utils.loss import ppo_loss


def train_epochs(
    optimizer_state: optax.OptState,
    buffer: RolloutBuffer,
    loss_c1=0.5,
    loss_c2_base=0.05,
    loss_c2_bonus=0.3,
):
    def grad_norm(grads):
        leaves = jax.tree_util.tree_leaves(grads)

        sq_sum = 0.0
        for g in leaves:
            sq_sum += jnp.sum(jnp.square(g))

        return jnp.sqrt(sq_sum)

    def single_epoch(carry, _):
        _parameters, _optimizer_state = carry

        def loss_fn(par):
            #escalonamento para c2
            current_c2 = loss_c2_base + loss_c2_bonus * (1.0 - success_rate)

            return ppo_loss(
                par,
                settings,
                buffer.obs_buffer,
                buffer.action_buffer,
                advantages,
                returns,
                buffer.logprob_buffer,
                batch_ptr=buffer.ptr,
                c1 = loss_c1,
                c2 = current_c2
            )

        # calcula os gradientes e atualiza os parametros
        (loss_val, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            _parameters
        )
        updates, new_optim_state = settings.optimizer.update(grads, _optimizer_state)
        new_parameters = cast(
            NetworkParameters, optax.apply_updates(_parameters, updates)
        )

        """
        grads_have_nan = tree_any_nan(grads)
        jax.lax.cond(
            grads_have_nan,
            lambda _: jax.debug.print(" NAN DETECTED IN GRADIENTS! Loss: {}", loss_val),
            lambda _: None,
            None,
        )
        """

        new_carry = (new_parameters, new_optim_state)

        return new_carry, {
            "loss": loss_val,
            "grad_norm": grad_norm(grads),
            **aux_metrics,
        }

    # após  o scan, teremos o seguinte:
    # metric[key].shape  = (epochs, *metric_shape)
    # Ex: loss.shape  = (epochs, )
    (new_parameters, new_optim_state), metrics = jax.lax.scan(
        single_epoch,
        (network_parameters, optimizer_state),
        jnp.arange(settings.epochs),
    )

    return new_parameters, new_optim_state, metrics


def update_goal(state, progress, settings: TrainingSettings):
    vmapped_get_goal = jax.vmap(
        partial(get_goal, settings.robot_shared_data.range_config, progress)
    )
    batched_rng, batched_goal = vmapped_get_goal(state["rng"])
    return {
        **state,
        "rng": batched_rng,
        "goal": batched_goal,
        "step": jnp.zeros_like(state["step"]),
        "success": jnp.zeros_like(state["success"]),
    }


jax.jit


def ppo_train(
    rng: jax.Array,
    starting_network_params: NetworkParameters,
    settings: TrainingSettings,
):
    """The complete, JIT-compiled training function."""

    def get_success_rate(state):
        # Transforma qualquer contagem > 0 em 1 (sucesso) ou 0 (falha)
        # A média dará um valor entre  0.0 e 1.0 (0% a 100%)
        # success.shape: (num_envs, )
        success = state["success"]
        return jax.lax.stop_gradient(jnp.mean(success))
    
    def get_fullbuffer_rate(buffer: BatchedBuffer):
        # retorna o percentual de ambientes que pararam, mas não foram por ter atingido o objetivo.
        # ou seja precisaram de mais passos para atingir o alvo
        x = ~buffer.done_flag & buffer.stop_flag
        return jax.lax.stop_gradient(jnp.mean(x))

    def collect_rollouts(
        state, runpar: RunningParameters, network_params: NetworkParameters
    ):
        # Faz um rollout (usando a função vetorizada)
        new_state, new_buffer = rollout(settings, network_params, state, runpar)

        # Vetoriza a função GAE
        vmapped_gae = jax.vmap(
            partial(general_advantage_estimator, settings, network_params),
            in_axes=(0, 0, 0, 0),
        )

        # calcula as vantagens (usando a função vetorizada)
        advantages, returns = vmapped_gae(
            new_buffer.obs_buffer,
            new_buffer.reward_buffer,
            new_buffer.ptr,
            new_buffer.done_flag,
        )

        return new_state, new_buffer, advantages, returns

    def new_goal_step(carry, goal_idx):
        """This is the body of the scan, representing one full update."""
        runpar, optim_state, network_params, current_state = carry
        runpar = cast(RunningParameters, runpar)
        network_params = cast(NetworkParameters, network_params)

        # atualiza os goals com base no estado atual
        new_goal_state = update_goal(current_state, runpar.progress, settings)

        # coleta os dados e atualiza os parâmetros correntes
        new_state, batched_buffer, advantages, returns = collect_rollouts(
            new_goal_state, runpar, network_params
        )
        success_rate = get_success_rate(new_state)
        full_buffer_termination_rate = get_fullbuffer_rate(batched_buffer)

        # metricas tem shape (epochs, *metric_shape)
        network_params, optim_state, training_metrics = train_epochs(
            settings, network_params, optim_state, batched_buffer, advantages, returns, success_rate
        )
        runpar = runpar.update(batched_buffer.obs_buffer, success_rate)

        newcarry = (runpar, optim_state, network_params, new_state)
        err_tol = settings.robot_shared_data.reward_config.err_tol.update(
            runpar.progress
        )
        return newcarry, (
            training_metrics,
            success_rate,
            batched_buffer.reward_buffer,
            new_state["err"],
            err_tol,
            new_state["ctrl_norm"],
            full_buffer_termination_rate,
        )

    # loop principal de trainamento, executado por lax.scan
    runpar = RunningParameters.init(
        (settings.network_settings.obs_size,), update_threshold=settings.target_success
    )
    rng1, initial_state = create_initial_state(rng, runpar.progress, settings)

    # após  o scan, teremos o seguinte:
    # training_metrics[key].shape = (numberof_goals, epochs, *metric_shape)
    # mean_envs_success_rate.shape = (numberof_goals,)
    # rewards.shape = (numberof_goals, num_envs, rollout_steps +1)
    (
        final_carry,
        (training_metrics, mean_envs_success_rate, rewards, err, err_tol, ctrl_norm, full_buffer_termination_rate),
    ) = jax.lax.scan(
        new_goal_step,
        (runpar, settings.optimizer_state, starting_network_params, initial_state),
        jnp.arange(settings.active_numberof_goals),
    )

    # shape das perdas é: (numberof_goals, epochs,). Para exibir no formato (epochs, )
    avg_loss = jnp.mean(training_metrics["loss"], axis=0)
    avg_entropy = jnp.mean(training_metrics["entropy"], axis=0)
    avg_grad_norm = jnp.mean(training_metrics["grad_norm"], axis=0)
    avg_kl_div = jnp.mean(training_metrics["kl_div"], axis=0)

    # shape de recompensas é: (numberof_goals, num_envs, rollout_steps +1)
    # jax.debug.print("rewards shape: {}", rewards.shape)

    total_rollout_reward = jnp.sum(
        rewards, axis=2
    )  # soma as recompensas do rollout: (numberof_goals, num_envs)
    mean_rewards_vs_goals = jnp.mean(total_rollout_reward, axis=1)  # (numberof_goals, )

    # mean reward across timestamp
    total_goals_reward = jnp.sum(rewards, axis=0)  # (num_envs, rollout_steps+1)
    mean_rewards_vs_timestamp = jnp.mean(
        total_goals_reward, axis=0
    )  # (rollout_steps+1, )

    print(f"err shape: {err.shape}")
    print(f"ctrl_norm shape:{ctrl_norm.shape}")

    # err.shape = (numberof_goals, num_envs,)
    avg_err = jnp.mean(err, axis=1)
    avg_ctrl_norm = jnp.mean(ctrl_norm, axis=1)

    metrics = {
        "avg_loss": avg_loss,
        "avg_entropy": avg_entropy,
        "avg_gradnorm": avg_grad_norm,
        "mean_rewards_vs_goals": mean_rewards_vs_goals,
        "mean_rewards_vs_timestamp": mean_rewards_vs_timestamp[:-1],
        "success_rate": mean_envs_success_rate,
        "avg_err": avg_err,
        "err_tol": err_tol,
        "avg_kl_div": avg_kl_div,
        "avg_ctrl_norm": avg_ctrl_norm,
        "fullbuffer_termination_rate": full_buffer_termination_rate,
    }

    # final_carry = (runpar, optim_state, network_params)
    return final_carry, metrics


def create_initial_state(rng: jax.Array, progress, settings: TrainingSettings):
    rng, rng1 = jax.random.split(rng)

    # Inicializa os estados para os ambientes em paralelo
    # (num_envs, features_dim)
    num_envs = settings.num_envs
    batched_rng = jax.random.split(rng, num_envs)

    mjx_data = mjx.make_data(settings.robot_shared_data._mjx_model)
    batched_mjx_data = jax.tree_util.tree_map(
        lambda x: jax.numpy.repeat(x[None], num_envs, axis=0), mjx_data
    )

    vmapped_get_goal = jax.vmap(
        partial(get_goal, settings.robot_shared_data.range_config, progress)
    )
    batched_rng, batched_goal = vmapped_get_goal(batched_rng)

    # Crie um estado temporário para rodar o pipeline
    temp_state = {
        "rng": batched_rng,
        "goal": batched_goal,
        "mjx_data": batched_mjx_data,
        "obs": jnp.zeros((num_envs, settings.network_settings.obs_size)),  # placeholder
        "last_action": jnp.zeros((num_envs, settings.network_settings.action_size)),
        "err": jnp.ones((num_envs,)) * jnp.inf,
        "step": jnp.zeros((num_envs,)),
        "success": jnp.zeros((num_envs,)),
        "ctrl_norm": jnp.zeros((num_envs,)),
    }

    # Rode apenas o pipeline de observação para obter o estado REAL inicial
    # Isso garante que a primeira obs que o agente vê não seja zero
    runpar_init = RunningParameters.init(
        (settings.network_settings.obs_size,), settings.numberof_goals
    )

    def get_single_obs(s):
        # s é um único 'state' (scalars/unbatched arrays)
        # StateMonad.pure({}) inicia o pdata como um dict vazio
        pipe = obs_pipeline(
            settings.robot_shared_data,
            runpar_init.obs_stat,
            StateMonad.pure({}),
            settings.obs_noise_scale,
        )
        _, out_data = pipe.run(s)
        return out_data["obs"]

    # vmap mapeia 'get_single_obs' sobre a primeira dimensão de todos os arrays no temp_state
    initial_obs = jax.vmap(get_single_obs)(temp_state)

    return rng1, {
        **temp_state,
        "obs": initial_obs,  # Agora contém dados reais do MuJoCo
    }
