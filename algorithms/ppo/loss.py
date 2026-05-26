# -*- coding:utf-8 -*-
###
# File:  loss.py
# Created Date: 25/05/2026 07:23:03
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 25/05/2026 03:57:08
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
import jax.numpy as jnp
from algorithms.ppo.networks import Policy, Value

def ppo_loss(
    policy_network: Policy,
    value_network: Value,
    batch_obs,  # shape: (num_envs, max_steps +1, *obs_shape)
    batch_actions,  # shape: (num_envs, max_steps +1, *action_shape)
    batch_advantages,  # shape: (num_envs, max_steps)
    batch_returns,  # shape: (num_envs, max_steps)
    old_log_probs,  # shape: (num_envs, max_steps +1)
    batch_ptr,  # ADICIONADO: shape (num_envs,) vindo do buffer.ptrff
    c1,
    c2,
    eps=1e-4,
):
    
    batch_advantages = jax.lax.stop_gradient(batch_advantages)
    batch_returns = jax.lax.stop_gradient(batch_returns)
    batch_obs = jax.lax.stop_gradient(batch_obs[:, :-1, :])
    batch_actions = jax.lax.stop_gradient(batch_actions[:, :-1, :])
    old_log_probs = jax.lax.stop_gradient(old_log_probs[:, :-1])
    batch_ptr = jax.lax.stop_gradient(batch_ptr)

    # mascara para os passos validos
    max_steps = batch_advantages.shape[1]
    steps_arr = jnp.arange(max_steps)

    # Compara a matriz de passos com o ponteiro de cada ambiente
    valid_mask = steps_arr[None, :] < batch_ptr[:, None]  # shape: (num_envs, max_steps)
    total_valid_elements = jnp.maximum(jnp.sum(valid_mask), eps)

    def masked_norm(values, mask):
        # media dos elementos válidos
        mean = jnp.sum(values * mask) / total_valid_elements

        # variancia dos elementos validos
        variance = jnp.sum(jnp.square(values - mean) * mask) / total_valid_elements
        std = jnp.sqrt(variance + eps)

        # normaliza e zera o padding
        normalized_adv = ((values - mean) / std) * mask
        return normalized_adv

    # normaliza as vantagens
    batch_advantages = masked_norm(batch_advantages, valid_mask)

    # forward
    alpha, beta = policy_network(batch_obs)
    values = value_network(batch_obs)

    # logprobs
    clipped_actions = jnp.clip(batch_actions, eps, 1.0 - eps)
    logprobs = jax.scipy.stats.beta.logpdf(clipped_actions, alpha, beta)
    logprobs = jnp.sum(logprobs, axis=2)

    # ratio
    logratio = logprobs - old_log_probs

    # Clipa o valor antes de passar para a exponencial
    # jnp.exp(10) é ~22000 (bem grande) e  jnp.exp(-10) é ~0.00004, sendo mais que suficiente
    safe_logratio = jnp.clip(logratio, -10.0, 10.0)
    ratio = jnp.exp(safe_logratio)

    # KL divergence
    kl_div = jnp.sum(-safe_logratio * valid_mask) / total_valid_elements

    # PPO objective (Modificado para aplicar a máscara)
    unclipped = ratio * batch_advantages
    clipped = jnp.clip(ratio, 1 - eps, 1 + eps) * batch_advantages

    raw_policy_loss = -jnp.minimum(unclipped, clipped)
    policy_loss = jnp.sum(raw_policy_loss * valid_mask) / total_valid_elements

    # Value loss (Modificado para ignorar passos inválidos)
    raw_value_loss = (batch_returns - values) ** 2
    value_loss = c1 * (jnp.sum(raw_value_loss * valid_mask) / total_valid_elements)

    # entropia = E[-log pi(a)]
    raw_entropy = -logprobs
    entropy = (jnp.sum(raw_entropy * valid_mask) / total_valid_elements)

    total_loss = policy_loss + value_loss - c2*entropy
    return total_loss, (entropy, policy_loss, value_loss, kl_div)