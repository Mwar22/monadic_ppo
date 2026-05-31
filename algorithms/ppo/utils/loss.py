# -*- coding:utf-8 -*-
###
# File:  loss.py
# Created Date: 25/05/2026 07:23:03
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 31/05/2026 02:36:57
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
from algorithms.ppo.src.agent import Agent

def ppo_loss(
    agent: Agent,
    policy_obs,
    value_obs,
    actions,  
    advantages,  
    returns,  
    old_log_probs,
    c1=0.5,
    c2=0.1,
    eps=1e-4,
):
    
    advantages = jax.lax.stop_gradient(advantages)
    returns = jax.lax.stop_gradient(returns)
    policy_obs = jax.lax.stop_gradient(policy_obs)
    value_obs = jax.lax.stop_gradient(value_obs)
    actions = jax.lax.stop_gradient(actions)
    old_log_probs = jax.lax.stop_gradient(old_log_probs)

    #normalizamos as vantagens
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # logprob, entropia segundo a politica atual
    logprobs, entropy = agent.policy.evaluate_actions(policy_obs, actions)

    # value segundo parametros atuais
    values = agent.value(policy_obs).squeeze(-1)

    # clipa log_ratio (jnp.exp(10) é ~22000 (bem grande) e  jnp.exp(-10) é ~0.00004, sendo mais que suficiente)
    safe_logratio = jnp.clip(logprobs - old_log_probs, -10.0, 10.0)
    ratio = jnp.exp(safe_logratio)

    # KL divergence
    kl_div = jnp.mean(-safe_logratio)

    # perdas de entropia
    entropy_loss = jnp.mean(entropy)

    # perda para a política
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1 - eps, 1 + eps) * advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped)) 

    # perdas pela função de valor
    value_loss = jnp.mean((returns -values) ** 2)

    total_loss = policy_loss + c1*value_loss - c2*entropy_loss
    return total_loss, (entropy_loss, policy_loss, value_loss, kl_div)