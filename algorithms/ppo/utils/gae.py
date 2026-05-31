# -*- coding:utf-8 -*-
###
# File:  gae.py
# Created Date: 26/05/2026 11:12:39
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 30/05/2026 03:27:18
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

def general_advantage_estimator(
    rewards: jax.Array,  
    dones: jax.Array,   
    values: jax.Array,
    gamma: float,
    lam: float    
):
    def gae_scan_fn(carry, transition):
        gae_next = carry
        
        # Desempacotamos o tempo 't' e o 't+1' diretamente
        reward_t, value_t, value_next, done_t = transition

        # Se done_t == 1, o episódio acabou. O estado 't+1' pertence a um novo episódio.
        # Portanto, multiplicamos por 0 para isolar o futuro e não vazar vantagem.
        not_done = 1.0 - done_t
        
        delta = reward_t + gamma * value_next * not_done - value_t
        gae = delta + gamma * lam * gae_next * not_done

        # Retornamos o GAE atualizado para o carry e salvamos o GAE na saída
        return gae, gae

    # Fatiamos os arrays para alinhar o presente (t) e o futuro (t+1)
    transitions = (
        rewards[:-1],       # r_t
        values[:-1],   # V_t
        values[1:],    # V_{t+1} (O valor do próximo estado, já alinhado!)
        dones[:-1]        # d_t
    )
    
    # O carry inicial é apenas um vetor de zeros (num_envs,)
    initial_carry = jnp.zeros(rewards.shape[1])
    
    # Scan reverso
    _, advantages = jax.lax.scan(
        gae_scan_fn, 
        initial_carry, 
        transitions, 
        reverse=True
    )

    returns = advantages + values[:-1]

    return advantages, returns