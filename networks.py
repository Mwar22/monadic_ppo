"""
Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------
Redes que representam actor/critic
"""

import jax
import flax.serialization
import jax.numpy as jnp
import flax.linen as nn
from dataclassutils import NetworksSettings, NetworkParameters, RunningParameters
from typing import cast


##################################################### FUNÇÕES #########################################################

#cria configuração relacionada as redes (actor/critic)
def create_networks(rng:jax.Array, obs_size:int, action_size:int):
    rng, rng_actor, rng_critic = jax.random.split(rng, 3)
    dummy_obs= jnp.zeros((1, obs_size)) 

    actor = Actor(action_size, discrete=True)
    critic = Critic()
    actor_params = actor.init(rng_actor, dummy_obs)
    critic_params = critic.init(rng_critic, dummy_obs)

    return rng, NetworksSettings(obs_size, action_size, actor, critic), NetworkParameters(actor_params, critic_params)

##################################################### MODELOS #########################################################
activation = lambda x: nn.leaky_relu(x)

hidden_init = nn.initializers.orthogonal(jnp.sqrt(2))
actor_init = nn.initializers.orthogonal(0.005)
critic_init = nn.initializers.orthogonal(0.01)

min_alpha_beta = 1.1
max_alpha_beta = 100.0
smooth_bound = lambda x: min_alpha_beta + (max_alpha_beta - min_alpha_beta) * jax.nn.sigmoid(x)
                              
class Actor(nn.Module):
    action_dim: int
    discrete: bool

    @nn.compact
    def __call__(self, obs):
        # Primeira camada com skip connection
        x1 = nn.Dense(256, kernel_init=hidden_init, dtype=jnp.float32)(obs)
        x1 = activation(x1)
        
        x2 = nn.Dense(256, kernel_init=hidden_init, dtype=jnp.float32)(x1)
        x2 = activation(x2)
        
        x3 = nn.Dense(64, kernel_init=hidden_init, dtype=jnp.float32)(x2)
        x3 = activation(x3)

        alpha_logits = nn.Dense(self.action_dim, kernel_init=actor_init)(x3)
        beta_logits = nn.Dense(self.action_dim, kernel_init=actor_init)(x3)
        
        # limita os valores para alpha e beta dentro de faixas conhecidas (evita inst. numerica)
        alpha = smooth_bound(alpha_logits)
        beta = smooth_bound(beta_logits)

        return alpha, beta


class Critic(nn.Module):
    @nn.compact
    def __call__(self, obs):
       
        x1 = nn.Dense(256, kernel_init=hidden_init, dtype=jnp.float32)(obs)
        x1 = activation(x1)
        
        x2 = nn.Dense(256, kernel_init=hidden_init, dtype=jnp.float32)(x1)
        x2 = activation(x2)
        
        x3 = nn.Dense(64, kernel_init=hidden_init, dtype=jnp.float32)(x2)
        x3 = activation(x3)

        value = nn.Dense(1, kernel_init=critic_init, dtype=jnp.float32)(x3)
        return value.squeeze(-1)


from jax import jit, custom_jvp

@custom_jvp
@jit
def cauchy_activation(x, lambda1= 0.01, lambda2= 0.01, d=1.0):
    #para estabilidade numerica
    eps = 1e-12
    
    x2_d2 = x**2 + d**2 + eps
    return (lambda1 * x + lambda2) / x2_d2

# gradiente customizado
@cauchy_activation.defjvp
def cauchy_activation_jvp(primals, tangents):
    x, lambda1, lambda2, d = primals
    x_dot, lambda1_dot, lambda2_dot, d_dot = tangents

    #para estabilidade numerica
    eps = 1e-12

    #cria uma versão "clipada" de d, diferenciavel, e evitando valores muito pequenos (explosão de gradiente)
    d_safe = jnp.sqrt(d**2 + 1e-6) # se |d|  << 1e-3, d_safe ≃ 1e-3. Caso contrário d_safe ≃ d
    
    #calcula valores que se repetem
    x2 = x**2
    d2 = d_safe**2
    x2_d2 = x2 + d2 + eps
    x2_d2_sq = x2_d2 ** 2
    inv = 1 / x2_d2_sq

    
    y = (lambda1 * x + lambda2) / x2_d2

    #derivadas parciais
    dy_dx = ((d2 - x2) * lambda1  - 2 * x * lambda2)*inv
    dy_dlambda1 = x / x2_d2
    dy_dlambda2 = 1 / x2_d2
    dy_dd = -2 * d_safe * (lambda1 *x + lambda2) * inv * (d / d_safe)

    tangent_out = dy_dx * x_dot + dy_dlambda1 * lambda1_dot + dy_dlambda2 * lambda2_dot + dy_dd * d_dot
    return y, tangent_out

class CauchyActivationModule(nn.Module):
    init_lambda1: float = 0.01
    init_lambda2: float = 0.01
    d: float = 1.0

    @nn.compact
    def __call__(self, x):
        # trainable params
        lambda1 = self.param("lambda1", lambda rng: jnp.array(self.init_lambda1))
        lambda2 = self.param("lambda2", lambda rng: jnp.array(self.init_lambda2))
        return cauchy_activation(x, lambda1, lambda2, self.d)