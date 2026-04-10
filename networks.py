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
from dataclassutils import NetworksSettings, NetworkParameters
from typing import cast


##################################################### FUNÇÕES #########################################################
def save_params(network_params: NetworkParameters, filename="trained_params.msgpack"):
    state_bytes = flax.serialization.to_bytes(network_params)

    with open(filename, "wb") as f:
        f.write(state_bytes)

    print(f"Model weights saved successfully to: {filename}")

def load_params(empty_params: NetworkParameters, filename="trained_params.msgpack"):
    with open(filename, "rb") as f:
        state_bytes = f.read()

    # restaura os parametros a partir dos dados serializados
    return cast(NetworkParameters, flax.serialization.from_bytes(empty_params, state_bytes))

#cria configuração relacionada as redes (actor/critic)
def create_networks(rng:jax.Array, obs_size:int, action_size:int):
    rng, rng_actor, rng_critic = jax.random.split(rng, 3)
    dummy_obs= jnp.zeros((1, obs_size)) 

    actor = Actor(action_size, discrete=True)
    critic = Critic()
    actor_params = actor.init(rng_actor, dummy_obs),
    critic_params = critic.init(rng_critic, dummy_obs),

    return rng, NetworksSettings(obs_size, action_size, actor, critic), NetworkParameters.init(actor_params, critic_params)

##################################################### MODELOS #########################################################
activation_str = "sigmoid"
activation = lambda x: nn.relu(x)

class Actor(nn.Module):
    action_dim: int
    discrete: bool

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(256)(obs)
        x = activation(x)
        x = nn.LayerNorm()(x)
        
   
        x = nn.Dense(256)(x)
        x = activation(x)
        x = nn.LayerNorm()(x)
       
        # 2 pois é uma distribuição, gerando metade para os parametros alfa e metade para beta
        logits = nn.Dense(2 * self.action_dim)(x)
        return logits


class Critic(nn.Module):
    activation_name: str = "relu" #default

    @nn.compact
    def __call__(self, obs):
       
        x = nn.Dense(256)(obs)
        x = activation(x)
        x = nn.LayerNorm()(x)
        
        x = nn.Dense(256)(x)
        x = activation(x)
        x = nn.LayerNorm()(x)
        
        value = nn.Dense(1)(x)
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