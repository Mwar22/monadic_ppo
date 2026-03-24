"""
mathutils.py

Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------

Contem funções auxiliares com operações matemáticas auxiliares.
"""

import jax
from jax import numpy as jnp
from jax.scipy.special import gammaln, digamma


class ProgressScheduler:
    @staticmethod
    def linear(step, total_steps):
        return jnp.clip(step / total_steps, 0.0, 1.0)

    @staticmethod
    def power(step, total_steps, p=2.0):
        # p > 1: demora mais a crescer no início (Warm-up)
        # p < 1: cresce rápido no início e abranda no fim
        return jnp.power(jnp.clip(step / total_steps, 0.0, 1.0), p)

    @staticmethod
    def sigmoid(step, total_steps, center=0.5, sharpness=10.0):
        # Curva em S: início lento, meio rápido, final lento
        x = step / total_steps
        return jax.nn.sigmoid(sharpness * (x - center))
    

def conv2jax_quat(mujoco_quat: jnp.ndarray) -> jnp.ndarray:
    """Converte quaternion no formato (w, x, y, z) -> (x, y, z, w)"""
    return jnp.array([mujoco_quat[1], mujoco_quat[2], mujoco_quat[3], mujoco_quat[0]])


def exp_scale_reward(gain, sigma, value: jax.Array) -> jax.Array:
    return gain * jnp.exp(-value / sigma)


def l1_l2_reward(gain_l1, gain_l2, value: jax.Array):
    return gain_l2 * jnp.linalg.norm(value, ord=2) + gain_l1 * jnp.linalg.norm(
        value, ord=1
    )


def _cost_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    """
    Penaliza as diferenças entre os vetores de ações por meio da norma L2

    Parameters
    ----------
    act: jax.Array
        Ação atual.

    last_act: jax.Array
        Ação anterior
    """
    return jnp.linalg.norm(act - last_act, ord=2)


def stand_still_reward(
    gain,
    position_velocities: jax.Array,
    orientation_velocities: jax.Array,
    default_pose: jax.Array,
    joint_angles: jax.Array,
) -> jax.Array:
    """
    Penaliza caso o comando (velocidades de movimentação) indicar que o robô deva estar parado.
    Caso penalizado, a penalização é de acordo com a norma L1 encima das diferenças entre os
    ângulos em joint_angles e a pose _default_pose.

    Parameters
    ----------
    commands: Dict[str, jax.Array]
        Dicionário que mapeia uma informação dos comandos sampleados.

    joint_angles: jax.Array
        Angulos atuais das juntas do robô
    """
    linear_velocity = jnp.linalg.norm(position_velocities)
    angular_velocity = jnp.linalg.norm(orientation_velocities)

    mask = jnp.logical_and(linear_velocity < 0.001, angular_velocity < 0.001)
    return (
        gain * jnp.linalg.norm(joint_angles - default_pose) * mask.astype(jnp.float32)
    )


def cont_sample_beta(logits: jax.Array, rng: jax.Array, min_alpha_beta=0.1):
    """
    Sample continuous actions in [0,1] using independent Beta distributions
    parameterized by logits.

    Args:
        logits: shape (action_dim,), any real numbers
        rng: JAX PRNGKey
        min_alpha_beta: minimum value for alpha and beta to avoid numerical issues

    Returns:
        action: shape (action_dim,)
        logprob: shape (action_dim,)
    """

    # mapeia os logits para parametros positivos para serem utilizados na distribuição beta
    alpha_logits, beta_logits = jnp.split(logits, 2, axis=-1)
    alpha = jax.nn.softplus(alpha_logits) + min_alpha_beta
    beta  = jax.nn.softplus(beta_logits) + min_alpha_beta

    # separa o rng para amostras independentes
    rng, subkey = jax.random.split(rng)
    actions = jax.random.beta(subkey, alpha, beta)

    # Clip actions to be just inside (0, 1) to avoid -inf logpdf
    clipped_actions = jnp.clip(actions, 1e-6, 1.0 - 1e-6)

    # logprob para cada dimensão
    logprobs = jax.scipy.stats.beta.logpdf(clipped_actions, alpha, beta)
    return actions, jnp.sum(logprobs, axis=-1)

def beta_entropy(alpha, beta):
    lnB = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    H = (
        lnB
        - (alpha - 1) * digamma(alpha)
        - (beta - 1) * digamma(beta)
        + (alpha + beta - 2) * digamma(alpha + beta)
    )
    return H


def shift_array(array: jax.Array, ptr: jax.Array)-> jax.Array:
    """
    Função auxiliar para deslocar um determinado array buffer para a direita,
    de tal forma que o ultimo elemento do topo da "pilha", indicado por ptr, 
    seja o ultimo elemento do array:
     
    Se:
        a = [1, 3, 4, 6, 7, 0, 0, 0, 0], e 
        ptr = 5

    Então
        a_shifted = [0, 0, 0, 0, 1, 3, 4, 6, 7]

    Parameters
    ----------
    array : jax.Array
        Trata-se de um array/pilha começando na posição 0
    ptr : jax.Array
        Posição do próximo elemento que poderá ser incluso

    Returns
    -------
    jax.Array
        retorna o array trabalhado.
    """
    steps = array.shape[0]
    shift = steps - ptr
    
    # shift circular para empurrar os dados validos para o fim
    shifted = jnp.roll(array, shift, axis=0)
    
    # tudos os indices abaixo de shift são marcados para serem zerados
    mask = jnp.arange(steps) < shift
    
    # concatena as dimensões para uma mascara para qualquer formato,
    # tal como um array de multiplas dimensões
    broadcast_dims = (steps,) + (1,) * (array.ndim - 1)
    return jnp.where(mask.reshape(broadcast_dims), 0.0, shifted)
