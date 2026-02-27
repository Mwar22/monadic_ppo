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
