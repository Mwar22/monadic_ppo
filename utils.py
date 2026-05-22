"""
mathutils.py

Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------

Contem funções auxiliares com operações matemáticas auxiliares.
"""

import jax
import flax.serialization
import mujoco
from mujoco import MjModel  # type: ignore
from etils import epath
from jax import numpy as jnp
from jax.scipy.special import gammaln, digamma 
from jax.scipy.spatial.transform import Rotation
from typing import Union, Dict, Any, List, TypeVar, cast
from monads import MaybeM

mujoco: Any
T = TypeVar("T")

class Scheduler:
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


def exp_scale_reward2(gain, x_zero: float, error_value: jax.Array) -> jax.Array:
    """_summary_

    Parameters
    ----------
    gain : Valor da recompensa quando o erro é zero
    x_zero : float
        Valor do erro para o qual a recompensa é nula (cruza o eixo das abscissas)
    error_value : jax.Array
        Valor do erro a ser avaliado

    Returns
    -------
    jax.Array
        valor da recompensa
    """
    inv_omega = 1.763222834351896710225201776951 # Inverso da contante omega (que por sí é o resultado de W(1), onde W é a função W de lambert. Solução de u*exp(u) = 1)
    sigma = x_zero*inv_omega
    return gain * (jnp.exp(-error_value / sigma) - error_value)

def exp_scale_reward(gain, x_zero: float, error_value: jax.Array) -> jax.Array:
    """
    Recompensa Exponencial Pura:
    - Se error_value == 0 -> Recompensa = gain
    - Se error_value é grande -> Recompensa se aproxima de 0 (mas nunca é negativa)
    - x_zero atua como o 'raio' de suavidade (sigma).
    """
    # Usamos uma Gaussiana (Sino) para dar um platô suave perto do alvo
    return -gain * error_value

def l1_l2_reward(gain_l1, gain_l2, value: jax.Array):
    return gain_l2 * jnp.linalg.norm(value, ord=2) + gain_l1 * jnp.linalg.norm(
        value, ord=1
    )

def position_error(goal_position: jax.Array, tool_position: jax.Array) -> jax.Array:
    """
    Calcula o erro de posição.

    Parameters
    ----------
    data: mjx.Data
        Estado dinâmico que atualiza a cada step.

    info: dict[str, Any]
        Dicionario de informações
    """
    return jnp.linalg.norm(goal_position - tool_position, ord=2)


def orientation_error(
    goal_orientation: jax.Array, tool_orientation: jax.Array
) -> jax.Array:
    """
    Calcula o erro de orientação

    Parameters
    ----------
    data: mjx.Data
        Estado dinâmico que atualiza a cada step.

    info: dict[str, Any]
        Dicionario de informações
    """

    # comando medido em rpy
    r_target = Rotation.from_euler("zyx", goal_orientation)
    r_measured = Rotation.from_quat(tool_orientation)

    # calcula a transformação  "erro", com base em: r_measured = r_error * r_target
    r_error = r_measured * r_target.inv()

    r_error = r_measured * r_target.inv()
    return jnp.linalg.norm(r_error.as_rotvec())


def cost_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
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

def cont_sample_beta(rng: jax.Array, alpha: jax.Array, beta: jax.Array, eps=1e-4):

    # separa o rng para amostras independentes
    rng, subkey = jax.random.split(rng)
    actions = jax.random.beta(subkey, alpha, beta)

    # Clipa as ações para ficar dentro de (0, 1)
    clipped_actions = jnp.clip(actions, eps, 1.0 - eps)

    # logprob para cada dimensão
    logprobs = jax.scipy.stats.beta.logpdf(clipped_actions, alpha, beta)
    return clipped_actions, jnp.sum(logprobs, axis=-1)

def beta_entropy(a, b):

    lnB = gammaln(a) + gammaln(b) - gammaln(a + b)
    H = (
        lnB
        - (a - 1) * digamma(a)
        - (b - 1) * digamma(b)
        + (a + b - 2) * digamma(a + b)
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


def ema(ema_success: jax.Array, current_success_rate:jax.Array, alpha=0.9):
    """ Exponential mooving average

    Parameters
    ----------
    ema_success : jax.Array
        Valor atual para a média móvel exponencial

    current_success_rate : jax.Array
        Taxa de sucesso atual

    alpha : float, optional
        Fator de suavização, default = 0.9

    Returns
    -------
    jax.Array
        Nova média móvel exponencial
    """
    return (alpha * ema_success) + ((1 - alpha) * current_success_rate)

def stdNormalize(values: jax.Array):
    mean = jnp.mean(values)
    std = jnp.std(values)
    values = (values - mean) / (std + 1e-8)
    return values




def update_assets(
    assets: Dict[str, Any],
    path: Union[str, epath.Path],
    glob: str = "*",
    recursive: bool = False,
):
    for f in epath.Path(path).glob(glob):
        if f.is_file():
            assets[f.name] = f.read_bytes()
        elif f.is_dir() and recursive:
            update_assets(assets, f, glob, recursive)



def maybe_filled_list(list)-> MaybeM:
    if isinstance(list, List) and len(list) > 0:
        return MaybeM.just(list)
    return MaybeM.nothing()


def save(data, filename="data.msgpack"):
    state_bytes = flax.serialization.to_bytes(data)

    with open(filename, "wb") as f:
        f.write(state_bytes)

    print(f"Data saved successfully to: {filename}")

def load(empty_params: T, filename="data.msgpack")->T:
    with open(filename, "rb") as f:
        state_bytes = f.read()

    # restaura os parametros a partir dos dados serializados
    return cast(T, flax.serialization.from_bytes(empty_params, state_bytes))