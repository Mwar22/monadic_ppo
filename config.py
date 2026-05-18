"""
Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------
Arquivo com as configurações
"""

import jax
import jax.numpy as jnp
from flax import struct
from dataclasses import field
from typing import Callable, Self, List


@struct.dataclass
class MujocoSimConfig:
    """
    Configurações base para o treino
    """

    ctrl_dt: float = 1.0 / 5e1  # time step para o controle (s), 50Hz
    sim_dt: float = 1.0 / 1e3  # time step para a simulação (s), 1kHz
    
    impl: str = "jax"

    @classmethod
    def init(cls,
            ctrl_freq: float = 50.0,
            sim_freq:float = 1000,
        ):
        return cls(
            1.0/ctrl_freq,
            1.0/sim_freq,
        )
    
    @property
    def dt(self) -> float:
        return self.ctrl_dt

    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return int(round(self.dt / self.sim_dt))
    
   

@struct.dataclass
class RewardConfigParameter:
    update: Callable

    @classmethod
    def const(cls, value):
        return cls(lambda _: value)

    @classmethod
    def linear_tracking(cls, start_value, end_value):
        def func(p):
            return start_value - (start_value - end_value) * p

        return cls(func)

    @classmethod
    def inv_sqrt_tracking(cls, start_value, end_value):
        def func(p):
            return start_value - (start_value - end_value) * (p**0.5)

        return cls(func)
    
    @classmethod
    def oneshot_cos(cls, max_value=1.0, pct_start=0.3, div_factor=25.0, final_div_factor=100.0, range_value=1.0):
        omega_first = jnp.pi/(pct_start*range_value)
        omega_second = jnp.pi/(range_value*(1 - pct_start))
        phi = -jnp.pi*pct_start/(1-pct_start)

        def first_half(p):
            intermediate = 1 + div_factor + (1- div_factor)*jnp.cos(omega_first*p)
            return max_value*intermediate/(2*div_factor)
        
        def second_half(p):
            intermediate = 1 + final_div_factor - (1- final_div_factor)*jnp.cos(omega_second*p + phi)
            return max_value*intermediate/(2*final_div_factor)
        
        def func(p):
            first = first_half(p)
            second = second_half(p)

            return jnp.where(p < pct_start*range_value, first, second)
        
        return cls(func)

@struct.dataclass
class RewardConfig:
    # --- Incentivo de Posição ---
    # O ganho máximo quando o erro é zero
    pos_incentive_gain = RewardConfigParameter.const(1000.0)

    # Valor que define o comportamento da recompensa combinada exponencial e linear.
    # para erros acima de xzero, tem-se penalidades (valores negativos)
    # abaixo de xzero, tem-se recompensas (valores positivos)
    # No início do treino (progress=0), xzero=0.5
    # No fim do treino (progress=1), xzero=0.01
    pos_incentive_xzero = RewardConfigParameter.linear_tracking(0.3, 0.1)

    # --- Incentivo de Orientação ---
    #rot_incentive_gain = RewardConfigParameter.const(100.0)
    #rot_incentive_sigma = RewardConfigParameter.linear_tracking(0.5, 0.05)

    # --- Sucesso e Falha ---
    success_reward = RewardConfigParameter.const(100.0)
    failure_penalty = RewardConfigParameter.const(-100.0)
    limitbreach_penalty_gain = RewardConfigParameter.const(-10.0)

    # --- Tolerância ---
    # No início do treino (progress=0), err_tol=0.8
    # No fim do treino (progress=1), err_tol=0.1
    #err_tol = RewardConfigParameter.oneshot_cos(max_value=0.4, div_factor=2, final_div_factor=40)
    err_tol = RewardConfigParameter.linear_tracking(0.3, 0.01)

    # --- Regularização ---
    torques_penalty = RewardConfigParameter.const(-1e-6)
    velocity_penalty = RewardConfigParameter.const(-1e-6)

    # cost action - penalidade por diferença entra ação atual e passada
    # penaliza delta de ações muito grandes no final
    tar_penalty_gain = RewardConfigParameter.linear_tracking(-0.001, -0.1)



@struct.dataclass
class VectorRange:
    min_values: jax.Array
    max_values: jax.Array

    def sample_normal(self, rng, progress):
        scale = jnp.maximum(0.01, progress)

        dim = self.min_values.shape[0]

        mean = (self.min_values + self.max_values) / 2
        std = (self.max_values - self.min_values) * scale

        rng, subkey = jax.random.split(rng)
        z = jax.random.normal(subkey, shape=(dim,))

        return rng, mean + std * z


@struct.dataclass
class RangeConfig:
    position: VectorRange
    position_velocities: VectorRange
    orientation: VectorRange

    @classmethod
    def init(
        cls, pos_min, pos_max, posvel_min, posvel_max, ori_min, ori_max
    ):
        return cls(
            VectorRange(pos_min, pos_max),
            VectorRange(posvel_min, posvel_max),
            VectorRange(ori_min, ori_max),
        )
