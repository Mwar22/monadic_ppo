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
from typing import Callable, Tuple


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
    
    @classmethod
    def cos_tracking(cls, max_value=1.0, div_factor=10.0, range_value=1.0):
        omega = jnp.pi/range_value

        def func(p):
            intermediate = 1 + div_factor + (div_factor - 1)*jnp.cos(omega*p)
            return max_value*intermediate/(2*div_factor)
        
        return cls(func)


@struct.dataclass
class RewardConfig:
    pos_incentive_gain = RewardConfigParameter.const(0.6)

    # --- Sucesso e Falha ---
    success_reward = RewardConfigParameter.const(75.0)

    #chegou no final (progresso=1) e não atingiu sucesso
    failure_penalty = RewardConfigParameter.const(-50.0)
    limitbreach_penalty_gain = RewardConfigParameter.const(-0.05)

    # --- Tolerância ---
    err_tol = RewardConfigParameter.linear_tracking(0.25, 0.05)

    # --- Regularização ---
    #torques_penalty = RewardConfigParameter.const(-1e-6)
    velocity_penalty = RewardConfigParameter.const(-1e-4)

    # cost action - penalidade por diferença entra ação atual e passada
    # penaliza delta de ações muito grandes no final
    tar_penalty_gain = RewardConfigParameter.linear_tracking(-0.001, -0.01)

    #penalidade proporcional ao numero de passos
    # o objetivo é ajudar o robô a selecionar a rota que gaste menos passos
    # enquanto o progresso é baixo, a penalidade é pequena para o robô explorar mais
    steps_penalty = RewardConfigParameter.linear_tracking(-0.001, -0.05)




@struct.dataclass
class VectorRange:
    min_values: jax.Array
    max_values: jax.Array

    def sample_normal(self, rng: jax.Array, progress: float)-> Tuple[jax.Array, jax.Array]:
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
