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

    ctrl_dt: float = 0.02  # time step para o controle (s)
    sim_dt: float = 0.001  # time step para a simulação (s), 1kHz
    action_scale: float = 0.02
    obs_noise: float = 0.05
    impl: str = "jax"
    #nconmax: int = 24 * 8192
    #njmax: int = 88

    @property
    def dt(self) -> float:
        return self.ctrl_dt

    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return int(round(self.dt / self.sim_dt))


@struct.dataclass
class RewardConfigParameter:
    update: Callable[[float], float]

    @classmethod
    def const(cls, value):
        return cls(lambda _: value)

    @classmethod
    def linear_tracking(cls, start_value, end_value):
        def func(p):
            return start_value - (start_value - end_value)*p
        
        return cls(func)
    
    @classmethod
    def inv_sqrt_tracking(cls, start_value, end_value):
        def func(p):
            return start_value - (start_value - end_value)*(p**0.5)
        
        return cls(func)
        
@struct.dataclass
class RewardConfig:
    # --- Incentivo de Posição ---
    # O ganho máximo quando o erro é zero
    pos_incentive_gain = RewardConfigParameter.const(400.0)

    # 'Largura' da recompensa: se o erro for igual a sigma, a recompensa cai para ~36%
    # No início do treino (progress=0), sigma=0.5
    # No fim do treino (progress=1), sigma=0.1
    pos_incentive_sigma = RewardConfigParameter.linear_tracking(0.8, 0.1)

    # --- Incentivo de Orientação ---
    rot_incentive_gain = RewardConfigParameter.const(100.0)
    rot_incentive_sigma = RewardConfigParameter.linear_tracking(0.5, 0.05)

    # --- Sucesso e Falha ---
    success_reward = RewardConfigParameter.const(500.0)
    failure_penalty = RewardConfigParameter.const(-100.0)
    
    # --- Tolerância ---
    # No início do treino (progress=0), err_tol=0.8
    # No fim do treino (progress=1), err_tol=0.1
    err_tol = RewardConfigParameter.linear_tracking(0.8, 0.1)
    
    # --- Regularização ---
    torques_penalty = RewardConfigParameter.const(-1e-2)
    velocity_penalty = RewardConfigParameter.const(-1e-3)


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
    numberof_goals: int
    position: VectorRange 
    position_velocities: VectorRange 
    orientation: VectorRange

    @classmethod
    def init(cls,
        numberof_goals,
        pos_min, pos_max,
        posvel_min, posvel_max,
        ori_min, ori_max
    ):
        return cls(
            numberof_goals,
            VectorRange(pos_min, pos_max),
            VectorRange(posvel_min, posvel_max),
            VectorRange(ori_min, ori_max),
        )

    


