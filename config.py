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
    pos_incentive_gain = RewardConfigParameter.const(500.0)

    # 'Largura' da recompensa: se o erro for igual a sigma, a recompensa cai para ~36%
    # No início do treino (progress=0), sigma=0.5
    # No fim do treino (progress=1), sigma=0.1
    pos_incentive_sigma = RewardConfigParameter.inv_sqrt_tracking(0.8, 0.02)

    # --- Incentivo de Orientação ---
    rot_incentive_gain = RewardConfigParameter.const(200.0)
    rot_incentive_sigma = RewardConfigParameter.linear_tracking(0.5, 0.05)

    # --- Sucesso e Falha ---
    success_reward = RewardConfigParameter.const(1000.0)
    failure_penalty = RewardConfigParameter.const(-100.0)
    
    # --- Tolerância ---
    # No início do treino (progress=0), err_tol=0.8
    # No fim do treino (progress=1), err_tol=0.05
    err_tol = RewardConfigParameter.linear_tracking(0.8, 0.02)
    
    # --- Regularização ---
    torques_penalty = RewardConfigParameter.const(-1e-2)
    velocity_penalty = RewardConfigParameter.const(-1e-3)


@struct.dataclass
class VectorRange:
    num_values: int
    values: jax.Array

    @classmethod
    def init(cls, num_values: int, min_values: jax.Array, max_values: jax.Array)->Self:
        num = max(num_values -1, 1)
        values = jnp.vstack([jnp.linspace(start, stop, num=num) for start, stop in zip(min_values, max_values)])
        return cls(num_values, values)
    

    def sample_normal(self, rng, progress):
   
        scale = jnp.maximum(0.01, progress)

        mean = (self.num_values - 1) / 2
        std = mean * scale

        dim = self.values.shape[0]

        rng, subkey = jax.random.split(rng)
        z = jax.random.normal(subkey, shape=(dim,))

        indices = jnp.clip(
            jnp.round(mean + std * z),
            0,
            self.num_values - 1
        ).astype(jnp.int32)

        retval = self.values[jnp.arange(dim), indices]
        return rng, retval
    
@struct.dataclass
class RangeConfig:
    position_steps: int
    orientation_steps: int

    position: VectorRange 
    position_velocities: VectorRange 
    orientation: VectorRange

    @property
    def numberof_goals(self):
        return self.position_steps * self.orientation_steps

    @classmethod
    def init(
        cls,
        pos_min: jax.Array,
        pos_max: jax.Array,
        posvel_min: jax.Array,
        posvel_max: jax.Array,
        ori_min: jax.Array,
        ori_max: jax.Array,
        pos_steps: int,
        posvel_steps: int,
        ori_steps: int,
    )->Self:
        return cls(
            pos_steps,
            ori_steps,
            VectorRange.init(pos_steps, pos_min, pos_max),
            VectorRange.init(posvel_steps, posvel_min, posvel_max),
            VectorRange.init(ori_steps, ori_min, ori_max),
        )

    


