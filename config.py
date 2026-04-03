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
    sim_dt: float = 0.005  # time step para a simulação (s)
    episode_length: float = 500  # 5 sec (250*ctrl_dt)
    action_repeat: float = 1
    action_scale: float = 0.3
    obs_noise: float = 0.05
    impl: str = "jax"
    nconmax: int = 24 * 8192
    njmax: int = 88

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
    pos_incentive_sigma = RewardConfigParameter.inv_sqrt_tracking(0.8, 0.1)

    # --- Incentivo de Orientação ---
    rot_incentive_gain = RewardConfigParameter.const(100.0)
    rot_incentive_sigma = RewardConfigParameter.linear_tracking(0.5, 0.2)

    # --- Sucesso e Falha ---
    success_reward = RewardConfigParameter.const(1000.0)
    failure_penalty = RewardConfigParameter.const(-100.0)
    
    # --- Tolerância ---
    # No início do treino (progress=0), err_tol=0.6
    # No fim do treino (progress=1), err_tol=0.1
    err_tol = RewardConfigParameter.linear_tracking(0.3, 0.05)
    
    # --- Regularização ---
    torques_penalty = RewardConfigParameter.const(-1e-5)


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

    num_values: int = 300

    position: VectorRange = VectorRange.init(
        300,
        jnp.array([-0.6, -0.6, 1]),
        jnp.array([0.6, 0.6, 1])
    )
    position_velocities: VectorRange = VectorRange.init(
        300,
        jnp.array([0, 0, 0]),
        jnp.array([0.1, 0.1, 0.1])
    )
    orientation: VectorRange = VectorRange.init(
        300,
        jnp.array([-2, -2, -2]),
        jnp.array([2, 2, 2])
    )


    


