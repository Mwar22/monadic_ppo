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


@struct.dataclass
class EnviromentConfig:
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
class RewardConfig:
    # --- Incentivo de Posição ---
    # O ganho máximo quando o erro é zero
    pos_incentive_gain: float = 3.0   

    # 'Largura' da recompensa: se o erro for igual a sigma, a recompensa cai para ~36%
    pos_incentive_sigma: float = 0.2  

    # --- Incentivo de Orientação ---
    rot_incentive_gain: float = 1.5
    rot_incentive_sigma: float = 0.2

    # --- Sucesso e Falha ---
    success_reward: float = 500.0
    failure_penalty: float = -100.0
    
    # --- Tolerância ---
    start_err_tol: float = 0.4
    min_err_tol: float = 0.03
    
    # --- Regularização ---
    torques_penalty: float = -0.0001


@struct.dataclass
class RangeConfig:
    goal_position: jax.Array = field(
        default_factory=lambda: jnp.array(
            [[-0.6, 0.6, 0.1], [-0.6, 0.6, 0.1], [0, 0.6, 0.1]]
        )
    )

    goal_orientation: jax.Array = field(
        default_factory=lambda: jnp.array([[-2, 2, 0.1], [-2, 2, 0.1], [-2, 2, 0.1]])
    )

    @property
    def x(self):
        return self.goal_position[0, :]

    @property
    def y(self):
        return self.goal_position[1, :]

    @property
    def z(self):
        return self.goal_position[2, :]

    @property
    def row(self):
        return self.goal_orientation[0, :]

    @property
    def pitch(self):
        return self.goal_orientation[1, :]

    @property
    def yaw(self):
        return self.goal_orientation[2, :]


@struct.dataclass
class Target:
    position: jax.Array
    orientation: jax.Array

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    @property
    def row(self):
        return self.orientation[0]

    @property
    def pitch(self):
        return self.orientation[1]

    @property
    def yaw(self):
        return self.orientation[2]


@struct.dataclass
class ResetConfig:
    rnd_range: float = 0.1  # fator para o range percentual de qpos. Em termos de intervalo percentual: [qpos_rnd_range, 1 - qpos_rnd_range]
    clip_range: float = 0.1  # qpos fica cravado no intervalo percentual: [qpos_clip_range, 1 - qpos_clip_range]
