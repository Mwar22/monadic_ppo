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
    # Recompensa por acompanhar a posição do efetuador final (distância Euclidiana)
    position_error_penalty: float = -1.0  # 20

    # Recompensa por acompanhar a orientação do efetuador final
    orientation_error_penalty: float = -1.0  # 15

    success_reward: float = 100.0  # Large positive reward for succeeding
    failure_penalty: float = -0.25  # Large negative penalty for failing

    # Regularização L2 dos torques nas juntas, para evitar torques muito grandes (energia)
    torques_penalty: float = -0.0005

    # Penaliza mudanças bruscas na ação, incentivando controle suave
    # Também uma regularização L2: penaliza o quadrado da diferença entre ações consecutivas
    action_rate: float = -0.01

    # Encoraja não ter movimento quando as velocidades de comando são zero. Reguarização L2
    stand_still: float = 0.5

    # Penalidade para término antecipado do episódio (ex: falha)
    termination: float = -1.0

    # para incentivar o robô a sair do lugar
    tracking_incentive_gain: float = 3.0

    # recompensa exponencialmente decrescente com o erro = exp(-error^2/sigma).
    tracking_sigma: float = 0.95


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
