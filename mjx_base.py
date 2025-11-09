"""
Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------
Base class for ThorRobot
"""


import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from flax import struct
from typing import Any, Dict, Optional, Union, Sequence
from dataclasses import field

@struct.dataclass
class EnviromentConfig:
    """
    Configurações base para o treino
    """

    ctrl_dt: float = 0.02         # time step para o controle (s)
    sim_dt: float = 0.005         # time step para a simulação (s)
    episode_length: float = 500   # 5 sec (250*ctrl_dt)
    action_repeat: float = 1
    action_scale: float = 0.1
    obs_noise: float = 0.05
    impl: str = 'jax'
    nconmax: int = 24 * 8192
    njmax: int = 88

    @property
    def dt(self)->float:
       return self.ctrl_dt
    
    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return int(round(self.dt / self.sim_dt))

@struct.dataclass
class RewardConfig:
    # Recompensa por acompanhar a posição do efetuador final (distância Euclidiana)
    position_error: float = 20 #20

    # Recompensa por acompanhar a orientação do efetuador final
    orientation_error: float = 15 #15

    
    # Regularização L2 dos torques nas juntas, para evitar torques muito grandes (energia)
    torques: float = -0.0005
    
    # Penaliza mudanças bruscas na ação, incentivando controle suave
    # Também uma regularização L2: penaliza o quadrado da diferença entre ações consecutivas
    action_rate: float = -0.01

    # Encoraja não ter movimento quando as velocidades de comando são zero. Reguarização L2
    stand_still: float = -0.5
    
    # Penalidade para término antecipado do episódio (ex: falha)
    termination: float = -1.0

    # recompensa exponencialmente decrescente com o erro = exp(-error^2/sigma).
    tracking_sigma: float = 0.75

@struct.dataclass
class RangeConfig:
    goal_position: jax.Array = field(default_factory= lambda: jnp.array([
        [-0.6, 0.6, 0.1],
        [-0.6, 0.6, 0.1],
        [0, 0.6, 0.1]
    ])
    )
    
    goal_orientation: jax.Array = field(default_factory= lambda: jnp.array([
        [-2, 2, 0.1],
        [-2, 2, 0.1],
        [-2, 2, 0.1]
    ])
    )

    @property
    def x(self):
        return self.goal_position[0,:]
    
    @property
    def y(self):
        return self.goal_position[1,:]
    
    @property
    def z(self):
        return self.goal_position[2,:]
    
    @property
    def row(self):
        return self.goal_orientation[0,:]
    
    @property
    def pitch(self):
        return self.goal_orientation[1,:]
    
    @property
    def yaw(self):
        return self.goal_orientation[2,:]
    
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
    clip_range:float = 0.1  # qpos fica cravado no intervalo percentual: [qpos_clip_range, 1 - qpos_clip_range]


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

def init(
    model: mjx.Model,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
) -> mjx.Data:
  """Initialize MJX Data."""
  data = mjx.make_data(model)
  if qpos is not None:
    data = data.replace(qpos=qpos)
  if qvel is not None:
    data = data.replace(qvel=qvel)
  if ctrl is not None:
    data = data.replace(ctrl=ctrl)
  if act is not None:
    data = data.replace(act=act)
  if mocap_pos is not None:
    data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
  if mocap_quat is not None:
    data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
  data = mjx.forward(model, data)
  return data

def mjx_step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
  def single_step(data, _):
    data = data.replace(ctrl=action)
    data = mjx.step(model, data)
    return data, None

  return jax.lax.scan(single_step, data, (), n_substeps)[0]



def get_sensor_data(model: mujoco.MjModel, data: mjx.Data, sensor_name: str) -> jax.Array:
    """
    Obtem os dados de um determinado sensor, de acordo com seu nome

    Parameters
    ----------
    model: mujoco.MjModel
        Modelo do mujoco.

    data: mjx.Data
        Estado dinâmico que atualiza a cada step.

    sensor_name: str
        nome do sensor que se deseja obter os dados.

    Returns
    -------
    ret: jax.Array
        Dados obtidos do sensor.
    """
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]



###############

def dof_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
    """
    Obtem a dimensionalidade de cada junta em qvel.

    Parameters
    ----------
    joint_type: Union[int, mujoco.mjtJoint]
        Tipo da Junta. Ou é um inteiro ou é um enum do tipo mujoco.mjtJoint

    Returns
    -------
    ret: int
        Numero de graus de liberdade para qvel para dado tipo de junta.

    """

    # para obter o valor, caso seja um enum
    if isinstance(joint_type, mujoco.mjtJoint):
        joint_type = joint_type.value

    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]

def qpos_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
    """
    Obtem a dimensionalidade de cada junta em qpos.

    Parameters
    ----------
    joint_type: Union[int, mujoco.mjtJoint]
        Tipo da Junta. Ou é um inteiro ou é um enum do tipo mujoco.mjtJoint

    Returns
    -------
    ret: int
        Numero de graus de liberdade para qpos para dado tipo de junta.
    """

    # para obter o valor, caso seja um enum
    if isinstance(joint_type, mujoco.mjtJoint):
        joint_type = joint_type.value
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]

def get_qpos_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    """
    Obtem os indices com os endereços em qpos cada grau de liberdade de cada junta.

    Parameters
    ----------
    model: mujoco.MjModel
        Modelo do mujoco.

    joint_names: Sequence[str]
        Sequência de strings que compoem os nomes das juntas.

    Returns
    -------
    ret: np.ndarray
        Array com os endereços  em qpos para cada junta.
    """
    ranges = [
        np.arange(
            model.jnt_qposadr[model.joint(name).id],
            model.jnt_qposadr[model.joint(name).id] + qpos_width(model.jnt_type[model.joint(name).id])
        )
        for name in joint_names
    ]
    return np.concatenate(ranges)

def get_qvel_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    """
    Obtem os indices com os valores de velocidade para cada grau de liberdade de cada junta.

    Parameters
    ----------
    model: mujoco.MjModel
        Modelo do mujoco.

    joint_names: Sequence[str]
        Sequência de strings que compoem os nomes das juntas.

    Returns
    -------
    ret: np.ndarray
        Array com os endereços  em qvel para cada junta.
    """
    ranges = [
        np.arange(
            model.jnt_dofadr[model.joint(name).id],
            model.jnt_dofadr[model.joint(name).id] + dof_width(model.jnt_type[model.joint(name).id])
        )
        for name in joint_names
    ]
    return np.concatenate(ranges)