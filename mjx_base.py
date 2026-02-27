"""
Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------
Base class for ThorRobot
"""

import jax
import mujoco
import numpy as np
from etils import epath
from mujoco import mjx
from typing import Any, Dict, Optional, Union, Sequence


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


def get_sensor_data(
    model: mujoco.MjModel, data: mjx.Data, sensor_name: str
) -> jax.Array:
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
            model.jnt_qposadr[model.joint(name).id]
            + qpos_width(model.jnt_type[model.joint(name).id]),
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
            model.jnt_dofadr[model.joint(name).id]
            + dof_width(model.jnt_type[model.joint(name).id]),
        )
        for name in joint_names
    ]
    return np.concatenate(ranges)

