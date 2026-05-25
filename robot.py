"""
Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------
Tarefa para o thor alcançar um alvo
"""

from __future__ import annotations
import mujoco
import jax
from mujoco import MjModel  # type: ignore
from jax import numpy as jnp
from mujoco import mjx
from etils import epath
from flax import struct
from typing import Any, Dict, Tuple, List, cast, Protocol, Self
from config import RangeConfig, RewardConfig, MujocoSimConfig
from enviroment import StateMonad
from utils import update_assets



########################################## para o pylance não reclamar #############################################
mujoco: Any
###################################################################################################################


class CanonicalSpace(struct.PyTreeNode):
    """ Permite transformar o espaço de observação dentro do intervalo [-1, 1]"""
    max_extent: jax.Array
    center: jax.Array

    @classmethod
    def init(cls, min: jax.Array, max: jax.Array)->Self:
        max_extent = jnp.max(max - min)
        center = (max + min)/2.0
        return cls(max_extent, center)

    def transform(self, value: jax.Array)->jax.Array:
        return 2*(value - self.center)/self.max_extent
    
    def inv_transform(self, cvalue:jax.Array)->jax.Array:
        return (self.max_extent/2.0)*cvalue + self.center

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

class RobotSharedData(struct.PyTreeNode):
    mj_model: MjModel
    mjx_model: mjx.Model
    world_space: CanonicalSpace
    joint_space:CanonicalSpace
    ctrl_dt: float 
    sim_dt: float
    tool_tip_id:int
    ground_id:int
    geom_ids: jax.Array
  
    @classmethod
    def init(
        cls,
        xml_path: epath.Path,
        model_path: epath.Path,
        meshes_path: epath.Path,
        world_space: CanonicalSpace,
        name: str,
        ctrl_dt: float = 1.0 / 5e1 , # time step para o controle (s), 50Hz
        sim_dt: float = 1.0 / 1e3,  # time step para a simulação (s), 1kHz
        impl: str = "jax"
    ) -> Self:
        """
        Método fábrica para o ambiente
        """

        # Obtem os assets com base nos caminhos
        assets = {}
        update_assets(assets, model_path, "*.xml")
        update_assets(assets, meshes_path)

        # configura modelos do mujoco e do mujoco mjx_env
        xml_text = xml_path.read_text(encoding="utf-8").encode("utf-8")
        mj_model = mujoco.MjModel.from_xml_string(  # type: ignore
            xml_text, assets=assets
        )

        # configura o timestep
        mj_model.opt.timestep = sim_dt

        # dimensões de altura e largura
        mj_model.vis.global_.offwidth = 3840
        mj_model.vis.global_.offheight = 2160

        # cria o modelo mjx na gpu
        mjx_model = mjx.put_model(mj_model, impl=str(impl))

        min, max = mj_model.actuator_ctrlrange.T
        joint_space = CanonicalSpace.init(min, max)

        # ids para a ponta colocada no robô
        robot_id = mj_model.body(name).id
        tool_tip_id = mj_model.site("tool_tip").id
        ground_id = mj_model.geom("ground").id

        #obtem as geometrias, para detectar colisões
        start_adr = mj_model.body_geomadr[robot_id]
        num_geoms = mj_model.body_geomnum[robot_id]
        geom_ids = jnp.array([start_adr + i for i in range(num_geoms)])

        return cls(mj_model, mjx_model, world_space, joint_space, ctrl_dt, sim_dt, tool_tip_id, ground_id, geom_ids)
    
    @property
    def dt(self) -> float:
        return self.ctrl_dt

    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return int(round(self.dt / self.sim_dt))
    
    @property
    def lowers(self):
        l, _ = self.mj_model.actuator_ctrlrange.T
        return l

    @property
    def uppers(self):
        _, u = self.mj_model.actuator_ctrlrange.T
        return u
    
    def sensor_data(self, mjx_data: mjx.Data, sensor_name: str) -> jax.Array:
        sensor_id = self.mj_model.sensor(sensor_name).id
        sensor_adr = self.mj_model.sensor_adr[sensor_id]
        sensor_dim = self.mj_model.sensor_dim[sensor_id]
        return mjx_data.sensordata[sensor_adr : sensor_adr + sensor_dim]
    
    def collision(self, mjx_data: mjx.Data):
        g1 = mjx_data.contact.geom1
        g2 = mjx_data.contact.geom2
        dist = mjx_data.contact.dist

        # só existe colisão se dist <= 0
        is_collision = dist <= 0

        # checa se algum contato envolveu o solo
        ground_hit = jnp.any(is_collision & ((g1 == self.ground_id) | (g2 == self.ground_id)))

        # Checa se auto colidiu
        g1_is_robot = jnp.isin(g1, self.geom_ids)
        g2_is_robot = jnp.isin(g2, self.geom_ids)
        self_hit = jnp.any(is_collision & g1_is_robot & g2_is_robot)

        return ground_hit, self_hit



def mujoco_step_monad(rsd: RobotSharedData, ctrl_action: jax.Array, n_substeps):
    """monad que apenas avança a física do MuJoCo usando um controle bruto."""
    
    def fn(state):
        def single_step(data, _):
            data = data.replace(ctrl=ctrl_action)
            data = mjx.step(rsd.mjx_model, data)
            return data, None
        
        mjx_data =  jax.lax.scan(single_step, state["mjx_data"], (), n_substeps)[0]
        new_state = {**state, "mjx_data": mjx_data}
        return new_state, {} # 
    
    return StateMonad(fn)

def read_sensors_monad(rsd: RobotSharedData):
    """monad que apenas extrai os números crus dos sensores da fisica."""
    def fn(state):
        pdata = {
            "tool_position": rsd.sensor_data(state["mjx_data"], "tool_position"),
            "joint_vel": state["mjx_data"].qvel(state["mjx_data"]),
            "joint_angles": state["mjx_data"].qpos(state["mjx_data"]),
        }
        return state, pdata
    return (StateMonad.pure({})
        .bind(fn)
        .map(lambda pdata: {**pdata, "tool_position": rsd.world_space.transform(pdata["tool_position"])})
        .map(lambda pdata: {**pdata, "joint_angles": rsd.joint_space.transform(pdata["joint_angles"])})
    )

def update_obs_monad(data, obs_noise=0.0):
    def func(state):
        rng, rng1 = jax.random.split(state["rng"])
        obs = data["obs"]

        obs_processed = jnp.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

        # Use jnp.where ou simplesmente multiplique pelo ruído para evitar o 'if'
        noise = jax.random.uniform(rng, obs_processed.shape, minval=-1.0, maxval=1.0)
        obs_processed = obs_processed + (obs_noise * noise)

        # Se for remover o histórico em favor de um estado markoviano mais simples:
        new_state = {**state, "rng": rng1, "obs": obs_processed}
        return new_state, {**data, "obs": obs_processed}

    return StateMonad(func)


def as_array_monad(d: Dict[str, Any]) -> StateMonad:
    """
    :: d -> StateMonad s c
    """

    def func(state):
        obs_list = [
            state["last_action"],  # (6, )
            jnp.array([d["position_error"]]),  # (1, )
            d["joint_angles"],  # (6, )
        ]
        obs_array = jnp.concatenate(obs_list)
        # (13,)

        return state, {**d, "obs": obs_array}

    return StateMonad(func)