# -*- coding:utf-8 -*-
###
# File:  robot.py
# Created Date: 24/05/2026 08:47:36
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 04/06/2026 11:01:01
# Modified By: Lucas de Jesus 
# -----
# Copyright (c) 2026
# 
# This file is subject to the terms and conditions defined in
# the 'LICENSE.txt' file found in the root of this source tree.
# Please read LICENSE.txt for full copyright and licensing details.
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
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
from mujoco import mjx, MjModel  # type: ignore
from jax import numpy as jnp
from etils import epath
from flax import struct, nnx
from typing import Any, Dict, Self, Union, List, Tuple, cast
from src.canonical_space import CanonicalSpace, new_cs, transform_to_cs, transform_vel_to_cs
from src.agent import Policy, Value, Agent, ResetData, StepData
from src.enviroment import MujocoEnv, mujoco_step, mujoco_reset
from .actor import Actor
from .critic import Critic


########################################## para o pylance não reclamar #############################################
mujoco: Any
###################################################################################################################


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

class ThorEnv(struct.PyTreeNode):
    _mjx_model: mjx.Model
    _world_space: CanonicalSpace
    _joint_space:CanonicalSpace
    ctrl_dt: float = struct.field(pytree_node=False)
    sim_dt: float = struct.field(pytree_node=False)
    ground_id:int
    geom_ids: jax.Array
    start_mjx_data:mjx.Data
    sensor_map: dict = struct.field(pytree_node=False)
    lowers: jax.Array
    uppers: jax.Array
   
    @classmethod
    def init(
        cls,
        xml_path: epath.Path,
        model_path: epath.Path,
        meshes_path: epath.Path,
        world_space: CanonicalSpace,
        name: str,
        sensor_names: List[str],
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
        start_mjx_data = mjx.make_data(mjx_model)

        min, max = mj_model.actuator_ctrlrange.T
        joint_space = new_cs(min, max)

        # ids para a ponta colocada no robô
        robot_id = mj_model.body(name).id
        ground_id = mj_model.geom("ground").id

        #obtem as geometrias, para detectar colisões
        start_adr = mj_model.body_geomadr[robot_id]
        num_geoms = mj_model.body_geomnum[robot_id]
        geom_ids = jnp.array([start_adr + i for i in range(num_geoms)])

        sensor_map = {}
        for name in sensor_names:
            sid = mj_model.sensor(name).id
            sensor_map[name] = (int(mj_model.sensor_adr[sid]), int(mj_model.sensor_dim[sid]))

        return cls(
            mjx_model,
            world_space,
            joint_space,
            ctrl_dt,
            sim_dt,
            ground_id,
            geom_ids,
            start_mjx_data,
            sensor_map,
            min,
            max,
        )
    
    ##################################### Essential methods to be a MujocoEnv ######################################
    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return int(round(self.ctrl_dt / self.sim_dt))
    
    @property
    def mjx_model(self)->mjx.Model:
        return self._mjx_model
    
    @property
    def world_space(self)->CanonicalSpace:
        return self._world_space
    @property
    def joint_space(self)->CanonicalSpace:
        return self._joint_space
    
    def failed(self, mjx_data: mjx.Data):
        g1 = mjx_data._impl.contact.geom1   # type: ignore[attr-defined]
        g2 = mjx_data._impl.contact.geom2   # type: ignore[attr-defined]
        dist = mjx_data._impl.contact.dist  # type: ignore[attr-defined]

        # só existe colisão se dist <= 0
        is_collision = dist <= 0

        # checa se algum contato envolveu o solo
        ground_hit = jnp.any(is_collision & ((g1 == self.ground_id) | (g2 == self.ground_id)))

        # Checa se auto colidiu
        g1_is_robot = jnp.isin(g1, self.geom_ids)
        g2_is_robot = jnp.isin(g2, self.geom_ids)
        self_hit = jnp.any(is_collision & g1_is_robot & g2_is_robot)

        return ground_hit|self_hit
    
    def sensor_data(self, sensor_name: str, mjx_data: mjx.Data) -> jax.Array:
        adr, dim = self.sensor_map[sensor_name]
        return mjx_data.sensordata[adr : adr + dim]
    
    @property
    def def_qpos(self)->jax.Array:
        return self.start_mjx_data.qpos
    
    @property
    def def_qvel(self)->jax.Array:
        return self.start_mjx_data.qvel
    
    ################################################################################################################
    @property
    def dt(self) -> float:
        return self.ctrl_dt
#####################################################################################################################
#####################################################################################################################

class ThorAgent(Agent):

    def __init__(
        self,
        env: MujocoEnv,
        rngs,
        max_step_rads = 0.05
        
    ):
        #mjx_data temporário
        mjx_data = mjx.make_data(env.mjx_model)
        policy_obs, action_obs = ThorAgent.compose_obs(env, mjx_data)

        self._policy = Actor(policy_obs.shape[0], mjx_data.ctrl.shape[0], rngs)
        self._value = Critic(action_obs.shape[0], rngs)
        self.max_step_rads = max_step_rads
    
    @property
    def policy(self)->Policy:
        return self._policy

    @property
    def value(self)-> Value:
        return self._value
    

    @staticmethod
    def compose_obs(env: MujocoEnv, mjx_data:mjx.Data)->Tuple[jax.Array, jax.Array]:
        cs_tool_pos = transform_to_cs(env.world_space, env.sensor_data("tool_position", mjx_data,))
        cs_qpos = transform_to_cs(env.joint_space, mjx_data.qpos)
        cs_qvel = transform_vel_to_cs(env.joint_space, mjx_data.qvel)

        policy_obs = jnp.concat([cs_qpos, cs_qvel])
        value_obs = jnp.concat([cs_tool_pos, policy_obs])
        return policy_obs, value_obs
    

    def reset(self, env: MujocoEnv,  rngs: nnx.Rngs, mjx_data:mjx.Data)->Tuple[ResetData, mjx.Data]:
        
        mjx_data = mujoco_reset(env, env.def_qpos, mjx_data)

        # coleta observações para o novo mjx_data
        policy_obs, value_obs = ThorAgent.compose_obs(env, mjx_data)
       
        #obtem uma ação com base na observação para a política
        action = self.policy.sample(policy_obs, rngs)

        #calcula logprob, entropia e value
        logprob, entropy = self.policy.evaluate_actions(policy_obs, action)
        value = self.value(value_obs)

        return ResetData(action, logprob, value, entropy), mjx_data
    
    def step(self, env: MujocoEnv,  action: jax.Array, target: jax.Array, mjx_data:mjx.Data,)->Tuple[StepData, mjx.Data]:
        #avança a física de acordo com a ação 
        delta = (2*action - 1) * self.max_step_rads

        mjx_data = mujoco_step(env, mjx_data.ctrl + delta, mjx_data)
    
        #calcula o erro de posição
        cs_tool_pos = transform_to_cs(env.world_space, env.sensor_data("tool_position", mjx_data))
        error = cast(jax.Array, jnp.linalg.norm(target - cs_tool_pos, ord=2))

        # sucesso se o erro for menor que uma dada tolerância
        success = error < 1e-2

        #falha se auto-colidiu ou colidiu com o solo
        failed = env.failed(mjx_data)

        #a coleta terminou se o robô atingiu o alvo ou se auto-colidiu ou colidiu com o solo
        done = failed | success

        reward = -error  + 100*success -100*failed
        return StepData(reward, done, {"error": error}), mjx_data