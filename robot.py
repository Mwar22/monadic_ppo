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
import flax.linen as nn
from mujoco import MjModel  # type: ignore
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx
from etils import epath
from flax import struct
from typing import Any, Dict, Tuple, List, cast, Callable, Sequence, Protocol, Self
from config import RangeConfig, RewardConfig, MujocoSimConfig
from enviroment import StateMonad
from utils import l1_l2_reward, exp_scale_reward, conv2jax_quat, cont_sample_beta, cost_action_rate, update_assets, maybe_filled_list
from typing import TYPE_CHECKING, runtime_checkable
from monads import MaybeM, ListM, ReaderWriterM

if TYPE_CHECKING:
    from dataclassutils import NetworksSettings, NetworkParameters, RunningParameters, RunningAvg

########################################## para o pylance não reclamar #############################################
mujoco: Any

class SamplingFunction(Protocol):
    """Define o contrato para funções de recompensa em RL."""
    def __call__(self, rng: jax.Array, progress: float, shape: Tuple[int])->float:
        ...
####################################################################################################################

@struct.dataclass
class Actuators:
    ids: jax.Array
    names: List[str]
    lowers: jax.Array
    uppers: jax.Array

    @classmethod
    def init(cls, mj_model, joint_ids: jax.Array) -> MaybeM[Self]:
        """ Obtem uma lista com um dicionario para cada  atuador de um dado corpo, contendo o id e o nome """
        actuator_ids, actuator_names = Actuators._from_joint_ids(mj_model, joint_ids)

        ids = actuator_ids.value
        names = actuator_names.value

        if ids is None or names is None:
            return MaybeM.nothing()

        lowers = mj_model.actuator_ctrlrange[:, 0]
        uppers = mj_model.actuator_ctrlrange[:, 1]
        return MaybeM.just(cls(jnp.array(ids), names, lowers, uppers))
    
    @classmethod
    def _from_joint_ids(cls, mj_model, joint_ids: jax.Array)->tuple[MaybeM[list[int]], MaybeM[list[str]]]:
        """ Obtem uma lista com um dicionario para cada junta de um dado corpo, contendo o id e o nome """

        """ Obtem uma lista com um dicionario para cada  atuador de um dado corpo, contendo o id e o nome """

        
        actuator_ids = []
        actuator_names = []

        for act_id in range(mj_model.nu):
            # trntype tells us what this actuator is attached to 
            # (e.g., mjTRN_JOINT is the standard for motors/servos)
            target_type = mj_model.actuator_trntype[act_id]
            target_id = mj_model.actuator_trnid[act_id, 0]
            
            different_type = target_type != mujoco.mjtTrn.mjTRN_JOINT
            in_ids = jnp.isin(target_id, joint_ids)

            if different_type or not in_ids:
                return MaybeM.nothing(), MaybeM.nothing()

            actuator_ids.append(act_id)
            actuator_names.append(mj_model.actuator(act_id).name)

        return MaybeM.just(actuator_ids), MaybeM.just(actuator_names)
    
    @property
    def number_of(self):
        return len(self.ids)
    
    def on_range_by_id(self, actuator_values: jax.Array):
        def on_range_single(carry, _):
            i, on_range = carry

            val = actuator_values[i]
            id = self.ids[i]

            on_range &= (val < self.lowers[id]) | (val > self.uppers[id])
            
            return (i+1, on_range), _
        
        (_, on_range), _ = jax.lax.scan(on_range_single, (0, True), length=self.ids.shape[0])
        return on_range
        
    
@struct.dataclass
class Joints:
 
    mj_model: MjModel
    ids:  jax.Array
    names:  list[str]
    qpos_adr_list: List[jax.Array]
    qvel_adr_list: List[jax.Array]

    @classmethod
    def init(cls, mj_model, body_name: str) -> MaybeM[Self]:
      
        joint_ids, joint_names = Joints._from_body(mj_model, body_name)

        ids = joint_ids.value
        names = joint_names.value

        if ids is None or names is None:
            return MaybeM.nothing()
        
        
        qposadr = []
        qveladr = []
        for id in ids:
            qposadr.append(
                jnp.arange(
                    mj_model.jnt_qposadr[id],
                    mj_model.jnt_qposadr[id] + Joints.joint_infoby_typecode(mj_model.jnt_type[id])["qpos_width"]
                )
            )
            qveladr.append(
                jnp.arange(
                    mj_model.jnt_dofadr[id],
                    mj_model.jnt_dofadr[id] + Joints.joint_infoby_typecode(mj_model.jnt_type[id])["dof_width"]
                )
            )
        

        print(f"names: {names}")
        print(f"ids: {ids}")
        print(f"qposadr: {qposadr}")
        print(f"qveladr: {qveladr}")
        
        return MaybeM.just(cls(mj_model, jnp.array(ids), names, qposadr, qveladr))

    @classmethod
    def _get_subtree_bodies(cls, mj_model, root_id)->List[int]:
        children = []

        def dfs(b):
            for i in range(mj_model.nbody):
                if mj_model.body_parentid[i] == b:
                    children.append(i)
                    dfs(i)
        dfs(root_id)
        return children

    @classmethod
    def _from_body(cls, mj_model, body_name: str)->tuple[MaybeM[list[int]], MaybeM[list[str]]]:
        """ Obtem uma lista com um dicionario para cada junta de um dado corpo, contendo o id e o nome """

        def get_root_id(name: str) -> MaybeM[int]:
            try:
                return MaybeM.just(mj_model.body(name).id)
            except KeyError:
                return MaybeM.nothing()
            
        def get_joint_ids(root_id: int) -> MaybeM[List[int]]:
            try:
                bodies = Joints._get_subtree_bodies(mj_model, root_id)

                joint_ids = []
                for b in bodies:
                    jnt_start = mj_model.body_jntadr[b]
                    jnt_count = mj_model.body_jntnum[b]

                    if jnt_start >= 0:
                        joint_ids.extend(range(jnt_start, jnt_start+jnt_count))

                return MaybeM.just(joint_ids)
            
            except KeyError:
                return MaybeM.nothing()
            
       
        def extract_names(ids: List[int]):
            return [cast(str, mj_model.joint(id).name) for id in ids]


        joint_ids = get_root_id(body_name).bind(get_joint_ids)
        names = joint_ids.map(extract_names)

        return joint_ids, names
        
    
    @classmethod
    def joint_infoby_typecode(cls, tp: int):
        _map = {
            0: {"type": "free", "dof_width": 6, "qpos_width": 7}, # 3 valores de posição + 4 para rotação (quaternion)
            1: {"type": "ball", "dof_width": 3, "qpos_width": 4}, # 4 valores apenas rotação (quaternion)
            2: {"type": "slide", "dof_width": 1, "qpos_width": 1}, # 1 grau de liberdade. 1 valor
            3: {"type": "hinge", "dof_width": 1, "qpos_width": 1}, # 1 grau de liberdade. 1 valor
        }
        return _map[tp]
    
    @property
    def lowers(self):
        l, _ = self.mj_model.actuator_ctrlrange.T
        return l

    @property
    def uppers(self):
        _, u= self.mj_model.actuator_ctrlrange.T
        return u
    
    def coordinates_collided(self, joint_coordinates: List[jax.Array])->MaybeM[bool]:
        # aplica o modelo para a CPU e checa na física
        temp_data= mujoco.MjData(self.mj_model)

        def set_joint_coordinate(idx, _):
            temp_data.qpos[self.qpos_adr_list[idx]] = joint_coordinates[idx]
            return idx+1, _

        jax.lax.scan(set_joint_coordinate, (0, ), length=len(joint_coordinates))
        
        mujoco.mj_kinematics(self.mj_model, temp_data) # Calcula as posições
        mujoco.mj_collision(self.mj_model, temp_data)   # Checa colisões
    
        # ncon == 0 means no collision detected
        return temp_data.ncon == 0

@struct.dataclass
class Geoms:
    mj_model: MjModel
    ids: jax.Array
    names: List[str]

    @classmethod
    def init(cls, mj_model, body_name:str) -> MaybeM[Self]:
        geom_ids, geom_names = Geoms._from_body(mj_model, body_name)
        
        ids = geom_ids.value
        names = geom_names.value

        if ids is None or names is None:
            return MaybeM.nothing()

        return MaybeM.just(cls(mj_model, jnp.array(ids), names))
    

    @classmethod
    def _from_body(cls, mj_model, body_name: str)->tuple[MaybeM[list[int]], MaybeM[list[str]]]:
        """ Obtem uma lista com um dicionario para cada geometria de um dado corpo, contendo o id e o nome """

        def get_body_id(name: str) -> MaybeM[int]:
            try:
                return MaybeM.just(mj_model.body(name).id)
            except KeyError:
                return MaybeM.nothing()
            
        def extract_ids(body_id: int):
            start_adr = mj_model.body_geomadr[body_id]
            num_geoms = mj_model.body_geomnum[body_id]
            return [start_adr + i for i in range(num_geoms)]
        
        def extract_names(ids: List[int]):
            return [cast(str, mj_model.joint(idx).name) for idx in ids]


        ids = get_body_id(body_name).map(extract_ids)
        names = ids.map(extract_names)

        return ids, names
    
@struct.dataclass
class SafeActionSpace:
    pos_buffer: jax.Array   #buffer de posição de juntas, (length, )
    actuators: Actuators

    @classmethod
    def init(
        cls,
        length, 
        actuators: Actuators
    )->Self:
        
        return cls(
            jnp.zeros((length, actuators.number_of)),
            actuators
        )

    def sample(self, rng, progress, sampling_fn: SamplingFunction):
        """ Retorna uma posição/orientação que seja alcançável pelo robô,
            não havendo colisões consigo mesmo ou com o solo.

        Parameters
        ----------
        rng : _type_
            _description_
        progress : _type_
            _description_
        """
        pass

@struct.dataclass
class RobotSharedData:
    mj_model: MjModel
    mjx_model: mjx.Model
    joints: Joints
    actuators: Actuators
    geoms: Geoms
    tool_tip_id: int
    tool_base_id: int
    ground_id: int
    enviroment_config: MujocoSimConfig
    reward_config: RewardConfig
    range_config: RangeConfig

    @classmethod
    def init(
        cls,
        xml_path: epath.Path,
        model_path: epath.Path,
        meshes_path: epath.Path,
        enviroment_config: MujocoSimConfig,
        reward_config: RewardConfig,
        range_config: RangeConfig,
        robot_name: str,
    )->MaybeM[RobotSharedData]:
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
        mj_model.opt.timestep = enviroment_config.sim_dt

        # dimensões de altura e largura
        mj_model.vis.global_.offwidth = 3840
        mj_model.vis.global_.offheight = 2160

        #cria o modelo mjx na gpu
        mjx_model = mjx.put_model(mj_model, impl=str(enviroment_config.impl))

        m_joints = Joints.init(mj_model, robot_name)
        if m_joints.value is None:
            jax.debug.print("Could not find joints")
            return MaybeM.nothing()

        m_geoms = Geoms.init(mj_model, robot_name)
        if m_geoms.value is None:
            jax.debug.print("Could not find geoms")
            return MaybeM.nothing()

        m_actuators = Actuators.init(mj_model, m_joints.value.ids)
        if m_actuators.value is None:
            jax.debug.print("Could not find actuators")
            return MaybeM.nothing()

        # ids para a ponta colocada no robô
        tool_tip_id = mj_model.site("tool_tip").id
        tool_base_id = mj_model.site("tool_base").id
        ground_id = mj_model.geom("ground").id

        return MaybeM.just(cls(
            mj_model,
            mjx_model,
            m_joints.value,
            m_actuators.value,
            m_geoms.value,
            tool_tip_id,
            tool_base_id,
            ground_id,
            enviroment_config,
            reward_config,
            range_config,
        ))
    
    ####################################### Metodos da classe ################################################
   
    


    ################################## Propriedades (metodos) ###########################################
    @property
    def lowers(self):
        l, _ = self.mj_model.actuator_ctrlrange.T
        return l

    @property
    def uppers(self):
        _, u= self.mj_model.actuator_ctrlrange.T
        return u

    ################################### Metodos de interação externos ##################################
    def sensor_data(
        self, mjx_data: mjx.Data, sensor_name: str
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
        sensor_id = self.mj_model.sensor(sensor_name).id
        sensor_adr = self.mj_model.sensor_adr[sensor_id]
        sensor_dim = self.mj_model.sensor_dim[sensor_id]
        return mjx_data.sensordata[sensor_adr : sensor_adr + sensor_dim]


    def detect_collisions(self, mjx_data: mjx.Data):
        # mjx_data.contact.geom1 and geom2 are arrays of IDs
        g1 = mjx_data.contact.geom1
        g2 = mjx_data.contact.geom2
        dist = mjx_data.contact.dist
        
        # só existe colisão se dist <= 0
        is_collision = dist <= 0

        # checa se algum contato envolveu o solo
        hits_ground = is_collision & ((g1 == self.ground_id) | (g2 == self.ground_id))
        
        # Checa se auto colidiu
        g1_is_robot = jnp.isin(g1, self.geoms.ids)
        g2_is_robot = jnp.isin(g2, self.geoms.ids)
        is_self_hit = is_collision & g1_is_robot & g2_is_robot
        
        return hits_ground, is_self_hit

    
    def qpos(self, mjx_data: mjx.Data):
        return mjx_data.qpos
    
    def qvel(self, mjx_data: mjx.Data):
        return mjx_data.qvel

    def qfrc(self, mjx_data: mjx.Data):
        return mjx_data.qfrc_actuator
    
    def mjx_step(self, 
        data: mjx.Data,
        action: jax.Array,
    ) -> mjx.Data:
        def single_step(data, _):
            data = data.replace(ctrl=action)
            data = mjx.step(self.mjx_model, data)
            return data, None

        return jax.lax.scan(single_step, data, (), self.enviroment_config.n_substeps)[0]
    
    
    
##############################################################################################################





def position_error(goal_position: jax.Array, tool_position: jax.Array) -> jax.Array:
    """
    Calcula o erro de posição.

    Parameters
    ----------
    data: mjx.Data
        Estado dinâmico que atualiza a cada step.

    info: dict[str, Any]
        Dicionario de informações
    """
    return jnp.linalg.norm(goal_position - tool_position, ord=2)


def orientation_error(
    goal_orientation: jax.Array, tool_orientation: jax.Array
) -> jax.Array:
    """
    Calcula o erro de orientação

    Parameters
    ----------
    data: mjx.Data
        Estado dinâmico que atualiza a cada step.

    info: dict[str, Any]
        Dicionario de informações
    """

    # comando medido em rpy
    r_target = Rotation.from_euler("zyx", goal_orientation)
    r_measured = Rotation.from_quat(tool_orientation)

    # calcula a transformação  "erro", com base em: r_measured = r_error * r_target
    r_error = r_measured * r_target.inv()

    r_error = r_measured * r_target.inv()
    return jnp.linalg.norm(r_error.as_rotvec())


def check_done(
    rsd: RobotSharedData, joint_angles: jax.Array, position_error, orientation_error
):
    # se tiver alcançado os objetivos de posição e orientação
    done = (position_error < 0.0001) & (orientation_error < 0.0001)

    # termina se os limites de junta forem ultrapassados
    done |= jnp.any(joint_angles < rsd.lowers)
    done |= jnp.any(joint_angles > rsd.uppers)

    return done


#################################################################################################################
def sample_config_coordinates_curriculum(
    rsd: RobotSharedData, rng, config_name: str, progress: float, start_pos:jax.Array
):
    """
    Gera alvos que 'expandem' conforme o treino progride.
    progress: valor de 0.0 a 1.0 (ex: update_atual / total_updates)
    """
    config_value = getattr(rsd.range_config, config_name)
    
    
    # 2. Definimos o 'tamanho' do mundo atual baseado no progresso
    # No início (progress=0), o mundo tem 10% do tamanho. No fim, 100%.
    scale = jnp.maximum(0.1, progress) 
    
    low = start_pos - (start_pos - config_value[:, 0]) * scale
    high = start_pos + (config_value[:, 1] - start_pos) * scale

    samples = jax.random.uniform(rng, shape=(3,))
    scaled_coord = low + (high - low) * samples

    return {config_name + "_coordinates": scaled_coord}

def sample_config_coordinates(
    rsd:RobotSharedData, rng, config_name: str
):
    """
    Obtem comandos aleatórios para as posições
    """
    config_value = getattr(rsd.range_config, config_name)

    # faz um sampleamento
    samples = jax.random.uniform(rng, shape=(3,))

    # ajusta a escala para que fique dentro da faixa [min, max]
    scaled_coord = (
        config_value[:, 0] + (config_value[:, 1] - config_value[:, 0]) * samples
    )

    str_id = config_name + "_coordinates"
   
    return {str_id: scaled_coord}




def sample_config_velocities(
    rsd:RobotSharedData, rng, config_name: str
):
    """
    Obtem comandos aleatórios para as velocidades
    """
    config_value = getattr(rsd.range_config, config_name)

    # faz um sampleamento
    samples = jax.random.uniform(rng, shape=(3,))

    # ajusta a escala para que fique dentro da faixa [0, max]
    scaled_coord = config_value[:, 1] * samples

    str_id = config_name + "_velocities"
    return {str_id: scaled_coord}


##################################################################################################################

def normalize_obs(data, obs_stats:RunningAvg):
    def func(state):
        std = jnp.sqrt(obs_stats.var + 1e-8)
        norm_obs = (data["obs"] - obs_stats.mean) / std

        new_state = {**state,"obs": norm_obs}
        return new_state, {**data, "obs": norm_obs}
    return StateMonad(func)

def update_obs(data, obs_noise=0.0):
    def func(state):
        rng, rng1 = jax.random.split(state["rng"])

        # clipa a observação
        #normaliza a observação
        #std = jnp.sqrt(obs_stats.var + 1e-8)
        #norm_obs = (data["obs_array"] - obs_stats.mean) / std
        obs = data["obs_array"]

        #proteje a rede contra explosões no inicio
        obs_processed = jnp.clip(obs, -5.0, 5.0)

        # Adiciona um ruido adicional.
        # Este 'if' funciona com JIT contanto que obs_noise seja um valor estático.
        if obs_noise >= 0.0:
            noise = obs_noise * jax.random.uniform(
                rng, obs_processed.shape, minval=-1.0, maxval=1.0
            )
            obs_processed += noise

        # Adiciona a nova observação no buffer, deslocando as outras observações e descartando a mais antiga
        obs_history = (
            jnp.roll(
                state["obs"], obs_processed.size
            )  # desloca todo o array para a direita, obs_processd.size de distancia (circular)
            .at[: obs_processed.size]
            .set(obs_processed)  # adiciona os novos dados de observação
        )

        mean_mask = jnp.zeros_like(obs)
        
        new_state = {**state, "rng": rng1, "obs": obs_history}
        return new_state, {**data, "obs": obs_history}

    return StateMonad(func)


def concat_obs_as_array(d: Dict[str, Any]) -> StateMonad:
    """
    :: d -> StateMonad s c
    """

    def func(state):
        # Manually list keys to ensure order and handle scalars
        obs_list = [
            state["goal"]["goal_position_coordinates"],  # (3,)
            state["goal"]["goal_orientation_coordinates"],  # (3,)
            state["goal"]["goal_position_velocities"], # (3,)
            d["tool_position"],  # (3,)
            d["orientation"],  # (4,)
            d["torques"],  # (6,)
            d["joint_angles"],  # (6,)
            d["joint_vel"],  # (6,)
        ]
        obs_array = jnp.concatenate(obs_list)
        # 4 * (3,)  +  3 * (6, ) + (4,)= 34)

        return state, {**d, "obs_array": obs_array}

    return StateMonad(func)


def success_count(pdata):
    def func(state):
        # Transforma True em 1 e False em 0 e soma
        count = state["success_count"] + jnp.array(pdata["success"], dtype=jnp.int32)
        return {**state, "success_count": count}, pdata
    return StateMonad(func)

###################################################################################################################


def get_goal(range_config: RangeConfig, progress, rng):

    rng, position = range_config.position.sample_normal(rng, progress)
    rng, position_velocities = range_config.position_velocities.sample_normal(rng, progress)
    rng, orientation = range_config.orientation.sample_normal(rng, progress)
   

    goals = {
        "goal_position_coordinates": position,
        "goal_position_velocities": position_velocities,
        "goal_orientation_coordinates": orientation
    }
    return rng, goals

def sample_random_pose(rsd, rng):
    rng, subkey = jax.random.split(rng)
    # Generates a value between lower and upper limits for each joint
    random_q = jax.random.uniform(
        subkey, 
        shape=(len(rsd.joint_ids),), 
        minval=rsd.lowers, 
        maxval=rsd.uppers
    )
    return rng, random_q

def training_frame_setup(rsd: RobotSharedData, progress, rng):

    rng, position = rsd.range_config.position.sample_normal(rng, progress)
    rng, position_velocities = rsd.range_config.position_velocities.sample_normal(rng, progress)
    rng, orientation = rsd.range_config.orientation.sample_normal(rng, progress)
   

    goals = {
        "goal_position_coordinates": position,
        "goal_position_velocities": position_velocities,
        "goal_orientation_coordinates": orientation
    }

    #randomiza as posições de junta, de acordo com os limites
    rng, subkey = jax.random.split(rng)
    alpha = jax.random.uniform(subkey)
    rnd_start_joint_angles = rsd.lowers + (rsd.uppers -rsd.lowers) * alpha
    return rng, goals



def debug(pdata, name):
    def func(state):
        jax.debug.print("{} = {}", name, pdata[name])
        return state, pdata
    return StateMonad(func)


def obs_pipeline(rsd: RobotSharedData, obs_stats: RunningAvg, env: StateMonad):
    return (
        env.bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {**pdata, "tool_position": rsd.sensor_data(state["mjx_data"], "tool_position")},
                )
            )
        )
        .bind(lambda pdata: StateMonad(
            lambda state:(
                state,
                {
                    **pdata,
                    "position_error": position_error(
                        state["goal"]["goal_position_coordinates"], pdata["tool_position"]
                    ),
                }
            )
        ))

        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {**pdata, "orientation": rsd.sensor_data(state["mjx_data"], "tool_orientation")},
                )
            )
        )
        .map(
            lambda pdata:{
                **pdata, "orientation": conv2jax_quat(pdata["orientation"])
            }
        )
        .bind(lambda pdata: StateMonad(
            lambda state:(
                state,
                {
                    **pdata,
                    "orientation_error": orientation_error(
                        state["goal"]["goal_orientation_coordinates"], pdata["orientation"]
                    ),
                }
            )
        ))

        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {**pdata, "torques": rsd.qfrc(state["mjx_data"])},
                )
            )
        )
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {
                        **pdata,
                        "joint_angles": rsd.qpos(state["mjx_data"]),
                    },
                )
            )
        )
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {
                        **pdata,
                        "joint_vel": rsd.qvel(state["mjx_data"]),
                    },
                )
            )
        )
        .map(
            lambda pdata: {
                **pdata,
                "pose_dist": pdata["joint_angles"], ###CHANGE SO IT COUNTS RELATIVE!
            }
        )
        .bind(lambda pdata: concat_obs_as_array(pdata))
        .map(lambda pdata: {**pdata, "obs": pdata["obs_array"]})
        .bind(lambda pdata: normalize_obs(pdata, obs_stats))
        #.bind(
        #    lambda pdata: update_obs(pdata, obs_stats, rsd.enviroment_config.obs_noise)
        #)
        #.bind(lambda pdata: debug(pdata, "position_error"))
        #.bind(lambda pdata: debug(pdata, "orientation_error"))
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    {**state, "err": jnp.minimum(state["err"], pdata["position_error"])},
                    pdata
                )
            )
        )
    )


def reward_pipeline(progress, rsd: RobotSharedData,  env: StateMonad):
    reward_config = rsd.reward_config
    return (
        # Penalidade (custo) por erro de posição
        env.map(
            lambda pdata: {
                **pdata,
                "reward": (
                    # Incentivo de Posição
                    exp_scale_reward(
                        reward_config.pos_incentive_gain.update(progress),
                        reward_config.pos_incentive_sigma.update(progress),
                        pdata["position_error"]
                    ) + 

                    # Incentivo de Orientação
                    exp_scale_reward(
                        reward_config.rot_incentive_gain.update(progress),
                        reward_config.rot_incentive_sigma.update(progress),
                        pdata["orientation_error"]
                    )
                ),
            }
        )

       
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {
                        **pdata,
                        "reward": (
                            # penalidade por ações muito grandes
                            pdata["reward"] - 0.01 * cost_action_rate(pdata["action"], state["last_action"]) 

                            # Penalidade de torque para evitar movimentos espasmódicos
                            + jnp.sum(jnp.square(pdata["torques"])) * reward_config.torques_penalty.update(progress)

                            # Penalidade de velocidade para evitar movimentos espasmódicos
                            + jnp.sum(jnp.square(pdata["joint_vel"])) * reward_config.velocity_penalty.update(progress)
                        ),
                    }
                )
            )
        )
        # Tolerância de Erro Linear
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {
                        **pdata,
                        "err_tol": reward_config.err_tol.update(progress), 
                    }
                )
            )
        )
        # Verificação de Done e Sucesso
        .map(
            lambda pdata: {
                **pdata,
                "success": (pdata["position_error"] < pdata["err_tol"]) & 
                           (pdata["orientation_error"] < pdata["err_tol"]),
                "failure": jnp.any(pdata["joint_angles"] < rsd.lowers) | 
                           jnp.any(pdata["joint_angles"] > rsd.uppers),
            }
        )
        .bind(success_count)
        # Aplicação das Recompensas de Término
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {
                        **pdata,
                        "done": pdata["success"] | pdata["failure"],

                        # Bônus de Sucesso
                        "reward": pdata["reward"]
                        + pdata["success"] * reward_config.success_reward.update(progress)
                        + pdata["failure"] * reward_config.failure_penalty.update(progress),
                    },
                )
            )
        )
        .map(lambda pdata: {**pdata, "reward": jnp.clip(pdata["reward"], -5000.0, 5000.0)})
    )


####################################################################################################################
def get_action(network_settings: NetworksSettings, network_parameters: NetworkParameters):
    def fn(state):
        last_obs = state["obs"]
        rng1, rng2 = jax.random.split(state["rng"])

        output = network_settings.actor.apply(network_parameters.actor, last_obs)
        output = cast(jax.Array, output)
        action_value, logprob = cont_sample_beta(output, rng1)

        
        new_state = {**state, "rng": rng2}
        return new_state, {"action": action_value, "logprob": logprob}
    return StateMonad(fn)

def get_motor_targets(rsd: RobotSharedData, pdata):
    def fn(state):
        # escala ação para de [0, 1] para [-1, 1]
        action_value_action = jnp.clip(2.0 * pdata["action"]- 1.0, -1, 1)

        # configura novos alvos para os motores, de acordo com a ação
        current = rsd.qpos(state["mjx_data"])
        targets = current + action_value_action * rsd.enviroment_config.action_scale

        # para evitar que os limites de junta do robô sejam desrespeitados
        return state, {**pdata, "motor_targets":jnp.clip(targets, rsd.lowers, rsd.uppers)}
    return StateMonad(fn)

def mujoco_step(rsd: RobotSharedData, pdata):
    def fn(state):
        mjx_data = rsd.mjx_step(
            state["mjx_data"],
            pdata["motor_targets"],
        )
        state = {**state, "mjx_data": mjx_data}
        return state , pdata
    return StateMonad(fn)

def create_step(network_settings: NetworksSettings, network_parameters: NetworkParameters, robot_shared_data: RobotSharedData):
    """
    state.keys() = ["rng", "step", "goal", "obs_history", "action", "mjx_data"]
    """
   
    def shape_return(pdata):
        def fn(state):
            state = {**state, "step": state["step"] + 1}
            data = {
                "obs": pdata["obs"],
                "action": pdata["action"],
                "reward": pdata["reward"],
                "logprob": pdata["logprob"],
                "done": pdata["done"]
            }
            return state, data
        return StateMonad(fn)
    
    def last_action_update(pdata):
        def fn(state):
            state = {**state, "last_action": pdata["action"]}
            return state, pdata
        
        return StateMonad(fn)
    
    def step_fn(progress, state, runpar: RunningParameters):

        # obtem uma ação pela observação anterior
        pl = (get_action(network_settings, network_parameters)
            .bind(lambda pdata: get_motor_targets(robot_shared_data, pdata))    #obtem para os motores segundo a ação
            .bind(lambda pdata: mujoco_step(robot_shared_data, pdata))          #movimenta no mujoco
        )

        # obtem novas observações
        pl = obs_pipeline(robot_shared_data, runpar.obs_stat, pl)

        # de acordo com as observações obtem a recompensa
        pl = reward_pipeline(progress, robot_shared_data, pl)

        # dá a forma final aos valores de retorno
        pl = pl.bind(shape_return)
        pl = pl.bind(last_action_update)

        return pl.run(state)

    return step_fn

def create_reset(network_settings: NetworksSettings, network_parameters: NetworkParameters, robot_shared_data: RobotSharedData):
     def reset_fn(progress, state, runpar: RunningParameters):
         # 1. Sorteia o novo alvo
         rng, goal = get_goal(robot_shared_data.range_config, progress, state["rng"])
         
         # 2. Reseta os dados físicos do MuJoCo para a pose inicial
         # Isso garante que se o robô quebrou/caíu, ele volte a ficar em pé
         mjx_model = robot_shared_data.mjx_model
         init_data = mjx.make_data(mjx_model) # Pose padrão do modelo
         
         new_state = {
             **state,
             "rng": rng, 
             "goal": goal,
             "mjx_data": init_data, # Reset físico!
             "step": 0.0,           # Zera o contador de passos do episódio
             "success_count": 0.0   # Zera o contador de sucessos
         }
         return new_state, None
     return reset_fn