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
import mjx_base
import flax.linen as nn
from mujoco import MjModel  # type: ignore
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx
from etils import epath
from flax import struct
from typing import Any, Dict, Tuple, List, cast
from config import RangeConfig, RewardConfig, MujocoSimConfig
from enviroment import StateMonad
from mathutils import l1_l2_reward, exp_scale_reward, conv2jax_quat, cont_sample_beta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataclassutils import NetworksSettings, NetworkParameters, RunningParameters, RunningAvg


@struct.dataclass
class RobotSharedData:
    mjx_model: mjx.Model
    mj_model: MjModel
    init_q: jax.Array
    init_ctrl: jax.Array
    lowers: float
    uppers: float
    joint_ids: List[Any]
    joint_qposadr: Any
    joint_qveladr: Any
    default_pose: jax.Array
    tool_tip_id: int
    tool_base_id: int
    enviroment_config: MujocoSimConfig
    reward_config: RewardConfig
    range_config: RangeConfig
    
def create_rsd(
    xml_path: epath.Path,
    model_path: epath.Path,
    meshes_path: epath.Path,
    enviroment_config: MujocoSimConfig,
    reward_config: RewardConfig,
    range_config: RangeConfig,
    arm_joints: List[str]
):
    """
    Método fábrica para o ambiente
    """

    # Obtem os assets com base nos caminhos
    assets = {}
    mjx_base.update_assets(assets, model_path, "*.xml")
    mjx_base.update_assets(assets, meshes_path)

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

    # para rodar com o mjx
    mjx_model = mjx.put_model(mj_model, impl=str(enviroment_config.impl))

    # posiçao inicial, controle, limites e pose inicial
    keyframe = "home"
    init_q = jnp.array(mj_model.keyframe(keyframe).qpos)
    init_ctrl = jnp.array(mj_model.keyframe(keyframe).ctrl)
    lowers, uppers = mj_model.actuator_ctrlrange.T

    # ids para as juntas
    joint_ids = [mj_model.joint(j).id for j in arm_joints]

    # mapeia cada joint id para um qpos equivalente.
    joint_qposadr = mjx_base.get_qpos_ids(mj_model, arm_joints)

    # mapeia uma lista de itens qvel (dim = n° de graus de liberdade da junta) para cada junta
    joint_qveladr = mjx_base.get_qvel_ids(mj_model, arm_joints)

    # para obter a pose padrão
    default_pose = init_q[joint_qposadr]

    # ids para a ponta colocada no robô
    tool_tip_id = mj_model.site("tool_tip").id
    tool_base_id = mj_model.site("tool_base").id

    return RobotSharedData(
        mjx_model,
        mj_model,
        init_q,
        init_ctrl,
        lowers,
        uppers,
        joint_ids,
        joint_qposadr,
        joint_qveladr,
        default_pose,
        tool_tip_id,
        tool_base_id,
        enviroment_config,
        reward_config,
        range_config,
    )


##############################################################################################################3
def qpos(rsd: RobotSharedData, mjx_data: mjx.Data):
    return mjx_data.qpos[rsd.joint_qposadr]

def qvel(rsd:RobotSharedData, mjx_data: mjx.Data):
    return mjx_data.qvel[rsd.joint_qveladr]

def qfrc(mjx_data: mjx.Data):
    return mjx_data.qfrc_actuator

def sensor_data(
    rsd: RobotSharedData, mjx_data: mjx.Data, sensor_name: str
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
    sensor_id = rsd.mj_model.sensor(sensor_name).id
    sensor_adr = rsd.mj_model.sensor_adr[sensor_id]
    sensor_dim = rsd.mj_model.sensor_dim[sensor_id]
    return mjx_data.sensordata[sensor_adr : sensor_adr + sensor_dim]


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


def get_goal(rsd: RobotSharedData, progress, rng):

    rng, position = rsd.range_config.position.sample_normal(rng, progress)
    rng, position_velocities = rsd.range_config.position_velocities.sample_normal(rng, progress)
    rng, orientation = rsd.range_config.orientation.sample_normal(rng, progress)
   

    goals = {
        "goal_position_coordinates": position,
        "goal_position_velocities": position_velocities,
        "goal_orientation_coordinates": orientation
    }
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
                    {**pdata, "tool_position": sensor_data(rsd, state["mjx_data"], "tool_position")},
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
                    {**pdata, "orientation": sensor_data(rsd, state["mjx_data"], "tool_orientation")},
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
                    {**pdata, "torques": qfrc(state["mjx_data"])},
                )
            )
        )
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {
                        **pdata,
                        "joint_angles": qpos(rsd, state["mjx_data"]),
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
                        "joint_vel": qvel(rsd, state["mjx_data"]),
                    },
                )
            )
        )
        .map(
            lambda pdata: {
                **pdata,
                "pose_dist": pdata["joint_angles"] - rsd.default_pose,
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
                    ) +
                    # Penalidade L2 de torque para evitar movimentos espasmódicos
                    jnp.sum(jnp.square(pdata["torques"])) * reward_config.torques_penalty.update(progress)
                ),
            }
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

                        # Bônus de Sucesso + Bônus de Velocidade
                        "reward": pdata["reward"]
                        + pdata["success"] * (reward_config.success_reward.update(progress) + (30 - state["step"]) * 5.0)
                        + pdata["failure"] * reward_config.failure_penalty.update(progress),
                    },
                )
            )
        )
        .map(lambda pdata: {**pdata, "reward": jnp.clip(pdata["reward"], -1000.0, 1000.0)})
    )


####################################################################################################################
def create_step(network_settings: NetworksSettings, network_parameters: NetworkParameters, robot_shared_data: RobotSharedData):
    """
    state.keys() = ["rng", "step", "goal", "obs_history", "action", "mjx_data"]
    """

   
    def get_action():
        def fn(state):
            last_obs = state["obs"]
            rng1, rng2 = jax.random.split(state["rng"])

            output = network_settings.actor.apply(network_parameters.actor_params, last_obs)
            output = cast(jax.Array, output)
            action_value, logprob = cont_sample_beta(output, rng1)

            
            new_state = {**state, "rng": rng2, "action": action_value}
            return new_state, {"action": action_value, "logprob": logprob}
        return StateMonad(fn)

    def get_motor_targets(pdata):
        def fn(state):
            # escala ação para de [0, 1] para [-1, 1]
            action_value_action = 2.0 * pdata["action"]- 1.0

            # configura novos alvos para os motores, de acordo com a ação  selecionada a partir da posição atual
            current_pos = qpos(robot_shared_data, state["mjx_data"])
            motor_targets = current_pos + action_value_action * robot_shared_data.enviroment_config.action_scale

            # para evitar que os limites de junta do robô sejam desrespeitados
            return state, {**pdata, "motor_targets":jnp.clip(motor_targets, robot_shared_data.lowers, robot_shared_data.uppers)}
        return StateMonad(fn)

    def mujoco_step(pdata):
        def fn(state):
            mjx_data = mjx_base.mjx_step(
                robot_shared_data.mjx_model,
                state["mjx_data"],
                pdata["motor_targets"],
                robot_shared_data.enviroment_config.n_substeps,
            )
            state = {**state, "mjx_data": mjx_data}
            return state , pdata
        return StateMonad(fn)
    
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
    
    def normalize_obs_step(pdata: dict):
        return StateMonad(
            lambda state: (
                state,
                {
                    **pdata,
                    # 'obs' é o array concatenado que veio dos passos anteriores
                    "obs": (pdata["obs"] - state["obs_stats"].mean) / 
                        jnp.sqrt(state["obs_stats"].var + 1e-8)
                }
            )
        )
    
    def step_fn(progress, state, runpar: RunningParameters):

        # obtem uma ação pela observação anterior
        pl = (get_action()
            .bind(get_motor_targets)    #obtem para os motores segundo a ação
            .bind(mujoco_step)          #movimenta no mujoco
        )

        # obtem novas observações
        pl = obs_pipeline(robot_shared_data, runpar.obs_stat, pl)

        # de acordo com as observações obtem a recompensa
        pl = reward_pipeline(progress, robot_shared_data, pl)

        # dá a forma final aos valores de retorno
        pl = pl.bind(shape_return)

        return pl.run(state)

    return step_fn
