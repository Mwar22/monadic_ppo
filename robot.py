"""
Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------
Tarefa para o thor alcançar um alvo
"""

import mujoco
import jax
import mjx_base
from mujoco import MjModel  # type: ignore
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx
from etils import epath
from flax import struct
from typing import Any, Dict, Tuple, List
from config import ResetConfig, RangeConfig, RewardConfig, EnviromentConfig
from enviroment import StateMonad
from mathutils import l1_l2_reward, exp_scale_reward, conv2jax_quat


@struct.dataclass
class Robot:
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
    min_qpos_rnd: float
    max_qpos_rnd: float
    min_qpos_clp: float
    max_qpos_clp: float
    enviroment_config: EnviromentConfig
    reward_config: RewardConfig
    range_config: RangeConfig
    reset_config: ResetConfig


def create_robot(
    xml_path: epath.Path,
    model_path: epath.Path,
    meshes_path: epath.Path,
    enviroment_config: EnviromentConfig,
    reward_config: RewardConfig,
    range_config: RangeConfig,
    reset_config: ResetConfig,
    arm_joints: List[str],
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

    def calculate_joint_span(p: float = 0.1) -> Tuple[float, float]:
        """
        Calcula uma faixa de valores mínimos e maximos em termos de deviação percentual dos mínimos
        e máximos dos limites de junta. Por exemplo, para p = 0.1, teremos valores que se iniciam em 10%
        acima do valor mínimo e vão até 10% do valor máximo, considerando o range de valores disponíveis.

        Params
        ------
        p: float
            proporção escolhida.

        Returns
        -------
        ret: Tuple[float, float]
            valor mínimo e máximo respectivamente calculados.
        """
        span = uppers - lowers
        lower = lowers + p * span
        upper = uppers - p * span
        return lower, upper

    # para calcular os valores adequados de limites minimos e maximos para gerar incrementos aleatórios
    min_qpos_rnd, max_qpos_rnd = calculate_joint_span(reset_config.rnd_range)
    min_qpos_clp, max_qpos_clp = calculate_joint_span(reset_config.clip_range)

    return Robot(
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
        min_qpos_rnd,
        max_qpos_rnd,
        min_qpos_clp,
        max_qpos_clp,
        enviroment_config,
        reward_config,
        range_config,
        reset_config,
    )


##############################################################################################################3


def tool_position(model: MjModel, data: mjx.Data) -> jax.Array:
    return mjx_base.get_sensor_data(model, data, "tool_position")


def tool_quaternion(model: MjModel, data: mjx.Data) -> jax.Array:
    """
    Obtem o quaternion de orientação da ferramenta em relação ao Sistema de coordenadas global.
    o eixo da ferramenta se alinha com o eixo z do sistema global
    """
    mj_quat = mjx_base.get_sensor_data(model, data, "tool_orientation")
    return conv2jax_quat(mj_quat)


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
    robot: Robot, joint_angles: jax.Array, position_error, orientation_error
):
    # se tiver alcançado os objetivos de posição e orientação
    done = (position_error < 0.0001) & (orientation_error < 0.0001)

    # termina se os limites de junta forem ultrapassados
    done |= jnp.any(joint_angles < robot.lowers)
    done |= jnp.any(joint_angles > robot.uppers)

    return done


#################################################################################################################


def sample_config_coordinates(
    range_config: RangeConfig, config_name: str, goal_data: Dict[str, Any]
) -> StateMonad:
    """
    Obtem comandos aleatórios para as posições
    :: r, c, g -> StateMonad s g'
    """

    def func(state):
        """
        :: s -> (s', p)
        """
        rng, rng1 = jax.random.split(state["rng"])
        config_value = getattr(range_config, config_name)

        # faz um sampleamento
        samples = jax.random.uniform(rng, shape=(3,))

        # ajusta a escala para que fique dentro da faixa [min, max]
        scaled_coord = (
            config_value[:, 0] + (config_value[:, 1] - config_value[:, 0]) * samples
        )

        str_id = config_name + "_coordinates"
        new_state = {**state, "rng": rng1}
        return new_state, {**goal_data, str_id: scaled_coord}

    return StateMonad(func)


def sample_config_velocities(
    range_config: RangeConfig, config_name: str, goal_data: Dict[str, Any]
) -> StateMonad:
    """
    Obtem comandos aleatórios para as velocidades
    :: r, c, g -> StateMonad s g'
    """

    def func(state):
        """
        :: s -> (s', p)
        """
        rng, rng1 = jax.random.split(state["rng"])
        config_value = getattr(range_config, config_name)

        # faz um sampleamento
        samples = jax.random.uniform(rng, shape=(3,))

        # ajusta a escala para que fique dentro da faixa [0, max]
        scaled_coord = config_value[:, 1] * samples

        str_id = config_name + "_velocities"
        new_state = {**state, "rng": rng1}
        return new_state, {**goal_data, str_id: scaled_coord}

    return StateMonad(func)


##################################################################################################################


def update_obs_history(data, obs_noise=0.0):
    def func(state):
        rng, rng1 = jax.random.split(state["rng"])

        # clipa a observação
        obs_processed = jnp.clip(data["obs_array"], -100.0, 100.0)

        # Adiciona um ruido adicional.
        # Este 'if' funciona com JIT contanto que obs_noise seja um valor estático.
        if obs_noise >= 0.0:
            noise = obs_noise * jax.random.uniform(
                rng, obs_processed.shape, minval=-1.0, maxval=1.0
            )
            obs_processed += noise

        # Adiciona a nova observação no buffer, deslocando as outras observações e descartando a mais antiga
        new_obs_history = (
            jnp.roll(
                state["obs_history"], obs_processed.size
            )  # desloca todo o array para a direita, obs_processd.size de distancia (circular)
            .at[: obs_processed.size]
            .set(obs_processed)  # adiciona os novos dados de observação
        )
        new_state = {**state, "rng": rng1, "obs_history": new_obs_history}

        return new_state, {**data, "obs_history": new_obs_history}

    return StateMonad(func)


def concat_obs_as_array(d: Dict[str, Any]) -> StateMonad:
    """
    :: d -> StateMonad s c
    """

    def func(state):
        # Manually list keys to ensure order and handle scalars
        obs_list = [
            d["goal_position_coordinates"],  # (3,)
            d["goal_position_velocities"],  # (3,)
            d["goal_orientation_coordinates"],  # (3,)
            d["goal_orientation_velocities"],  # (3,)
            d["tool_position"],  # (3,)
            jnp.expand_dims(d["position_error"], axis=0),  # () -> (1,)
            d["orientation"],  # (4,)
            jnp.expand_dims(d["orientation_error"], axis=0),  # () -> (1,)
            d["torques"],  # (6,)
            d["joint_angles"],  # (6,)
            d["joint_vel"],  # (6,)
            d["pose_dist"],  # (6,)
        ]
        obs_array = jnp.concatenate(obs_list)
        # (3+3+3+3 + 3 + 1 + 4 + 1 + 6+6+6+6 = 45)

        return state, {**d, "obs_array": obs_array}

    return StateMonad(func)


###################################################################################################################


def goal_pipeline(env: StateMonad, range_config):
    return (
        env.bind(
            lambda pdata: sample_config_coordinates(
                range_config, "goal_position", pdata
            )
        )
        .bind(
            lambda pdata: sample_config_velocities(range_config, "goal_position", pdata)
        )
        .bind(
            lambda pdata: sample_config_coordinates(
                range_config, "goal_orientation", pdata
            )
        )
        .bind(
            lambda pdata: sample_config_velocities(
                range_config, "goal_orientation", pdata
            )
        )
    )


def tool_pipeline(env: StateMonad, model, data: mjx.Data):
    return (
        env.bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {**pdata, "tool_position": tool_position(model, state["mjx_data"])},
                )
            )
        )
        .map(
            lambda pdata: {
                **pdata,
                "position_error": position_error(
                    pdata["goal_position_coordinates"], pdata["tool_position"]
                ),
            }
        )
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {**pdata, "orientation": tool_quaternion(model, state["mjx_data"])},
                )
            )
        )
        .map(
            lambda pdata: {
                **pdata,
                "orientation_error": orientation_error(
                    pdata["goal_orientation_coordinates"], pdata["orientation"]
                ),
            }
        )
    )


def other_pipeline(robot: Robot, env: StateMonad, data: mjx.Data):
    return (
        env.bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {**pdata, "torques": state["mjx_data"].qfrc_actuator},
                )
            )
        )
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {
                        **pdata,
                        "joint_angles": state["mjx_data"].qpos[robot.joint_qposadr],
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
                        "joint_vel": state["mjx_data"].qvel[robot.joint_qveladr],
                    },
                )
            )
        )
        .map(
            lambda pdata: {
                **pdata,
                "pose_dist": pdata["joint_angles"] - robot.default_pose,
            }
        )
        .bind(lambda pdata: concat_obs_as_array(pdata))
        .bind(
            lambda pdata: update_obs_history(pdata, robot.enviroment_config.obs_noise)
        )
    )


def reward_pipeline(robot: Robot, env: StateMonad):
    reward_config = robot.reward_config
    return (
        # Penalidade (custo) por erro de posição
        env.map(
            lambda pdata: {
                **pdata,
                "reward": reward_config.position_error_penalty
                * pdata["position_error"],
            }
        )
        # penalidade (custo) por erro de orientação
        .map(
            lambda pdata: {
                **pdata,
                "reward": pdata["reward"]
                + reward_config.orientation_error_penalty * pdata["orientation_error"],
            }
        )
        # penalidade para coibir torques muito elevados
        .map(
            lambda pdata: {
                **pdata,
                "reward": pdata["reward"]
                + l1_l2_reward(reward_config.torques_penalty, 0, pdata["torques"]),
            }
        )
        # incentivo para sair do lugar (posição)
        .map(
            lambda pdata: {
                **pdata,
                "reward": pdata["reward"]
                + exp_scale_reward(
                    reward_config.tracking_incentive_gain,
                    reward_config.tracking_sigma,
                    pdata["position_error"],
                ),
            }
        )
        # incentivo para a orientação
        .map(
            lambda pdata: {
                **pdata,
                "reward": pdata["reward"]
                + exp_scale_reward(
                    reward_config.tracking_incentive_gain,  # You can use a different gain
                    reward_config.tracking_sigma,
                    pdata["orientation_error"],
                ),
            }
        )
        # penalidade por terminação
        .map(
            lambda pdata: {
                **pdata,
                # checa se houve sucesso (atingiu o alvo)
                "success": (pdata["position_error"] < 0.0001)
                & (pdata["orientation_error"] < 0.0001),
                # checa se atingiu o limite das juntas
                "failure": jnp.any(pdata["joint_angles"] < robot.lowers)
                | jnp.any(pdata["joint_angles"] > robot.uppers),
            }
        )
        .bind(
            lambda pdata: StateMonad(
                lambda state: (
                    state,
                    {
                        **pdata,
                        # aplica a correção baseado nas recompensas e penalidades combinados
                        "reward": pdata["reward"]
                        + pdata["success"]
                        * reward_config.success_reward
                        * (state["step"] < 500)
                        + pdata["failure"] * reward_config.failure_penalty,
                        # se atingiu o sucesso ou houve uma falha, termina o episódio
                        "done": pdata["success"] | pdata["failure"],
                    },
                )
            )
        )
        .map(
            lambda pdata: {
                **pdata,
                "reward": jnp.clip(pdata["reward"], -10000.0, 10000.0),
            }
        )
    )


####################################################################################################################
def create_reset(robot: Robot) -> StateMonad:
    def func(state):
        rng, rng2 = jax.random.split(state["rng"], 2)

        # inicializa as posições de junta e as velocidades e randomiza a posição inicial
        random_q = robot.init_q.copy()

        # incrementa de acordo com uma distribuição uniforme, em conformidade com os limites,
        # e depois clampea o resultado
        random_q += jax.random.uniform(
            rng,
            random_q.shape,
            minval=robot.min_qpos_rnd,
            maxval=robot.max_qpos_rnd,
        )
        random_q = jnp.clip(random_q, min=robot.min_qpos_clp, max=robot.max_qpos_clp)

        init_qvel = jnp.zeros(robot.mjx_model.nv, dtype=float)
        ctrl = robot.init_ctrl

        # Cria os dados de simulação
        mjx_data = mjx_base.init(
            robot.mjx_model,
            qpos=random_q,
            qvel=init_qvel,
            ctrl=ctrl,
        )

        # reseta o estado
        state["step"] = 0
        state["mjx_data"] = mjx_data

        # obtem um novo alvo a partir do pipeline
        gp = goal_pipeline(StateMonad.pure({}), robot.range_config)
        state, goal_data = gp.run(state)
        state["goal"] = goal_data

        tp = tool_pipeline(StateMonad.pure(goal_data), robot.mj_model, mjx_data)
        op = other_pipeline(robot, tp, mjx_data)
        state, obs = op.run(state)

        first_obs_frame = obs["obs_array"]

        # jnp.tile(first_obs_frame, 15) creates a (15 * 45,) array
        primed_obs_history = jnp.tile(first_obs_frame, 15)

        # Overwrite the bad obs_history in both the state and the obs_data dict
        state["obs_history"] = primed_obs_history
        obs["obs_history"] = primed_obs_history

        # para descobrir o tamanho do vetor de observação
        # print(f"obs_shape: {obs["obs_array"].shape}")

        # obtem as recompensas
        rp = reward_pipeline(robot, StateMonad.pure(obs))
        state, final_data = rp.run(state)

        state = {**state, "rng": rng2, "action": jnp.reshape(state["action"], (6,))}
        return state, {"obs": obs, "final_data": final_data}

    return StateMonad(func)


def create_step(
    robot: Robot,
) -> StateMonad:
    """
    state.keys() = ["rng", "step", "goal", "obs_history", "action", "mjx_data"]
    """

    def func(state):
        rng, rng2 = jax.random.split(state["rng"], 2)

        # escala para de [0, 1] para [-1, 1]
        action = (state["action"] * 2.0) - 1.0

        # configura novos alvos para os motores, de acordo com a ação  selecionada a partir da posição atual
        current_pos = state["mjx_data"].qpos[robot.joint_qposadr]
        motor_targets = current_pos + action * robot.enviroment_config.action_scale

        # para evitar que os limites de junta do robô sejam desrespeitados
        motor_targets = jnp.clip(motor_targets, robot.lowers, robot.uppers)

        mjx_data = mjx_base.mjx_step(
            robot.mjx_model,
            state["mjx_data"],
            motor_targets,
            robot.enviroment_config.n_substeps,
        )

        state = {**state, "mjx_data": mjx_data, "rng": rng}
        ########################################################

        # obtem a observação com base no alvo atual
        op = tool_pipeline(StateMonad.pure(state["goal"]), robot.mj_model, mjx_data)
        op = other_pipeline(robot, op, mjx_data)
        state, pdata = op.run(state)

        # obtem as recompensas e a flag done
        rp = reward_pipeline(robot, StateMonad.pure(pdata))
        state, final_data = rp.run(state)

        new_state = {**state, "rng": rng2, "step": state["step"] + 1}
        return new_state, final_data

    return StateMonad(func)
