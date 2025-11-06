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
import flax.linen as nn
from jax import jit
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx
from etils import epath
from flax import struct
from dataclasses import fields
from typing import Any, Dict, Optional, Union, Tuple, List , NewType
from mjx_base import ResetConfig, RangeConfig, RewardConfig, EnviromentConfig
from enviroment import EnvState, Data, State, Action


MjModel = NewType('MjModel', str)


def create_joystick(
        xml_path: epath.Path,
        model_path: epath.Path,
        meshes_path: epath.Path,
        enviroment_config: EnviromentConfig,
        reward_config: RewardConfig,
        range_config: RangeConfig,
        reset_config: ResetConfig,
        arm_joints: List[str]
    ):
    """
    Método fábrica para o ambiente
    """
    
    #Obtem os assets com base nos caminhos
    assets = {}
    mjx_base.update_assets(assets, model_path "*.xml")
    mjx_base.update_assets(assets, meshes_path)

    # configura modelos do mujoco e do mujoco mjx_env
    mj_model = MjModel.from_xml_string(
        xml_path.read_text(), assets=assets
    )

     # configura o timestep
    mj_model.opt.timestep = enviroment_config.sim_dt

    # dimensões de altura e largura
    mj_model.vis.global_.offwidth = 3840
    mj_model.vis.global_.offheight = 2160

    # para rodar com o mjx
    mjx_model = mjx.put_model(mj_model, impl=str(enviroment_config.impl))

    # posiçao inicial, controle, limites e pose inicial
    keyframe= "home"
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


    return Joystic(
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
        reset_config
    )

@struct.dataclass
class Joystic:
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
    

    def reset(self, state: State):
        """
        Reseta o ambiente.

        Parameters
        ----------
        rng: jax.Array
            Chave para gerar valores randômicos consistentes.

        Returns
        -------
        ret: mjx_env.State
            Estado atual do ambiente
        """
        rng, rng_q, rng_sample, rng_obs= jax.random.split(state.rng, 4)

        # inicializa as posições de junta e as velocidades e randomiza a posição inicial
        random_q = self.init_q.copy()

        # incrementa de acordo com uma distribuição uniforme, em conformidade com os limites,
        # e depois clampea o resultado
        random_q += jax.random.uniform(rng_q, random_q.shape, minval=self.min_qpos_rnd, maxval=self.max_qpos_rnd)
        random_q= jnp.clip(random_q, min=self.min_qpos_clp, max=self.max_qpos_clp)

        init_qvel = jnp.zeros(self.mjx_model.nv, dtype=float)
        ctrl = self.init_ctrl

        # Cria os dados de simulação
        mjx_data = mjx_base.init(
            self.mjx_model,
            qpos=random_q,
            qvel=init_qvel,
            ctrl=ctrl,
        )

        # informações
        info = {
            "step": 0,
            "last_act": jnp.zeros(self.mjx_model.nu),
            "last_vel": init_qvel[self.joint_qveladr],
            "tool/position": self._get_tool_position(mjx_data),
            "tool/orientation": self._get_tool_quaternion(mjx_data),
        }
        info.update(self.sample_command(rng_sample))

        # Observação inicial
        obs_history = jnp.zeros(15 * 39)  # store 15 steps of history
        obs = self._get_obs(data=mjx_data, info=info, obs_history=obs_history, rng=rng_obs)

        # flags de recompensas e dones
        reward, done = jnp.zeros(2)

        # salva as metricas
        metrics = {}
        for k in fields(self.reward_config):
            metrics[f"reward/{k}"] = jnp.zeros(())

        new_state = State(rng, {"obs": obs , "mjx_data": mjx_data})
        return new_state, Data(obs, reward, done, info)


    def step(self, state: State, action: Action):
        rng, cmd_rng, noise_rng, pert_rng = jax.random.split(state.rng, 4)


        # configura novos alvos para os motores, de acordo com a ação  selecionada a partir da posição padrão
        motor_targets = self.default_pose + action.value * self.enviroment_config.action_scale

        # para evitar que os limites de junta do robô sejam desrespeitados
        motor_targets = jnp.clip(motor_targets, self.lowers, self.uppers)

        data = mjx_base.mjx_step(
            self.mjx_model, state.data["mjx_data"], motor_targets, self.enviroment_config.n_substeps
        )

        obs = self._get_obs(data, state.info, state.obs, noise_rng)

        # obtem as posições de junta e velocidades atuais
        joint_angles = data.qpos[self.joint_qposadr]

        joint_vel = data.qvel[self.joint_qveladr]
    
        # se tiver alcançado os objetivos de posição e orientação
        done = (self._position_error(data, state.info) < 0.0001) & (self._orientation_error(data, state.info) < 0.0001)

        # termina se os limites de junta forem ultrapassados
        done |= jnp.any(joint_angles < self.lowers)
        done |= jnp.any(joint_angles > self.uppers)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done
        )

        # para aplicar as escalas nas recompensas utilizadas
        rewards = {
            k: v * getattr(self.reward_config, k) for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.enviroment_config.dt, 0.0, 10000.0)

        # regostrando
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["step"] += 1
        state.info["rng"] = rng

        # caso o numero de passos seja maior que 500, substitui por um comando novo
        new_command = self.sample_command(cmd_rng)
        state = state.replace(
            info={
                **state.info,
                **jax.tree.map(
                    lambda new_val, old_val: jnp.where(state.info["step"] > 500, new_val, old_val),
                    new_command,
                    {k: state.info[k] for k in new_command},
                ),
            }
        )

        # marcado como done ou caso tenha passado de 500 passos, reseta o contador de passos
        state.info["step"] = jnp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )

        # salva as metricas de cada recompensa
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v


        # converte done para float32 e atualiza o estado do ambiente
        done = jnp.float32(done)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state



    def _get_obs(
        self,
        state: State,
        data: Data,
        obs_history: jax.Array,
    ) -> jax.Array:
        obs = jnp.concatenate([
            data.info["goal/position"],
            data.info["goal/orientation"],
            data.info["goal/lin_velocity"],
            data.info["goal/ang_velocity"],
            data.info["tool/position"],
            data.info["tool/orientation"],
            self._position_error(data.info["goal/position"], state.data["mjx_data"]).reshape(1,),
            self._orientation_error(data.info["goal/orientation"], state.data["mjx_data"]).reshape(1,),
            state.data["mjx_data"].data.qfrc_actuator,     #torques
            state.data["mjx_data"].qpos[self.joint_qposadr] - self.default_pose,
            data.info["last_act"],
        ])

        rng, rng1 = jax.random.split(state.rng)
        obs = jnp.clip(obs, -100.0, 100.0)
        if self.enviroment_config.obs_noise >= 0.0:
            noise = self.enviroment_config.obs_noise * jax.random.uniform(
                rng, obs.shape, minval=-1.0, maxval=1.0
            )
            obs += noise
        obs = jnp.roll(obs_history, obs.size).at[: obs.size].set(obs)
        return obs

##############################################################################################################
def conv2jax_quat(mujoco_quat: jnp.ndarray) -> jnp.ndarray:
        """Converte quaternion no formato (w, x, y, z) -> (x, y, z, w)"""
        return jnp.array([mujoco_quat[1], mujoco_quat[2], mujoco_quat[3], mujoco_quat[0]])

##############################################################################################################3
def sample_config_coordinates(range_config: RangeConfig, config_name: str, goal_data: Dict[str, Any]) -> EnvState:
    """
    Obtem comandos aleatórios para as posições
    :: r, c, g -> EnvState s g'
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
        def scale(_, input):
            c, sample = input
            return c[0] + (c[1] - c[0]) * sample

        inputs = (config_value, samples)
        _, scaled_coord = jax.lax.scan(scale, None, inputs)

        str_id = config_name + "_coordinates"
        new_state = {**state, "rng": rng1}
        return new_state, {**goal_data, str_id: scaled_coord}
    
    return EnvState(func)
      
def sample_config_velocities(range_config: RangeConfig, config_name: str, goal_data: Dict[str, Any]) -> EnvState:
    """
    Obtem comandos aleatórios para as velocidades
    :: r, c, g -> EnvState s g'
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
        def scale(_, input):
            c, sample = input
            return c[1]* sample

        inputs = (config_value, samples)
        _, scaled_coord = jax.lax.scan(scale, None, inputs)

        str_id = config_name + "_velocities"
        new_state = {**state, "rng": rng1}
        return new_state, {**goal_data, str_id: scaled_coord}
    
    return EnvState(func)

def tool_position(model: MjModel, data: mjx.Data) -> jax.Array:
        return mjx_base.get_sensor_data(model, data, "tool_position")

def tool_quaternion(model: MjModel, data: mjx.Data) -> jax.Array:
    """
    Obtem o quaternion de orientação da ferramenta em relação ao Sistema de coordenadas global.
    o eixo da ferramenta se alinha com o eixo z do sistema global
    """
    mj_quat = mjx_base.get_sensor_data(model, data, "tool_orientation")
    return conv2jax_quat(mj_quat)

def position_error(model: MjModel, data: mjx.Data, position: jax.Array) -> jax.Array:
    """
    Calcula o erro de posição.

    Parameters
    ----------
    data: mjx.Data
        Estado dinâmico que atualiza a cada step.

    info: dict[str, Any]
        Dicionario de informações
    """
    return jnp.linalg.norm(position - tool_position(model, data), ord=2)

def orientation_error(model: MjModel, data: mjx.Data, orientation: jax.Array) -> jax.Array:
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
    r_target = Rotation.from_euler('zyx', orientation)
    r_measured = Rotation.from_quat(tool_quaternion(model, data))

    # calcula a transformação  "erro", com base em: r_measured = r_error * r_target
    r_error = r_measured * r_target.inv()

    r_error = r_measured * r_target.inv()
    return jnp.linalg.norm(r_error.as_rotvec())
##################################################################################################################

def exp_scale_reward(
    gain,
    sigma,
    value: jax.Array
) -> jax.Array:
    
    return gain * jnp.exp(-value/sigma)

def l1_l2_reward(
    gain_l1,
    gain_l2,
    value: jax.Array
):
    return gain_l2*jnp.linalg.norm(value, ord=2) + gain_l1*jnp.linalg.norm(value, ord=1)

def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    """
    Penaliza as diferenças entre os vetores de ações por meio da norma L2

    Parameters
    ----------
    act: jax.Array
        Ação atual.
    
    last_act: jax.Array
        Ação anterior
    """
    return jnp.linalg.norm(act - last_act, ord=2)

def stand_still_reward(
    gain,
    position_velocities: jax.Array,
    orientation_velocities: jax.Array,
    default_pose: jax.Array,
    joint_angles: jax.Array,
) -> jax.Array:
    """
    Penaliza caso o comando (velocidades de movimentação) indicar que o robô deva estar parado.
    Caso penalizado, a penalização é de acordo com a norma L1 encima das diferenças entre os
    ângulos em joint_angles e a pose _default_pose.

    Parameters
    ----------
    commands: Dict[str, jax.Array]
        Dicionário que mapeia uma informação dos comandos sampleados.

    joint_angles: jax.Array
        Angulos atuais das juntas do robô
    """
    linear_velocity = jnp.linalg.norm(position_velocities)
    angular_velocity = jnp.linalg.norm(orientation_velocities)

    mask = jnp.logical_and(linear_velocity < 0.001, angular_velocity < 0.001)
    return gain*jnp.linalg.norm(joint_angles - default_pose) *  mask.astype(jnp.float32)

def check_done(joystick: Joystic, joint_angles: jax.Array, position_error, orientation_error):
    
    # se tiver alcançado os objetivos de posição e orientação
    done = (position_error < 0.0001) & (orientation_error < 0.0001)

    # termina se os limites de junta forem ultrapassados
    done |= jnp.any(joint_angles < joystick.lowers)
    done |= jnp.any(joint_angles > joystick.uppers)
    
###################################################################################################################

def goal_pipeline(env: EnvState, range_config):
    return (
        env.bind(lambda goal_data: sample_config_coordinates(range_config, "position", goal_data))
        .bind(lambda goal_data: sample_config_velocities(range_config, "position", goal_data))
        .bind(lambda goal_data: sample_config_coordinates(range_config, "orientation", goal_data))
        .bind(lambda goal_data: sample_config_velocities(range_config, "orientation", goal_data))
        .map(lambda goal_data: {"goal": goal_data})
    )

def tool_pipeline(env: EnvState, model, data: mjx.Data):
    return (
        env.map(lambda goal_data: {**goal_data, "position" : tool_position(model, data)})
        .map(lambda data: {**data, "position_error":position_error(model, data, data["position"])})
        .map(lambda data: {**data, "orientation": tool_quaternion(model, data)})
        .map(lambda data: {**data, "orientation_error":orientation_error(model, data, data["orientation"])})
    )


def other_pipeline(joystick: Joystic, env: EnvState, data: mjx.Data):
    # obtem as posições de junta e velocidades atuais
    joint_angles = data.qpos[joystick.joint_qposadr]
    joint_vel = data.qvel[joystick.joint_qveladr]

    torques = data.qfrc_actuator,     #torques
    pose_dist = joint_angles - joystick.default_pose, #distancias de pose em relação à padrão

    return (
        env.map(lambda data: {
        **data, "torques":torques, "pose_dist":pose_dist, "joint_angles": joint_angles, "joint_vel": joint_vel
        })
    )


def reward_pipeline(joystick: Joystic, env: EnvState):
    reward_config = joystick.reward_config
    return (

        #recompensa para quanto menor o erro de posição
        env.map(lambda data: {**data, "reward": exp_scale_reward(1, reward_config.tracking_sigma, data["position_error"])})

        #recompensa para quanto menor o erro de orientação
        .map(lambda data: {**data, "reward": data["reward"] + exp_scale_reward(1, reward_config.tracking_sigma, data["orientation_error"])})
        
        #penalidade pelos torques
        .map(lambda data: {**data, "reward": data["reward"] + l1_l2_reward(reward_config.torques, 0, data["torques"])})

        #penalidade por terminação
        .map(lambda data: {**data, "done": check_done(joystick, data["joint_angles"], data["position_error"], data["orientation_error"])})
        .map(lambda data: {**data, "reward": data["reward"] +  reward_config.termination * (data["done"] & (data["step"] < 500))})

        # penalidade por ficar parado
        .map(lambda data: 
            {**data, 
            "reward": data["reward"] + 
                stand_still_reward(
                    reward_config.stand_still,
                    data["goal"]["position_velocities"],
                    data["goal"]["orientation_velocities"],
                    joystick.default_pose,
                    data["joint_angles"]
                )
            }
        )
        .map(lambda data: {**data, "reward": jnp.clip(data["reward"] * joystick.enviroment_config.dt, 0.0, 10000.0)})

    )

def create_step(
    joystick: Joystic,
    process_pipeline: EnvState,
    
    action: Action
) -> EnvState:
    
    def func(state):
        rng = state["rng"]
        rng, rng2 = jax.random.split(rng, 2)

        # configura novos alvos para os motores, de acordo com a ação  selecionada a partir da posição padrão
        motor_targets = joystick.default_pose + action.value * joystick.enviroment_config.action_scale

        # para evitar que os limites de junta do robô sejam desrespeitados
        motor_targets = jnp.clip(motor_targets, joystick.lowers, joystick.uppers)

        data = mjx_base.mjx_step(
            joystick.mjx_model, state.data["mjx_data"], motor_targets, joystick.enviroment_config.n_substeps
        )

        op = goal_pipeline(EnvState.pure({}), joystick.range_config)

        #obtem um novo alvo a partir do pipeline
        goal_data = {"goal": state["goal"]}
        if state["step"] > 500 or state["step"] == 0:
            state, goal_data = op.run(state)
            state["step"] = 0

        op = tool_pipeline(EnvState.pure(goal_data), joystick.mj_model, data)
        op = other_pipeline(joystick, op, data)
        state, obs = op.run(state)

        #obtem as recompensas
        rp = reward_pipeline(joystick, EnvState.pure(obs))
        state, rewards = rp.run(state)

        new_state ={**state, "rng": rng2, "step": state["step"] + 1, "goal": goal_data["goal"]}
        return new_state, Data(obs, rewards, None)

    return EnvState(func)


def step2(self, state: State, action: Action):
        rng, cmd_rng, noise_rng, pert_rng = jax.random.split(state.rng, 4)


        # configura novos alvos para os motores, de acordo com a ação  selecionada a partir da posição padrão
        motor_targets = self.default_pose + action.value * self.enviroment_config.action_scale

        # para evitar que os limites de junta do robô sejam desrespeitados
        motor_targets = jnp.clip(motor_targets, self.lowers, self.uppers)

        data = mjx_base.mjx_step(
            self.mjx_model, state.data["mjx_data"], motor_targets, self.enviroment_config.n_substeps
        )

        obs = self._get_obs(data, state.info, state.obs, noise_rng)

        # obtem as posições de junta e velocidades atuais
        joint_angles = data.qpos[self.joint_qposadr]

        joint_vel = data.qvel[self.joint_qveladr]
    
        # se tiver alcançado os objetivos de posição e orientação
        done = (self._position_error(data, state.info) < 0.0001) & (self._orientation_error(data, state.info) < 0.0001)

        # termina se os limites de junta forem ultrapassados
        done |= jnp.any(joint_angles < self.lowers)
        done |= jnp.any(joint_angles > self.uppers)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done
        )

        # para aplicar as escalas nas recompensas utilizadas
        rewards = {
            k: v * getattr(self.reward_config, k) for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.enviroment_config.dt, 0.0, 10000.0)

        # regostrando
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["step"] += 1
        state.info["rng"] = rng

        # caso o numero de passos seja maior que 500, substitui por um comando novo
        new_command = self.sample_command(cmd_rng)
        state = state.replace(
            info={
                **state.info,
                **jax.tree.map(
                    lambda new_val, old_val: jnp.where(state.info["step"] > 500, new_val, old_val),
                    new_command,
                    {k: state.info[k] for k in new_command},
                ),
            }
        )

        # marcado como done ou caso tenha passado de 500 passos, reseta o contador de passos
        state.info["step"] = jnp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )

        # salva as metricas de cada recompensa
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v


        # converte done para float32 e atualiza o estado do ambiente
        done = jnp.float32(done)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

