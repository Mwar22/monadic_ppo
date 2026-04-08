
import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from typing import Any, cast, Tuple, Callable, Self
from robot import RobotSharedData


@struct.dataclass
class RunningAvg:
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def init(cls, shape) -> Self:
        # Inicializamos com uma contagem pequena para evitar divisões por zero
        return cls(
            mean=jnp.zeros(shape),
            var=jnp.ones(shape),
            count=jnp.array(1e-4)
        )
    
    @jax.jit
    def update(self, batch_obs: jax.Array):
        """
        Atualiza média e variância usando o Algoritmo de Welford vetorizado.
        batch_obs esperado: (num_envs, num_steps, obs_dim)
        """
        # achata as dimensões de batch (envs * steps) pra ficar mais facil
        obs_dim = batch_obs.shape[-1]
        batch_obs = batch_obs.reshape(-1, obs_dim)
        
        # obtem as estatisticas do batch atual
        batch_mean = jnp.mean(batch_obs, axis=0)
        batch_var = jnp.var(batch_obs, axis=0)
        batch_count = jnp.array(batch_obs.shape[0], dtype=jnp.float32)

        # lógica do algoritmo de welford
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * (self.count * batch_count / total_count)
        new_var = M2 / total_count

        # evita que gradientes sejam calculados
        return jax.lax.stop_gradient(RunningAvg(mean=new_mean, var=new_var, count=total_count))

@struct.dataclass   
class RunningExponentialAvg:
    ema_value: jax.Array
    alpha: float

    @classmethod
    def init(cls, initial_ema_value: jax.Array = jnp.array(0), alpha: float =0.9) -> Self:
        return cls(initial_ema_value, alpha)
    
    def update(self, update_value:jax.Array):
        new_ema = (self.alpha * self.ema_value) + ((1 - self.alpha) * update_value)
        return RunningExponentialAvg(new_ema, self.alpha)
    
@struct.dataclass   
class RunningProgress:
    value: jax.Array
    step_size: float

    @classmethod
    def init(cls, initial_value: jax.Array = jnp.array(0), step_size=0.005) -> Self:
        return cls(initial_value, step_size)
    
    def update(self, success_rate, target_success):
        delta = jnp.where(success_rate > target_success, self.step_size, -self.step_size)
        new_value = jnp.clip(self.value + delta, 0.0, 1.0)
        return RunningProgress(new_value, self.step_size)

@struct.dataclass
class RunningParameters:
    obs_stat: RunningAvg
    ema_success: RunningExponentialAvg
    progress: RunningProgress
    success_rate: jax.Array
    target_success: float 

    @classmethod
    def init(cls, obs_shape, target_success: float = 0.6) -> Self:
        # Inicializamos com uma contagem pequena para evitar divisões por zero
        return cls(
            RunningAvg.init(obs_shape),
            RunningExponentialAvg.init(),
            RunningProgress.init(),
            jnp.array(0),
            target_success
        )
    
    def update(self, batch_obs: jax.Array, success_rate: jax.Array, target_success: float | None = None):
        new_obs_stat = self.obs_stat.update(batch_obs)
        new_ema_success = self.ema_success.update(success_rate)

        new_rp = lambda ts: RunningParameters(
            new_obs_stat,
            new_ema_success,
            self.progress.update(new_ema_success.ema_value, ts),
            success_rate,
            ts
        )
        
        if target_success is not None:
           return new_rp(target_success)
        
        return new_rp(self.target_success)
    

@struct.dataclass
class NetworkParameters:
    actor: Any
    critic: Any

    @classmethod
    def init(cls, actor_params, critic_params) -> Self:

        if isinstance(actor_params, Tuple):
            actor_params = actor_params[0]

        if isinstance(critic_params, Tuple):
            critic_params = critic_params[0]

        return cls(actor_params, critic_params)

    def update(self, actor_params, critic_params):
        return NetworkParameters(actor_params, critic_params)
    
@struct.dataclass
class NetworksSettings:
    obs_size: int
    action_size: int
    actor: nn.Module
    critic: nn.Module

    @classmethod
    def init(cls, 
        obs_size: int,
        action_size: int,
        actor: nn.Module,
        critic: nn.Module,
        params: Tuple[Any, Any],
    ):
        return cls(
            obs_size,
            action_size,
            actor,
            critic,
        )
    

@struct.dataclass
class TrainingSettings:
    network_settings: NetworksSettings
    cycles_per_goal: int
    epochs: int
    num_envs: int
    rollout_steps: int
    gamma: float
    gae_lambda: float

    robot_shared_data: RobotSharedData
    optimizer: optax.GradientTransformationExtraArgs
    optimizer_state: optax.OptState
    step_fn_creator: Callable
    target_success: float

    @classmethod
    def init(
        cls,
        network_settings: NetworksSettings,
        network_params: NetworkParameters,
        robot_shared_settings: RobotSharedData,
        optimizer_creator: Callable[[int], optax.GradientTransformationExtraArgs],
        step_fn_creator:Callable[[NetworksSettings, NetworkParameters, RobotSharedData], Callable],
        num_envs: int = 1,
        cycles_per_goal = 1,
        epochs: int = 1,
        rollout_steps: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        target_success: float = 0.6,
    ):
        # numero de passos para o escalonador de LR baseado na configuração
        total_steps = robot_shared_settings.range_config.numberof_goals * cycles_per_goal * epochs

        optimizer = optimizer_creator(total_steps)
        optimizer_params = optimizer.init(cast(optax.Params, network_params))

        
        return cls(
            network_settings,
            cycles_per_goal,
            epochs,
            num_envs,
            rollout_steps,
            gamma,
            gae_lambda,
            robot_shared_settings,
            optimizer,
            optimizer_params,
            step_fn_creator,
            target_success
        )
    

@struct.dataclass
class BatchedBuffer:
    obs_buffer: jax.Array       # (num_envs, rollout_steps +1, *obs_shape)
    action_buffer: jax.Array    # (num_envs, rollout_steps +1, *action_shape)
    reward_buffer: jax.Array    # (num_envs, rollout_steps +1)
    logprob_buffer: jax.Array   # (num_envs, rollout_steps +1)
    ptr: jax.Array  # (num_envs,)
    done_flag: jax.Array # (num_envs,)  se atingiu o alvo ou colidiu/ultrapassou os limites
    stop_flag: jax.Array # (num_envs,)  para parar com o rollout. True se done_flag estiver true ou o buffer estiver cheio

    def __str__(self):
        return f"obs_buffer: {self.obs_buffer}\n \
        action_buffer: {self.action_buffer}\n \
        reward_buffer: {self.reward_buffer}\n \
        logprob_buffer: {self.logprob_buffer}\n \
        ptr: {self.ptr}\n \
        done_flag: {self.done_flag} \
        stop_flag: {self.done_flag}"
    
    @property
    def num_steps(self):
        return self.reward_buffer.shape[1]
    
    @property
    def is_full(self):
        return self.ptr >= self.num_steps
    
    @classmethod
    def init(cls, settings: TrainingSettings):
        num_envs = settings.num_envs
        rollout_steps = settings.rollout_steps +1
    
        return cls(
            jnp.zeros((num_envs, rollout_steps, settings.network_settings.obs_size)),
            jnp.zeros((num_envs, rollout_steps, settings.network_settings.action_size)),
            jnp.zeros((num_envs, rollout_steps), dtype=jnp.float32),
            jnp.zeros((num_envs, rollout_steps), dtype=jnp.float32),
            jnp.zeros((num_envs,), dtype=jnp.int32),
            jnp.zeros((num_envs,), dtype=jnp.bool),
            jnp.zeros((num_envs,), dtype=jnp.bool),
        )
    
    @classmethod
    def push(
        cls,
        obs_buffer: jax.Array,
        action_buffer: jax.Array,
        reward_buffer: jax.Array,
        logprob_buffer: jax.Array,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        logprob: jax.Array,
        ptr: jax.Array
    ):
        """
        Adiciona um dado em um buffer de dimensões (rollout_steps, *data_shape)

        Parameters
        ----------
        obs_buffer: jax.Array
            Buffer considerando apenas um único ambiente, (rollout_steps, obs_shape)

        reward_buffer: jax.Array
            Buffer considerando apenas um único ambiente, (rollout_steps, )

        ptr: jax.Array
            Ponteiro para a posição atual no buffer

        obs: jax.Array
            observação

        reward: jax.Array
            recompensa
        """

        ptr = jnp.minimum(ptr, obs_buffer.shape[0] - 1)
        obs_buffer = obs_buffer.at[ptr].set(obs)
        action_buffer = action_buffer.at[ptr].set(action)
        reward_buffer = reward_buffer.at[ptr].set(reward)
        logprob_buffer = logprob_buffer.at[ptr].set(logprob)

        return obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr + 1
