from os import wait
import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from functools import partial
from typing import Dict, Any, cast, Tuple, Callable
from mathutils import  beta_entropy
from robot import RobotSharedData, get_goal
from mujoco import mjx


@struct.dataclass
class NetworksSettings:
    obs_size: int
    action_size: int
    actor: nn.Module
    critic: nn.Module
    params: Tuple[Any, Any]

    @property
    def actor_params(self):
        return self.params[0]
    
    @property
    def critic_params(self):
        return self.params[1]
    

@struct.dataclass
class TrainingSettings:
    network_settings: NetworksSettings
    num_episodes: int
    num_envs: int
    steps_per_episode: int
    gamma: float
    gae_lambda: float

    robot_shared_data: RobotSharedData
    optimizer: optax.GradientTransformationExtraArgs
    optimizer_state: optax.OptState
    step_fn: Callable

def create_training_settings(
    network_settings: NetworksSettings,
    robot_shared_settings: RobotSharedData,
    optimizer_creator: Callable[[float], optax.GradientTransformationExtraArgs],
    step_fn_creator:Callable[[NetworksSettings, RobotSharedData], Callable],
    num_envs: int = 1,
    num_episodes: int = 1,
    steps_per_episode: int = 1,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    optimizer = optimizer_creator(learning_rate)
    optimizer_params = optimizer.init(network_settings.params)
    step_fn = jax.jit(step_fn_creator(network_settings, robot_shared_settings))

    return TrainingSettings(
        network_settings,
        num_episodes,
        num_envs,
        steps_per_episode,
        gamma,
        gae_lambda,
        robot_shared_settings,
        optimizer,
        optimizer_params,
        step_fn
    )

@struct.dataclass
class BatchedBuffer:
    obs_buffer: jax.Array       # (num_envs, max_steps, *obs_shape)
    action_buffer: jax.Array    # (num_envs, max_steps, *action_shape)
    reward_buffer: jax.Array    # (num_envs, max_steps)
    logprob_buffer: jax.Array   # (num_envs, max_steps)
    ptr: jax.Array  # (num_envs,)
    done_flag: jax.Array  # (num_envs,)

    def __str__(self):
        return f"obs_buffer: {self.obs_buffer}\n \
        action_buffer: {self.action_buffer}\n \
        reward_buffer: {self.reward_buffer}\n \
        logprob_buffer: {self.logprob_buffer}\n \
        ptr: {self.ptr}\n \
        done_flag: {self.done_flag}"


def batched_buffer_create(settings: TrainingSettings):
    num_envs = settings.num_envs
    max_steps = settings.steps_per_episode
  
    return BatchedBuffer(
        jnp.zeros((num_envs, max_steps, settings.network_settings.obs_size)),
        jnp.zeros((num_envs, max_steps, settings.network_settings.action_size)),
        jnp.zeros((num_envs, max_steps), dtype=jnp.float32),
        jnp.zeros((num_envs, max_steps), dtype=jnp.float32),
        jnp.zeros((num_envs,), dtype=jnp.int32),
        jnp.zeros((num_envs,), dtype=jnp.bool),
    )


def push(
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
    Adiciona um dado em um buffer de dimensões (max_steps, *data_shape)

    Parameters
    ----------
    obs_buffer: jax.Array
        Buffer considerando apenas um único ambiente, (max_steps, obs_shape)

    reward_buffer: jax.Array
        Buffer considerando apenas um único ambiente, (max_steps, )

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


def rollout_step(
    step_fn,
    state: Dict[str, Any],
    obs_buffer: jax.Array,
    action_buffer: jax.Array,
    reward_buffer: jax.Array,
    logprob_buffer: jax.Array,
    ptr: jax.Array,
    done_flag: jax.Array,
):
    """
    Dá um step de rollout para um único ambiente

    Parameters
    ----------

    pipeline: StateMonad
        Pipeline de ajuste de objetivo, tomada de ação pelo agente, coleta de recompensas e formação do espaço de observação.
    """

    # Caso done_flag esteja como False
    def do_step(carry):
        state, obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag = carry
        rng, step_rng = jax.random.split(state["rng"])

        # executa o ambiente
        state = {**state, "rng": step_rng}  #atualiza o rng
        new_state, data = step_fn(state)

        # adiciona o dado no buffer
        obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr = push(
            obs_buffer,
            action_buffer,
            reward_buffer,
            logprob_buffer,
            data["obs"],
            data["action"],
            data["reward"],
            data["logprob"],
            ptr
        )

        # faz o update do rng
        new_state["rng"] = rng
        done_flag = data["done"] > 0.5
        return new_state, obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag

    # Caso done_flag esteja como True
    def no_step(carry):  #
        return carry

    return jax.lax.cond(
        done_flag, no_step, do_step, (state, obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag)
    )


def rollout(
    settings: TrainingSettings,
    init_state: Dict[str, Any],
    buffer: BatchedBuffer,
):
    # Match the structure of your 'state' dictionary exactly
    state_in_axes = {
        'action': 0,
        'goal': {
            'goal_orientation_coordinates': 0, # or 0 if batched
            'goal_orientation_velocities': 0,
            'goal_position_coordinates': 0,
            'goal_position_velocities': 0,
        },
        'mjx_data': 0, 
        'obs': 0,
        'rng': 0,
        'step': 0,
        'success_count':0
    }
    vmap_rollout_step = jax.vmap(
        partial(rollout_step, settings.step_fn),
        in_axes=(
            state_in_axes,  # Arg 0: state (was Arg 1 in your version)
            0,               # Arg 1: obs_buffer
            0,               # Arg 2: action_buffer
            0,               # Arg 3: reward_buffer
            0,               # Arg 4: logprob_buffer
            0,               # Arg 5: ptr
            0                # Arg 6: done_flag
        )
    )

    def scan_fn(carry, _):
        state, buffer = carry
        state, obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag = vmap_rollout_step(
            state,
            buffer.obs_buffer,
            buffer.action_buffer,
            buffer.reward_buffer,
            buffer.logprob_buffer,
            buffer.ptr,
            buffer.done_flag
        )
        buffer = BatchedBuffer(obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag)
        return (state, buffer), None

    (final_state, final_buffer), _ = jax.lax.scan(
        scan_fn, (init_state, buffer), None, length=settings.steps_per_episode
    )
    return final_state, final_buffer


def general_advantage_estimator(
    settings: TrainingSettings,
    obs_buffer: jax.Array,  # shape: (num_steps, *obs_shape)
    reward_buffer: jax.Array,
    ptr: jax.Array,
    done_flag:jax.Array,
):
    network_settings = settings.network_settings
    lam = settings.gae_lambda
    gamma = settings.gamma

    #num_steps
    N = obs_buffer.shape[0]

    # obtem os "values" por meio do critico
    values = network_settings.critic.apply(network_settings.critic_params, obs_buffer)  # (N,)
    values = cast(jax.Array, values)

    t = jnp.arange(N)

    # mascara que marca os timesteps preenchidos
    #valid_mask = [1, 1, ...,1, 0, 0, ..., 0]
    valid_mask = (t < ptr).astype(jnp.float32)

  
    #done_mask = [0, 0, ..., 1, 0, 0, ..., 0]
    done_mask = jnp.zeros((N,), dtype=jnp.float32)
    done_mask = done_mask.at[jnp.maximum(ptr - 1, 0)].set(
        jnp.where(done_flag, 1.0, 0.0)
    )

    # Transições (N-1)
    values_t = values[:-1]
    next_values = values[1:]
    rewards = reward_buffer[:-1]

    valid_t = valid_mask[:-1]
    valid_next = valid_mask[1:]
    done_t = done_mask[:-1]

    # --- stop propagation correctly ---
    not_done = (1.0 - done_t) * valid_next

    # --- bootstrap (critical for truncation) ---
    # if done → no bootstrap
    # if not done → bootstrap from V(s_T)
    bootstrap_value = jnp.where(
        done_t,
        0.0,
        values[jnp.minimum(ptr, N - 1)]
    )

    def gae_scan_fn(gae_next, inputs):
        reward, value, next_value, nd = inputs

        delta = reward + gamma * next_value * nd - value
        gae = delta + gamma * lam * nd * gae_next

        return gae, gae

    inputs = (rewards, values_t, next_values, not_done)

    _, advantages = jax.lax.scan(
        gae_scan_fn,
        bootstrap_value,
        inputs,
        reverse=True
    )

    returns = advantages + values_t

    # --- mask out padding ---
    advantages = advantages * valid_t
    returns = returns * valid_t

    return advantages, returns


def ppo_loss(
    settings: TrainingSettings,
    batch_obs,          #shape: (num_envs, max_steps, *obs_shape)
    batch_actions,      #shape: (num_envs, max_steps, *action_shape)
    batch_advantages,   #shape: (num_envs, max_steps - 1)
    batch_returns,      #shape: (num_envs, max_steps - 1)
    old_log_probs,      #shape: (num_envs, max_steps)
    clip_eps=0.2,
    c1=0.2,
    c2=0.1,
    min_alpha_beta=0.1,
):
    """
    Calculates the PPO loss.
    """
    batch_advantages = jax.lax.stop_gradient(batch_advantages)
    batch_returns = jax.lax.stop_gradient(batch_returns)

    # dropa 1 step, por conta que os dados do gae tem o shift de 1 para conseguir valores futuros
    batch_obs = jax.lax.stop_gradient(batch_obs[:, :-1, :])
    batch_actions = jax.lax.stop_gradient(batch_actions[:, :-1, :])
    old_log_probs = jax.lax.stop_gradient(old_log_probs[:, :-1])

    # forward 
    networks = settings.network_settings
    logits = cast(jax.Array, networks.actor.apply(networks.actor_params, batch_obs))
    values = networks.critic.apply(networks.critic_params, batch_obs)

    # parametrização
    alpha_logits, beta_logits = jnp.split(logits, 2, axis=-1)
    alpha = jax.nn.softplus(alpha_logits) + min_alpha_beta
    beta  = jax.nn.softplus(beta_logits) + min_alpha_beta

    # logprobs
    clipped_actions = jnp.clip(batch_actions, 1e-6, 1 - 1e-6)

    jax.debug.print("actions.shape = {}", clipped_actions.shape)
    jax.debug.print("alpha.shape = {}", alpha.shape)
    jax.debug.print("beta.shape = {}", beta.shape)
    logprobs = jax.scipy.stats.beta.logpdf(clipped_actions, alpha, beta)
    logprobs = jnp.sum(logprobs, axis=2)

    # ratio
    ratio = jnp.exp(logprobs - old_log_probs)

    # PPO objective
    unclipped = ratio * batch_advantages
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    # value loss
    value_loss = c1 * jnp.mean((batch_returns - values) ** 2)

    # entropy
    entropy = c2 * jnp.mean(beta_entropy(alpha, beta).sum(axis=2))

    return policy_loss + value_loss - entropy



##########################################################################


def grad_metrics(grads, params):
    leaves = jax.tree_util.tree_leaves(grads)

    # norma euclidiana (L2) do gradiente
    grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in leaves]))
    mean_abs_grad = jnp.mean(jnp.concatenate([jnp.ravel(jnp.abs(g)) for g in leaves]))
    max_grad = jnp.max(jnp.concatenate([jnp.ravel(jnp.abs(g)) for g in leaves]))

    num = sum([jnp.sum(jnp.square(g)) for g in leaves])
    den = sum([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)])
    grad_to_param_ratio = jnp.sqrt(num / (den + 1e-12))

    return {
        "grad_norm": grad_norm,
        "mean_abs_grad": mean_abs_grad,
        "max_grad": max_grad,
        "grad_to_param_ratio": grad_to_param_ratio,
    }


##########################################################################


def ppo_train(rng: jax.Array,settings: TrainingSettings):
    """The complete, JIT-compiled training function."""


    def _update_step(carry, _):
        """This is the body of the scan, representing one full update."""
        parameters, optim_state, rng = carry
        rng, initial_state_rng, rollout_rng = jax.random.split(rng, 3)

        #estado inicial 
        rng, initial_state = create_initial_state(initial_state_rng, settings)

        # cria um buffer
        buffer = batched_buffer_create(settings)

        # Faz um rollout (usando a função vetorizada)
        state, buffer= rollout(settings, initial_state,buffer)

        # Vetoriza a função GAE
        vmapped_gae = jax.vmap(
            partial(general_advantage_estimator, settings),
            in_axes=(0, 0, 0, 0)
        )

        # calcula as vantagens (usando a função vetorizada)
        advantages, returns = vmapped_gae(
            buffer.obs_buffer,
            buffer.reward_buffer,
            buffer.ptr,
            buffer.done_flag
        )


        
        # normaliza as vantagens, para prevenir problemas com os gradientes, com as recompensas ruidosas
        advantages_mean = jnp.mean(advantages)
        advantages_std = jnp.std(advantages)
        advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        advantages_std = jnp.maximum(advantages_std, 1e-3)

        #bloqueia o calculo de gradientes para as vantagens
        advantages_std = jax.lax.stop_gradient(advantages_std)
        returns = jax.lax.stop_gradient(returns)



        def loss_fn(par):
            return ppo_loss(
                settings,
                buffer.obs_buffer,
                buffer.action_buffer,
                advantages_std,
                returns,
                buffer.logprob_buffer,
            )

        # calcula os gradientes e atualiza os parametros
        loss_val, grads = jax.value_and_grad(loss_fn)(parameters)
        updates, new_optim_state = settings.optimizer.update(grads, optim_state)
        new_parameters = optax.apply_updates(parameters, updates)

        new_carry = (new_parameters, new_optim_state, rollout_rng)
        grad_info = grad_metrics(grads, parameters)

        return new_carry, {
            "loss": loss_val,
            "ptr": buffer.ptr,
            "success_count": state["success_count"],
            "avg_reward": jnp.mean(buffer.reward_buffer, axis=0),
            **grad_info,
        }

    # loop principal de trainamento, executado por lax.scan
    final_carry, metrics = jax.lax.scan(
        _update_step,
        (settings.network_settings.params, settings.optimizer_state, rng),
        None,
        length=settings.num_episodes,
    )

    return final_carry, metrics


def create_initial_state(rng: jax.Array, settings: TrainingSettings):
    rng, rng1 = jax.random.split(rng)

    # Inicializa os estados para os ambientes em paralelo
    # (num_envs, features_dim)
    num_envs = settings.num_envs
    batched_steps = jnp.zeros((num_envs,))
    batched_success_count = jnp.zeros((num_envs,))
    batched_rng = jax.random.split(rng, num_envs)
    batched_action = jnp.zeros((num_envs, settings.network_settings.action_size))
    batched_obs = jnp.zeros((num_envs, settings.network_settings.obs_size))

    vmapped_get_goal = jax.vmap(partial(get_goal, settings.robot_shared_data))
    batched_goal = vmapped_get_goal(batched_rng)

    mjx_data = mjx.make_data(settings.robot_shared_data.mjx_model)
    batched_mjx_data = jax.tree_util.tree_map(
        lambda x: jax.numpy.repeat(x[None], num_envs, axis=0),
        mjx_data
    )

    #estado inicial 
    return rng1, {
        "rng":batched_rng,
        "step":batched_steps,
        "goal":batched_goal,
        "obs":batched_obs,
        "action": batched_action,
        "mjx_data": batched_mjx_data,
        "success_count":batched_success_count, 
    }
   

