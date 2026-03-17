from os import wait
import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from enviroment import Data, State, StateMonad
from functools import partial
from typing import Dict, Any, cast
from jax.scipy.special import gammaln, digamma
from mathutils import cont_sample_beta, beta_entropy, shift_array
from robot import RobotSharedData, get_goal
from mujoco import mjx


@struct.dataclass
class BatchedBuffer:
    obs_buffer: jax.Array       # (num_envs, max_steps, *obs_shape)
    action_buffer: jax.Array    # (num_envs, max_steps, *action_shape)
    reward_buffer: jax.Array    # (num_envs, max_steps)
    logprob_buffer: jax.Array   # (num_envs, max_steps)
    ptr: jax.Array  # (num_envs,)
    done_flag: jax.Array  # (num_envs,)

    def __str__(self):
        return f"obs_buffer: {self.obs_buffer.shape}\n \
        action_buffer: {self.action_buffer.shape}\n \
        reward_buffer: {self.reward_buffer.shape}\n \
        logprob_buffer: {self.logprob_buffer.shape}\n \
        ptr: {self.ptr.shape}\n \
        done_flag: {self.done_flag.shape}"


def batched_buffer_create(num_envs, max_steps, obs_shape, action_shape):
    return BatchedBuffer(
        jnp.zeros((num_envs, max_steps, *obs_shape)),
        jnp.zeros((num_envs, max_steps, *action_shape)),
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
    step_fn,
    init_state: Dict[str, Any],
    num_steps: int,
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
        partial(rollout_step, step_fn),
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
        scan_fn, (init_state, buffer), None, length=num_steps
    )
    return final_state, final_buffer


def general_advantage_estimator(
    critic: nn.Module,
    critic_params,
    lam,
    gamma,
    obs_buffer: jax.Array,
    reward_buffer: jax.Array,
    ptr: jax.Array,
):
    # garante que o checador de tipos não reclame
    values = cast(jax.Array, critic.apply(critic_params, obs_buffer))

    # desloca para que os zeros não utilizados fiquem no começo
    values = shift_array(values, ptr)
    reward_buffer = shift_array(reward_buffer, ptr)

    # V(t) and V(t+1)
    values_t = values[:-1]
    next_values = values[1:]
    rewards = reward_buffer[:-1] # Usualmente alinham com values_t

    def gae_scan_fn(gae_next, step_inputs):
        reward, value, next_value = step_inputs
        
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = reward + gamma * next_value - value
        gae = delta + gamma * lam * gae_next

        return gae, gae

    # entradas para scan (T-1 steps)
    inputs = (rewards, values_t, next_values)

    _, advantages = jax.lax.scan(
        gae_scan_fn, 
        jnp.zeros((), dtype=jnp.float32), 
        inputs, 
        reverse=True
    )

    returns = advantages + values_t
    return advantages, returns


def ppo_loss(
    params,
    rsd: RobotSharedData,
    batch_obs,
    batch_actions,
    batch_advantages,
    batch_returns,
    old_log_probs,
    clip_eps=0.2,
    c1=0.2,
    c2=0.1,
    min_alpha_beta=0.1,
):
    """
    Calculates the PPO loss.
    """
    # dropa 1 step, por conta que os dados do gae tem o shift de 1 para conseguir valores futuros
    batch_obs = batch_obs[:, :-1, :]
    batch_actions = batch_actions[:, :-1, :]
    old_log_probs = old_log_probs[:, :-1]

    logits = rsd.actor.apply(params[0], batch_obs)
    logits = cast(jax.Array, logits)
    values = rsd.critic.apply(params[1], batch_obs)

    # Map actions -> their log probabilities under the current policy
    alpha = jnp.clip(jax.nn.softplus(logits), min_alpha_beta, 1000.0)
    beta = jnp.clip(jax.nn.softplus(1.0 - logits), min_alpha_beta, 1000.0)

    # Compute log probability of the *stored* actions
    clipped_batch_actions = jnp.clip(batch_actions, 1e-6, 1.0 - 1e-6)
    logprobs = jax.scipy.stats.beta.logpdf(clipped_batch_actions, alpha, beta)
    logprobs = jnp.sum(logprobs, axis=2)  # (batch, max_steps)

    ratio = jnp.exp(logprobs - old_log_probs)
  
    # Clipped surrogate objective
    unclipped = ratio * batch_advantages
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    # Value loss (MSE between returns and predicted values)
    value_loss = c1 * jnp.mean((batch_returns - values) ** 2)

    # Entropy bonus
    alpha = jnp.clip(jax.nn.softplus(logits), min_alpha_beta, 1000.0)
    beta = jnp.clip(jax.nn.softplus(1.0 - logits), min_alpha_beta, 1000.0)

    entropy = c2 * jnp.mean(beta_entropy(alpha, beta).sum(axis=1))

    total_loss = policy_loss + value_loss - entropy
    return total_loss


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


def ppo_train(
    rsd: RobotSharedData,
    params,
    step_fn,
    optimizer,
    optim_state,
    rng,
    num_envs,
    episodes: int,
    obs_dim,
    action_dim,
    max_steps_per_episode: int,
    gamma,
    lam,
):
    """The complete, JIT-compiled training function."""

    
    

    def _update_step(carry, _):
        """This is the body of the scan, representing one full update."""
        parameters, optim_state, rng = carry
        rng, rollout_rng, rng_loss = jax.random.split(rng, 3)

        #estado inicial 
        rng, initial_state = create_initial_state(rsd, rng, num_envs, obs_dim, action_dim)

        # cria um buffer
        buffer = batched_buffer_create(num_envs, max_steps_per_episode, rsd.obs_shape, rsd.action_shape)

        # Faz um rollout (usando a função vetorizada)
        final_state, final_buffer= rollout(
            step_fn, initial_state, max_steps_per_episode, buffer
        )

        # Vetoriza a função GAE
        vmapped_gae = jax.vmap(
            partial(general_advantage_estimator, rsd.critic, parameters[1], lam, gamma),
            in_axes=(0, 0, 0)
        )

        # calcula as vantagens (usando a função vetorizada)
        advantages, returns = vmapped_gae(
            buffer.obs_buffer,
            buffer.reward_buffer,
            buffer.ptr
        )


        #bloqueia o calculo de gradientes para as vantagens
        advantages = jax.lax.stop_gradient(advantages)

        # normaliza as vantagens, para prevenir problemas com os gradientes, com as recompensas ruidosas
        advantages_mean = jnp.mean(advantages)
        advantages_std = jnp.std(advantages)
        advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        advantages_std = jnp.maximum(advantages_std, 1e-3)

        # Prepara o batch para update
        # Aplica um flatten nas dimensões (num_envs, num_steps) em um único batch
        def flatten(x):
            return x.reshape(-1, *x.shape[2:])

        batch_obs = flatten(final_buffer.obs_buffer)
        batch_actions = flatten(final_buffer.action_buffer)
        batch_old_log_probs = flatten(final_buffer.logprob_buffer)
        batch_advantages = flatten(advantages)
        batch_returns = flatten(returns)

        def loss_fn(par):
            return ppo_loss(
                par,
                rsd,
                final_buffer.obs_buffer,
                final_buffer.action_buffer,
                advantages,
                returns,
                final_buffer.logprob_buffer,
            )

        # calcula os gradientes e atualiza os parametros
        loss_val, grads = jax.value_and_grad(loss_fn)(parameters)
        updates, new_optim_state = optimizer.update(grads, optim_state)
        new_parameters = optax.apply_updates(parameters, updates)

        new_carry = (new_parameters, new_optim_state, rollout_rng)
        grad_info = grad_metrics(grads, parameters)

        return new_carry, {
            "loss": loss_val,
            "avg_reward": jnp.mean(final_buffer.reward_buffer, axis=0),
            **grad_info,
        }

    # loop principal de trainamento, executado por lax.scan
    final_carry, metrics = jax.lax.scan(
        _update_step,
        (params, optim_state, rng),
        None,
        length=int(episodes),
    )

    return final_carry, metrics


def create_initial_state(rsd: RobotSharedData, rng, num_envs, obs_dim, action_dim):
    rng, rng1 = jax.random.split(rng)
    # Inicializa os estados para os ambientes em paralelo
    # (num_envs, features_dim)
    batched_steps = jnp.zeros((num_envs,))
    batched_success_count = jnp.zeros((num_envs,))
    batched_rng = jax.random.split(rng, num_envs)
    batched_action = jnp.zeros((num_envs, action_dim))
    batched_obs = jnp.zeros((num_envs, obs_dim))

    vmapped_get_goal = jax.vmap(partial(get_goal, rsd))
    batched_goal = vmapped_get_goal(batched_rng)

    mjx_data = mjx.make_data(rsd.mjx_model)
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
   
