from os import wait
import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import mathutils as mu
from flax import struct
from functools import partial
from typing import Dict, Any, cast, Tuple, Callable, Self
from mathutils import  beta_entropy, ema, stdNormalize
from robot import get_goal
from mujoco import mjx
from dataclassutils import RunningParameters, TrainingSettings, BatchedBuffer



def rollout_step(
    progress,
    step_fn,
    runpar: RunningParameters,
    state: Dict[str, Any],
    obs_buffer: jax.Array,
    action_buffer: jax.Array,
    reward_buffer: jax.Array,
    logprob_buffer: jax.Array,
    ptr: jax.Array,
    done_flag: jax.Array,
    stop_flag: jax.Array,
):
    """
    Dá um step de rollout para um único ambiente

    Parameters
    ----------

    pipeline: StateMonad
        Pipeline de ajuste de objetivo, tomada de ação pelo agente, coleta de recompensas e formação do espaço de observação.
    """

    # Caso stop_flag esteja como False
    def do_step(carry):
        _state, _obs_buffer, _action_buffer, _reward_buffer, _logprob_buffer, _ptr, _done_flag, _stop_flag = carry
        rng, step_rng = jax.random.split(_state["rng"])

        # executa o ambiente
        _state = {**_state, "rng": step_rng}  #atualiza o rng
        _state, data = step_fn(progress, _state, runpar)

        # adiciona o dado no buffer
        _obs_buffer, _action_buffer, _reward_buffer, _logprob_buffer, _ptr = BatchedBuffer.push(
            _obs_buffer,
            _action_buffer,
            _reward_buffer,
            _logprob_buffer,
            data["obs"],
            data["action"],
            data["reward"],
            data["logprob"],
            _ptr
        )

        # faz o update do rng
        _state["rng"] = rng
        _done_flag = data["done"] > 0.5 
        _stop_flag = _done_flag | (_state["step"] >= _reward_buffer.shape[0])

        return _state, _obs_buffer, _action_buffer, _reward_buffer, _logprob_buffer, _ptr, _done_flag, _stop_flag

    # Caso stop_flag esteja como True
    def no_step(carry):  #
        return carry

    return jax.lax.cond(
        stop_flag, no_step, do_step, (state, obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag, stop_flag)
    )


def rollout(
    progress,
    settings: TrainingSettings,
    init_state: Dict[str, Any],
    buffer: BatchedBuffer,
    runpar: RunningParameters,
):
    # Match the structure of your 'state' dictionary exactly
    state_in_axes = {
        'action': 0,
        'goal': {
            'goal_position_coordinates': 0,
            'goal_position_velocities': 0,
            'goal_orientation_coordinates': 0, # or 0 if batched
        },
        'mjx_data': 0, 
        'obs': 0,
        'rng': 0,
        'step': 0,
        'success_count':0,
        "err":0
    }
    state_out_axes = {**state_in_axes, 'obs_stats': 0}
    vmap_rollout_step = jax.vmap(
        partial(rollout_step, progress, settings.step_fn, runpar),
        in_axes=(
            state_in_axes,  # Arg 0: state (was Arg 1 in your version)
            0,               # Arg 1: obs_buffer
            0,               # Arg 2: action_buffer
            0,               # Arg 3: reward_buffer
            0,               # Arg 4: logprob_buffer
            0,               # Arg 5: ptr
            0,               # Arg 6: done_flag
            0,               # Arg 7: stop_flag
        ),
    )

    def scan_fn(carry, _):
        state, buffer = carry
        state, obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag, stop_flag = vmap_rollout_step(
            state,
            buffer.obs_buffer,
            buffer.action_buffer,
            buffer.reward_buffer,
            buffer.logprob_buffer,
            buffer.ptr,
            buffer.done_flag,
            buffer.stop_flag
        )

        buffer = BatchedBuffer(obs_buffer, action_buffer, reward_buffer, logprob_buffer, ptr, done_flag, stop_flag)
        return (state, buffer), None

    (final_state, final_buffer), _ = jax.lax.scan(
        scan_fn, (init_state, buffer), None, length=settings.steps_per_episode
    )
    return final_state, final_buffer


def general_advantage_estimator(
    settings: TrainingSettings,
    obs_buffer: jax.Array,    # Shape: (N + 1, *obs) 
    reward_buffer: jax.Array, # Shape: (N + 1,)
    ptr: jax.Array,           # Valor entre 0 e N
    done_flag: jax.Array,     # Flag de terminação no passo ptr-1
):
    gamma = settings.gamma
    lam = settings.gae_lambda

    N = reward_buffer.shape[0] -1

    values = settings.network_settings.critic.apply(
        settings.network_settings.critic_params, obs_buffer
    )
    values = jnp.squeeze(cast(jax.Array, values))

    # Se ptr=N, values[ptr] acessa o índice N (o último do buffer N+1)
    bootstrap_value = jnp.where(done_flag, 0.0, values[ptr])

    def gae_scan_fn(carry, t):
        gae_next, next_val_from_carry = carry
        
        # Injeta o bootstrap exatamente na borda da pilha de dados
        is_last_step = t == (ptr - 1)
        actual_next_val = cast(jax.Array, jnp.where(is_last_step, bootstrap_value, next_val_from_carry))
        
        # O done_buffer[t_idx] aqui só seria necessário se você tivesse 
        # múltiplos episódios dentro do mesmo buffer. No seu caso de "stack",
        # apenas a done_flag no final importa para o bootstrap.
        
        delta = reward_buffer[t] + gamma * actual_next_val - values[t]
        gae = delta + gamma * lam * gae_next
        
        return (gae, values[t]), gae

    #Scan Reverso sobre os N passos de transição
    initial_carry = (0.0, 0.0)
    _, advantages = jax.lax.scan(
        gae_scan_fn,
        initial_carry,
        jnp.arange(N),
        reverse=True
    )

    valid_mask = (jnp.arange(N) < ptr).astype(jnp.float32)
    advantages = advantages * valid_mask
    returns = (advantages + values[:-1]) * valid_mask

    return advantages, returns


def ppo_loss(
    params,
    settings: TrainingSettings,
    batch_obs,          #shape: (num_envs, max_steps +1, *obs_shape)
    batch_actions,      #shape: (num_envs, max_steps +1, *action_shape)
    batch_advantages,   #shape: (num_envs, max_steps)
    batch_returns,      #shape: (num_envs, max_steps)
    old_log_probs,      #shape: (num_envs, max_steps +1)
    clip_eps=0.2,
    c1=0.5,
    c2=0.05,
    min_alpha_beta=1.0,
):
    """
    Calculates the PPO loss.
    """
    batch_advantages = jax.lax.stop_gradient(batch_advantages)
    batch_returns = jax.lax.stop_gradient(batch_returns)

    # elimina a contagem do gradiente nestas variaveis
    batch_obs = jax.lax.stop_gradient(batch_obs[:, :-1, :])
    batch_actions = jax.lax.stop_gradient(batch_actions[:, :-1, :])
    old_log_probs = jax.lax.stop_gradient(old_log_probs[:, :-1])

    # forward 
    networks = settings.network_settings
    logits = cast(jax.Array, networks.actor.apply(params[0], batch_obs))
    values = cast(jax.Array, networks.critic.apply(params[1], batch_obs))


    # parametrização
    alpha_logits, beta_logits = jnp.split(logits, 2, axis=-1)
    alpha = jax.nn.softplus(alpha_logits) + min_alpha_beta
    beta  = jax.nn.softplus(beta_logits) + min_alpha_beta

    # logprobs
    clipped_actions = jnp.clip(batch_actions, 1e-6, 1 - 1e-6)

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

    total_loss = policy_loss + value_loss - entropy
    return total_loss, {"entropy": entropy, "policy_loss": policy_loss, "value_loss": value_loss}



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


def ppo_train(rng: jax.Array, settings: TrainingSettings):
    """The complete, JIT-compiled training function."""


    def _update_running_parameters(state, buffer, runpar: RunningParameters):
       
        # Transforma qualquer contagem > 0 em 1 (sucesso) ou 0 (falha)
        # A média dará um valor entre  0.0 e 1.0 (0% a 100%)
        success_per_env = state["success_count"] > 0
        success_rate = jnp.mean(success_per_env)

        #atualiza as estatisticas considerando as novas observações
        runpar = runpar.update(buffer.obs_buffer, success_rate)
        return runpar

    def _collect_dataset(state, buffer, runpar: RunningParameters):

        # Faz um rollout (usando a função vetorizada)
        state, buffer = rollout(runpar.progress.value, settings, state, buffer, runpar)

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
            buffer.done_flag,
        )

        # normaliza as vantagens, para prevenir problemas com os gradientes, com as recompensas ruidosas
        advantages = mu.stdNormalize(advantages)

        #bloqueia o calculo de gradientes para as vantagens
        advantages = jax.lax.stop_gradient(advantages)
        returns = jax.lax.stop_gradient(returns)
    
        return state, buffer, advantages, returns
    
    def _train(
        parameters,
        optimizer_state,
        buffer: BatchedBuffer,
        advantages: jax.Array,
        returns:jax.Array
    ):

        def _train_step(carry,_):
            _parameters, _optimizer_state = carry
    
            def loss_fn(par):
                return ppo_loss(
                    par,
                    settings,
                    buffer.obs_buffer,
                    buffer.action_buffer,
                    advantages,
                    returns,
                    buffer.logprob_buffer,
                )

            # calcula os gradientes e atualiza os parametros
            (loss_val, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(_parameters)
            updates, new_optim_state = settings.optimizer.update(grads, _optimizer_state)
            new_parameters = optax.apply_updates(_parameters, updates)

            new_carry = (new_parameters, new_optim_state)
            grad_info = grad_metrics(grads, _parameters)

            return new_carry, {
                "loss": loss_val,
                **grad_info,
                **aux_metrics,
            }

        (new_parameters, new_optim_state), metrics = jax.lax.scan(
            _train_step,
            (parameters, optimizer_state),
            jnp.arange(settings.num_episodes),
        )

        return new_parameters, new_optim_state, metrics

        
    def _new_goal_step(carry, _):
        """This is the body of the scan, representing one full update."""
        rng, runpar, params, optim_state = carry
        rng, initial_state_rng = jax.random.split(rng)

        #estado inicial 
        rng, initial_state = create_initial_state(initial_state_rng, runpar.progress.value, settings)

        # cria um buffer
        buffer = BatchedBuffer.init(settings)
        state, buffer, advantages, returns = _collect_dataset(initial_state, buffer, runpar)
        params, optim_state, metrics = _train(params, optim_state, buffer, advantages, returns)

        runpar = _update_running_parameters(state, buffer, runpar)

        return (rng, runpar, params, optim_state),  metrics



    # loop principal de trainamento, executado por lax.scan
    runpar = RunningParameters.init((settings.network_settings.obs_size, ))
    final_carry, metrics = jax.lax.scan(
        _new_goal_step,
        (rng, runpar, settings.network_settings.params, settings.optimizer_state),
        jnp.arange(settings.robot_shared_data.range_config.num_values),
    )

    return final_carry, metrics


def create_initial_state(rng: jax.Array, progress, settings: TrainingSettings):
    rng, rng1 = jax.random.split(rng)

    # Inicializa os estados para os ambientes em paralelo
    # (num_envs, features_dim)
    num_envs = settings.num_envs
    batched_steps = jnp.zeros((num_envs,))
    batched_success_count = jnp.zeros((num_envs,))
    batched_rng = jax.random.split(rng, num_envs)
    batched_action = jnp.zeros((num_envs, settings.network_settings.action_size))
    batched_obs = jnp.zeros((num_envs, settings.network_settings.obs_size))

    mjx_data = mjx.make_data(settings.robot_shared_data.mjx_model)
    batched_mjx_data = jax.tree_util.tree_map(
        lambda x: jax.numpy.repeat(x[None], num_envs, axis=0),
        mjx_data
    )

    vmapped_get_goal = jax.vmap(partial(get_goal, settings.robot_shared_data, progress))
    batched_rng, batched_goal = vmapped_get_goal(batched_rng)

    batched_err = jnp.ones((num_envs, ))*jnp.inf

    #estado inicial 
    return rng1, {
        "rng":batched_rng,
        "step":batched_steps,
        "goal":batched_goal,
        "obs":batched_obs,
        "action": batched_action,
        "mjx_data": batched_mjx_data,
        "success_count":batched_success_count, 
        "err": batched_err,
    }
   

