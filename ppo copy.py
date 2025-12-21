import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from enviroment import Data, State, StateMonad
from functools import partial
from typing import Dict, Any
from jax.scipy.special import gammaln, digamma


def rollout(pipeline: StateMonad, init_state: Dict[str, Any], num_steps: int):

    #para capturar o shape dos estados e dos dados
    dummy_rng, new_rng = jax.random.split(init_state["rng"])
    _, dummy_data = pipeline.run({**init_state, "rng": dummy_rng})
    

    zeros_data = jax.tree.map(lambda x: jnp.zeros_like(x), dummy_data)
    init_state["rng"] = new_rng

    def scan_fn(carry, _):
        state, done_flag = carry 

        def step_fn(state):
            rng, step_rng = jax.random.split(state["rng"])

            # executa o ambiente
            new_state, data = pipeline.run({**state, "rng": step_rng})

            # faz o update do rng
            new_state["rng"] = rng
            return new_state, data
        
        def skip_fn(state):
            return state, zeros_data
        

        new_state, data = jax.lax.cond(done_flag, skip_fn, step_fn, state)

        is_done_step = data["done"] > 0.5
        new_done_flag = jnp.logical_or(done_flag, is_done_step)

        return (new_state, new_done_flag), data


    (final_state, _), trajectory = jax.lax.scan(
        scan_fn, (init_state, jnp.array(False)), None, length=num_steps
    )
    return final_state, trajectory


def rollout_autoreset(pipeline: StateMonad, reset_fn, init_state: Dict[str, Any], num_steps: int):

    def scan_fn(carry_state, _):

        rng, step_rng, reset_rng = jax.random.split(carry_state["rng"], 3)

        step_state, data = pipeline.run({**carry_state, "rng": step_rng})
        
        # Garante o shape da ação
        step_state["action"] = jnp.reshape(step_state["action"], (6,))

        #definição das funções condicionais (só reseta se necessário)
        def get_reset_state(_):
             # Só roda se done=True
             reset_state = reset_fn({**step_state, "rng": reset_rng})
             return reset_state

        def get_step_state(_):
             # Só roda se done=False
             return step_state

        # Escolhe qual função executar para definir o próximo estado
        done = data["done"] > 0.5 # Booleano
        next_carry_state = jax.lax.cond(done, get_reset_state, get_step_state, operand=None)

        # Atualiza o RNG principal para o próximo loop
        next_carry_state["rng"] = rng
        return next_carry_state, data

    final_state, trajectory = jax.lax.scan(
        scan_fn, init_state, None, length=num_steps
    )
    
    return final_state, trajectory


def general_advantage_estimator(rewards, values, lam, gamma):
   
    
    #V(obs_0)...V(obs_{T-1})
    values_t = values[:-1] 

    # next_values é V(obs_1)...V(obs_T)
    next_values = values[1:]

    def gae_scan_fn(carry, step_inputs):
        gae_next = carry
        reward, value, next_value= [jnp.squeeze(x) for x in step_inputs]

        delta = reward + gamma * next_value  - value
        gae = delta + gamma * lam * gae_next

        # next_carry, y
        return gae, gae

    # faz do ultimo para o primeiro
    inputs = (rewards, values_t, next_values)

    _, advantages = jax.lax.scan(gae_scan_fn, jnp.asarray(0.0, dtype=jnp.float32), inputs, reverse=True)

    returns = advantages + values_t
    return advantages, returns

def cont_sample_beta(logits: jax.Array, rng: jax.Array, min_alpha_beta=0.1):
    """
    Sample continuous actions in [0,1] using independent Beta distributions
    parameterized by logits.
    
    Args:
        logits: shape (action_dim,), any real numbers
        rng: JAX PRNGKey
        min_alpha_beta: minimum value for alpha and beta to avoid numerical issues
    
    Returns:
        action: shape (action_dim,)
        logprob: shape (action_dim,)
    """

    # mapeia os logits para parametros positivos para serem utilizados na distribuição beta
    alpha = jnp.clip(jax.nn.softplus(logits), min_alpha_beta, 1000.0)
    beta = jnp.clip(jax.nn.softplus(1 - logits), min_alpha_beta, 1000.0)

    #separa o rng para amostras independentes
    rng, subkey = jax.random.split(rng)
    actions = jax.random.beta(subkey, alpha, beta)

    # Clip actions to be just inside (0, 1) to avoid -inf logpdf
    clipped_actions = jnp.clip(actions, 1e-6, 1.0 - 1e-6)

    #logprob para cada dimensão
    logprobs = jax.scipy.stats.beta.logpdf(clipped_actions, alpha, beta)
    return actions, jnp.sum(logprobs, axis=-1)

def beta_entropy(alpha, beta):
    lnB = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    H = lnB - (alpha - 1) * digamma(alpha) - (beta - 1) * digamma(beta) + (alpha + beta - 2) * digamma(alpha + beta)
    return H

def ppo_loss(
    params,
    policy,
    critic,
    batch_obs,
    batch_actions,
    batch_advantages,
    batch_returns,
    batch_old_log_probs,
    rng,
    clip_eps=0.2,
    c1=0.2,
    c2=0.1,
    min_alpha_beta=0.1
):
    """
    Calculates the PPO loss.
    """
    # Get policy logits and critic values for the batch of observations
    logits = policy.apply(params["policy"], batch_obs)
    values = critic.apply(params["critic"], batch_obs)

    # Map actions -> their log probabilities under the current policy
    alpha = jnp.clip(jax.nn.softplus(logits), min_alpha_beta, 1000.0)
    beta = jnp.clip(jax.nn.softplus(1.0 - logits), min_alpha_beta, 1000.0)

    # Compute log probability of the *stored* actions
    clipped_batch_actions = jnp.clip(batch_actions, 1e-6, 1.0 - 1e-6)
    logprobs = jax.scipy.stats.beta.logpdf(clipped_batch_actions, alpha, beta)
    logprobs = jnp.sum(logprobs, axis=1)  # (batch,)

    ratio = jnp.exp(logprobs - batch_old_log_probs)

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

    #norma euclidiana (L2) do gradiente
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
        "grad_to_param_ratio": grad_to_param_ratio
    }

##########################################################################


def ppo_train(
    pipeline,
    step_fn,
    reset_fn,
    obs_shape,
    params,
    opt_state,
    init_envs_states,
    rng,
    policy,
    critic,
    optimizer,
    episodes: int,
    max_steps_per_episode: int,
    gamma,
    lam,
):
    """The complete, JIT-compiled training function."""

    #(state = {
    #   "reward": jnp.array( max_steps_per_episode),
    #   "value": jnp.array( max_steps_per_episode),
    #   "ptr": 0,
    #   "count":0
    # }, None))
    buffer  = StateMonad.pure(None)

    state, data = step_fn({**carry_state, "rng": step_rng})

    def buffer_push(data, new_value):
        def fn(state):
            ptr = state["ptr"]
            count = state["count"]
            reward = state["reward"]
            value = state["value"]
            loss = state["loss"]

            reward = reward.at[ptr].set(data["final_data"]["reward"])
            value = value.at[ptr].set(new_value)

            ptr += 1
            count += 1

            return {"rewards": reward,"values": value, "loss": loss, "ptr": ptr, "count": count}, None
        return StateMonad(fn)
    
    def buffer_reset():
        def fn(state):
            new_state = {
                "rewards": jnp.zeros(max_steps_per_episode),
                "values": jnp.zeros(max_steps_per_episode),
                "loss": jnp.zeros(max_steps_per_episode),
                "ptr": 0,
                "count": 0
            }
            return new_state, None
        return StateMonad(fn)

    def scan_fn(carry, _):

            carry_state, _  = carry
            rng, step_rng, reset_rng = jax.random.split(carry_state["rng"], 3)

            
            def rollout_running():
                """"""
                state, data = step_fn({**carry_state, "rng": step_rng})
                obs = data["obs"]["obs_history"]
                rewards = data["final_data"]["reward"]

                #aplica a observação na rede do critico para obter o valor
                value = critic.apply(params["critic"], obs)

                #salva no buffer
                buffer.bind(lambda obs_shape: 
                    StateMonad(lambda s:
                        (s, obs.shape)
                    )
                )
                buffer.bind(buffer_push(data, value))

                # faz o update do rng
                state["rng"] = rng
                return state, data
            
            def rollout_done():
                #calcula o gae
                buffer_state, _ = buffer.run({
                    "rewards": jnp.zeros(max_steps_per_episode),
                    "values": jnp.zeros(max_steps_per_episode),
                    "ptr": 0,
                    "count": 0
                })

                #calcula as vantagens
                advantages, returns = general_advantage_estimator(
                    buffer_state["rewards"],
                    buffer_state["values"],
                    lam,
                    gamma
                )

                #normaliza as vantagens, para prevenir problemas com os gradientes, com as recompensas ruidosas
                advantages_mean = jnp.mean(advantages)
                advantages_std = jnp.std(advantages)
                advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
                advantages_std = jnp.maximum(advantages_std, 1e-3)
            
                buffer.bind(buffer_reset())
                return carry

            # Escolhe qual função executar para definir o próximo estado
            done = carry_state["done"] > 0.5 # Booleano
            state, data = jax.lax.cond(done, rollout_done, rollout_running, operand=None)

            # Atualiza o RNG principal para o próximo loop
            state["rng"] = rng
            return state, data

    start_carry = (init_envs_states, None)
    final_state, trajectory = jax.lax.scan(
        scan_fn, start_carry, None, length=max_steps_per_episode * episodes
    )


















    # Vetoriza a função de rollout, para rodar de forma paralela entre os ambientes
    # rollout(pipeline, init_state, num_steps)
    vmapped_rollout = jax.vmap(
        partial(rollout, pipeline),
        in_axes=(0, None),
        out_axes=0,
    )

    # Vetoriza a função GAE
    # general_advantage_estimator (critic, critic_params, data, lam, gamma)
    def gae_for_vmap(data, final_state_obs):
        return general_advantage_estimator(critic, params["critic"], data, final_state_obs, lam, gamma)

    vmapped_gae = jax.vmap(gae_for_vmap)

    def _update_step(carry, _):
        """This is the body of the scan, representing one full update."""
        params, opt_state, env_states, rng = carry
        rng, rollout_rng, rng_loss = jax.random.split(rng, 3)

        # Faz um rollout (usando a função vetorizada)
        final_states, trajectory = vmapped_rollout(env_states, max_steps_per_episode)

        # calcula as vantagens (usando a função vetorizada)
        final_state_obs_history = final_states["obs_history"]
        advantages, returns = vmapped_gae(trajectory, final_state_obs_history)

        #normaliza as vantagens, para prevenir problemas com os gradientes, com as recompensas ruidosas
        advantages_mean = jnp.mean(advantages)
        advantages_std = jnp.std(advantages)
        advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        advantages_std = jnp.maximum(advantages_std, 1e-3)

        # Prepara o batch para update
        # Aplica um flatten nas dimensões (num_envs, num_steps) em um único batch
        def flatten(x):
            return x.reshape(-1, *x.shape[2:])
        

        # shape (num_envs, obs_dim)
        # obs_0 ... obs_{T-1}
        initial_obs_history = env_states["obs_history"]

    
        # precisamos de um batch com shape shape (num_envs, num_steps, obs_dim)
        # contendo: [ obs_{t-1}, obs_0, obs_1, ..., obs_{T-2} ]
        # alinhando com a tragetória: [ action_0, action_1, ..., action_{T-1}
        trajectory_obs_history = trajectory["obs"]["obs_history"]

        batch_obs = jnp.concatenate(
            [
                jnp.expand_dims(initial_obs_history, axis=1), # (num_envs, 1, obs_dim)
                trajectory_obs_history[:, :-1]                # (num_envs, num_steps-1, obs_dim)
            ],
            axis=1
        )
        batch_obs = flatten(batch_obs)
        
        # The rest of the data is now correctly aligned
        batch_actions = flatten(trajectory["action"])
        batch_old_log_probs = flatten(trajectory["logprob"])
        batch_advantages = flatten(advantages)
        batch_returns = flatten(returns)

        def loss_fn(params):
            return ppo_loss(
                params,
                policy,
                critic,
                batch_obs,
                batch_actions,
                batch_advantages,
                batch_returns,
                batch_old_log_probs,
                rng_loss
            )

        # calcula os gradientes e atualiza os parametros
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        new_carry = (new_params, new_opt_state, final_states, rollout_rng)
        grad_info = grad_metrics(grads, params)

        return new_carry, {"loss": loss_val, "reward": trajectory["final_data"]["reward"], **grad_info}

    # loop principal de trainamento, executado por lax.scan
    final_carry, metrics = jax.lax.scan(
        _update_step,
        (params, opt_state, init_envs_states, rng),
        None,
        length=int(episodes),
    )

    return final_carry, metrics
