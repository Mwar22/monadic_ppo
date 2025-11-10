import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from enviroment import Data, State, EnvState
from functools import partial
from typing import Dict, Any
from jax.scipy.special import gammaln, digamma


def rollout(pipeline: EnvState, init_state: Dict[str, Any], num_steps: int):
    def scan_fn(state, _):
        rng, step_rng = jax.random.split(state["rng"])

        # executa o ambiente
        new_state, data = pipeline.run({**state, "rng": step_rng})

        #mantem o shape da ação constante
        #jax.debug.print("action shape: {}", new_state["action"].shape)
        new_state["action"] = jnp.reshape(new_state["action"], (6,))

        # faz o update do rng
        new_state["rng"] = rng
        return new_state, data

    final_state, trajectory = jax.lax.scan(
        scan_fn, init_state, None, length=num_steps
    )
    return final_state, trajectory

def general_advantage_estimator(
    critic: nn.Module, critic_params, data, final_state_obs, lam, gamma
):
    obs = data["obs"]["obs_history"]
    dones = data["final_data"]["done"]
    rewards = data["final_data"]["reward"]

    # Cria all_obs, que é (T_steps, obs_dim) + (1, obs_dim)
    all_obs = jnp.concatenate(
        [obs, jnp.expand_dims(final_state_obs, axis=0)], axis=0
    )
    #V(obs_0)...V(obs_T)
    values = critic.apply(critic_params, all_obs)
    
    #V(obs_0)...V(obs_{T-1})
    values_t = values[:-1] 

    # next_values é V(obs_1)...V(obs_T)
    next_values = values[1:]

    def gae_scan_fn(carry, step_inputs):
        gae_next = carry
        reward, value, next_value, done = [jnp.squeeze(x) for x in step_inputs]

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae_next

        # next_carry, y
        return gae, gae

    # faz do ultimo para o primeiro
    inputs = (rewards, values_t, next_values, dones)

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


@partial(
    jax.jit,
    static_argnames=(
        "pipeline",
        "policy",
        "critic",
        "optimizer",
        "num_updates",
        "num_steps_per_update",
        "gamma",
        "lam",
    ),
)
def ppo_train(
    pipeline,
    params,
    opt_state,
    init_envs_states,
    rng,
    policy,
    critic,
    optimizer,
    num_updates,
    num_steps_per_update,
    gamma,
    lam,
):
    """The complete, JIT-compiled training function."""

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
        final_states, trajectory = vmapped_rollout(env_states, num_steps_per_update)

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
        return new_carry, {"loss": loss_val, "reward": trajectory["final_data"]["reward"]}

    # loop principal de trainamento, executado por lax.scan
    final_carry, metrics = jax.lax.scan(
        _update_step,
        (params, opt_state, init_envs_states, rng),
        None,
        length=int(num_updates),
    )

    return final_carry, metrics
