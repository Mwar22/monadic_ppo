import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from enviroment import Data, State, EnvState
from functools import partial


def rollout(
    pipeline: EnvState,
    init_state: State,
    num_steps: int,
):
    rng, _ = jax.random.split(init_state.rng)

    def scan_fn(state: State, _):
        new_state, data = pipeline.run(state)
        return new_state, data

    final_state, trajectory = jax.lax.scan(
        scan_fn, State(rng, init_state.data), None, length=num_steps
    )
    return final_state, trajectory


def general_advantage_estimator(
    critic: nn.Module, critic_params, data: Data, lam, gamma
):
    obs, rewards, dones = data.obs, data.reward, data.done

    values = critic.apply(critic_params, obs)
    values_t = values[:-1]
    next_values = values[1:]

    def gae_scan_fn(carry, step_inputs):
        gae_next = carry
        reward, value, next_value, done = step_inputs

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae_next

        # next_carry, y
        return gae, gae

    # faz do ultimo para o primeiro
    inputs = (rewards[:-1], values_t, next_values, dones[:-1])
    _, advantages = jax.lax.scan(gae_scan_fn, 0.0, inputs, reverse=True)

    returns = advantages + values_t
    return advantages, returns


def ppo_loss(
    params,
    policy,
    critic,
    batch_obs,
    batch_actions,
    batch_advantages,
    batch_returns,
    batch_old_log_probs,
    clip_eps=0.2,
    c1=0.5,
    c2=0.01,
):
    """
    Calculates the PPO loss.
    """
    # Get policy logits and critic values for the batch of observations
    logits = policy.apply(params["policy"], batch_obs)
    values = critic.apply(params["critic"], batch_obs)

    log_probs = jax.nn.log_softmax(logits)

    # Gather the log probabilities of the actions taken
    batch_actions_int = batch_actions.astype(jnp.int32)
    logp_act = jnp.take_along_axis(
        log_probs, batch_actions_int[:, None], axis=1
    ).squeeze(1)

    # Ratio between new and old policies
    ratio = jnp.exp(logp_act - batch_old_log_probs)

    # Clipped surrogate objective
    unclipped = ratio * batch_advantages
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    # Value loss (MSE between returns and predicted values)
    value_loss = c1 * jnp.mean((batch_returns - values) ** 2)

    # Entropy bonus
    probs = jax.nn.softmax(logits)
    entropy = c2 * jnp.mean(-jnp.sum(probs * log_probs, axis=1))

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
    init_env_states,
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
    def gae_for_vmap(data):
        return general_advantage_estimator(critic, params["critic"], data, lam, gamma)

    vmapped_gae = jax.vmap(gae_for_vmap)

    def _update_step(carry, _):
        """This is the body of the scan, representing one full update."""
        params, opt_state, env_states, rng = carry
        rng, rollout_rng = jax.random.split(rng)

        # Faz um rollout (usando a função vetorizada)
        final_states, trajectory = vmapped_rollout(env_states, num_steps_per_update)

        # calcula as vantagens (usando a função vetorizada)
        advantages, returns = vmapped_gae(trajectory)

        # Prepara o batch para update
        # Aplica um flatten nas dimensões (num_envs, num_steps) em um único batch
        def flatten(x):
            return x.reshape(-1, *x.shape[2:])

        # MODIFIED: Correctly slice actions and log_probs to match advantages
        batch_obs = flatten(trajectory.obs[:, :-1])
        batch_actions = flatten(trajectory.info["action"].value[:, :-1])
        batch_old_log_probs = flatten(trajectory.info["action"].logprob[:, :-1])
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
            )

        # calcula os gradientes e atualiza os parametros
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        new_carry = (new_params, new_opt_state, final_states, rollout_rng)
        return new_carry, {"loss": loss_val, "reward": trajectory.reward}

    # loop principal de trainamento, executado por lax.scan
    final_carry, metrics = jax.lax.scan(
        _update_step,
        (params, opt_state, init_env_states, rng),
        None,
        length=int(num_updates),
    )

    return final_carry, metrics
