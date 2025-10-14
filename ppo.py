import jax
import distrax
import optax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from enviroment import Env, Data, State, Action
from enviroment import create_env, my_step
from typing import Tuple
from jax import make_jaxpr
from functools import partial


class Policy(nn.Module):
    action_dim: int
    discrete: bool

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(2)(obs)
        x = nn.tanh(x)
        logits = nn.Dense(self.action_dim)(x)  # discrete actions
        return logits


class Critic(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(3)(obs)
        x = nn.tanh(x)
        value = nn.Dense(1)(x)
        return value.squeeze(-1)


#######################################################################################
def get_action(policy: nn.Module, params) -> Env:
    """
    :: p -> s -> (s', d)
    """

    def disc_sample(logits: jax.Array, rng: jax.Array):
        """amostra a saida da politica para o caso discreto"""
        action = jax.random.categorical(rng, logits)
        return action.astype(jnp.float32), jax.nn.log_softmax(logits)[action]

    def func(state: State) -> Tuple[State, Action]:
        last_obs = state.data["obs"]
        rng1, rng2 = jax.random.split(state.rng)

        # forward
        output = policy.apply(params, last_obs)
        action_value, logprob = disc_sample(output, rng1)

        return State(rng2, state.data), Action(action_value, logprob)

    return Env(func)


def update_reward(data: Data):
    GOAL = 200
    last_obs = data.info["last_obs"]
    current_obs = data.obs

    # mudança da distancia para o alvo
    dist_old = jnp.abs(last_obs - GOAL)
    dist_new = jnp.abs(current_obs - GOAL)

    progress_reward = (dist_old - dist_new).squeeze(-1)
    goal_reward = jnp.where(data.done, 20, -0.01)

    reward = progress_reward + goal_reward
    return Data(data.obs, reward, data.done, data.info)


def update_done(data: Data):
    done = (data.obs == 200).squeeze(-1)
    return Data(data.obs, data.reward, done, data.info)


def step(action: Action) -> Env:
    def func(state: State):
        rng = state.rng
        last_obs = state.data["obs"]

        action_remap = action.value - 1.0
        new_obs = jnp.clip(last_obs + action_remap, 0, 200)

        new_state = State(rng, {"obs": new_obs})
        return new_state, Data(
            new_obs,
            jnp.array(0),
            jnp.array(0),
            info={"action": action, "last_obs": last_obs},
        )

    return Env(func)


def compose_pipeline(policy: nn.Module, policy_params) -> Env:
    # Aplica o pipeline de transformações.
    # primeiro obtem as ações por meio da politica
    # depois obtem o proximo passo com a ação recebida

    env = create_env()
    return (
        env.bind(lambda _: get_action(policy, policy_params))
        .bind(lambda action: step(action))
        .map(update_done)
        .map(update_reward)
    )


def rollout(policy: nn.Module, policy_params, init_state: State, num_steps: int):
    rng, _ = jax.random.split(init_state.rng)
    pipeline = compose_pipeline(policy, policy_params)

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
        "policy",
        "critic",
        "optimizer",
        "num_updates",
        "num_steps_per_update",
        "num_envs",
    ),
)
def train(
    params,
    opt_state,
    init_env_states,
    rng,
    policy,
    critic,
    optimizer,
    num_updates,
    num_steps_per_update,
    num_envs,
    gamma,
    lam,
):
    """The complete, JIT-compiled training function."""

    # Vetoriza a função de rollout, para rodar de forma paralela entre os ambientes
    # rollout(policy, policy_params, init_state, num_steps)
    vmapped_rollout = jax.vmap(
        partial(rollout, policy, params["policy"]),
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


# --- Hyperparameters ---
NUM_UPDATES = 1000
NUM_ENVS = 128
NUM_STEPS_PER_UPDATE = 500
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
ACTION_DIM = 3

# --- Inicialização ---
rng = jax.random.PRNGKey(42)
rng, policy_rng, critic_rng, env_rng = jax.random.split(rng, 4)

# Inicializa os modelos
policy = Policy(action_dim=ACTION_DIM, discrete=True)
critic = Critic()
dummy_obs_single = jnp.zeros((1, 1))
params = {
    "policy": policy.init(policy_rng, dummy_obs_single),
    "critic": critic.init(critic_rng, dummy_obs_single),
}

# Inicializa o otimizador
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(params)

# Inicializa os estados para os ambientes em paralelo
# (num_envs, features_dim)
init_obs = jnp.zeros((NUM_ENVS, 1))
env_rngs = jax.random.split(env_rng, NUM_ENVS)
env_states = State(rng=env_rngs, data={"obs": init_obs})

# --- Executa o treinamento ---
print("JIT compiling and starting training...")
(final_params, _, _, _), metrics = train(
    params,
    opt_state,
    env_states,
    rng,
    policy=policy,
    critic=critic,
    optimizer=optimizer,
    num_updates=NUM_UPDATES,
    num_steps_per_update=NUM_STEPS_PER_UPDATE,
    num_envs=NUM_ENVS,
    gamma=GAMMA,
    lam=GAE_LAMBDA,
)
avg_loss = jnp.mean(metrics["loss"][-100:])
print(f" Training finished! Average loss of last 100 steps: {avg_loss:.4f}")

# plotagem dos dados
avg_rewards_per_update = jnp.mean(metrics["reward"], axis=1)
avg_episode_rewards = jnp.sum(avg_rewards_per_update, axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)
ax1.plot(metrics["loss"])
ax1.set_title("Training Loss")
ax1.set_xlabel("Update Step")
ax1.set_ylabel("Loss")

ax2.plot(avg_episode_rewards)
ax2.set_title("Average Episode Reward")
ax2.set_xlabel("Update Step")
ax2.set_ylabel("Average Reward")

plt.savefig("training_plots.png")
print("\nTraining plots saved to training_plots.png")
