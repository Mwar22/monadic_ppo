import jax
import distrax
import optax
import jax.numpy as jnp
import flax.linen as nn
from enviroment import Env, Data, Action
from enviroment import create_env, my_step


class Policy(nn.Module):
    action_dim: int
    discrete: bool

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(64)(obs)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        logits = nn.Dense(self.action_dim)(x)  # discrete actions
        return logits


class Critic(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(64)(obs)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        value = nn.Dense(1)(x)
        return value.squeeze(-1)


#######################################################################################
def get_action(policy: nn.Module) -> Env:
    """
    :: p -> s -> (s', d)
    """

    def disc_eval(logits, rng):
        action = jax.random.categorical(rng, logits)
        return action, jax.nn.log_softmax(logits)[action]

    def cont_eval(mu, sigma, rng):
        dist = distrax.MultivariateNormalDiag(mu, sigma)
        action = dist.sample(seed=rng)
        return action, dist.log_prob(action)

    def func(state):
        last_obs, rng = state
        rng, rng2 = jax.random.split(rng)

        output = policy(last_obs)

        action, logprob = jax.lax.cond(
            policy.discrete,
            lambda logits: disc_eval(logits, rng),  # discrete
            lambda mu_sigma: cont_eval(
                mu_sigma[0], mu_sigma[1], rng
            ),  #  musigma = (mu, sigma)
            operand=output,
        )

        new_state = (last_obs, rng2)
        action_data = {"action": action, "logprob": logprob}

        return new_state, action_data

    return Env(func)


def update_reward(data: Data):
    new_data = data.clone()
    return new_data


def update_done(data: Data):
    new_data = data.clone()
    return new_data


def step(action_data):
    def func(state):
        last_obs, rng = state
        new_state = None
        return new_state, Data(obs=last_obs, info=action_data)

    return Env(func)


def compose_pipeline(env: Env) -> Env:
    # Aplica o pipeline de transformações.
    # primeiro obtem as ações por meio da politica
    # depois obtem o proximo passo com a ação recebida
    policy = Policy(action_dim=1, discrete=True)
    return (
        env.bind(lambda _: get_action(policy))
        .bind(lambda action_data: step(action_data))
        .map(update_reward)
        .map(update_done)
    )


def rollout(init_state: jax.Array, num_steps: int, base_env: Env):
    pipeline = compose_pipeline(base_env)

    def scan_fn(state, _):
        new_state, data = pipeline.run(state)
        return new_state, data

    final_state, traj = jax.lax.scan(scan_fn, init_state, None, length=num_steps)
    return final_state, traj


def general_advantage_estimator(data, critic, lam, gamma):
    obs, rewards, dones = data.obs, data.reward, data.done

    values = critic(obs)
    next_values = jnp.concatenate([values[1:], values[-1:]])  # bootstrap
    inputs = (rewards, values, next_values, dones)

    def gae_scan_fn(carry, inputs):
        gae_next = carry
        reward, value, next_value, done = inputs

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae_next
        return gae, gae

    # faz do ultimo para o primeiro
    _, advantages = jax.lax.scan(gae_scan_fn, 0.0, inputs, reverse=True)
    returns = advantages + values
    return advantages, returns


def ppo_loss(
    data,
    policy,
    critic,
    params,
    actions,
    advantages,
    returns,
    old_log_probs,
    clip_eps=0.2,
    c1=0.5,
    c2=0.0,
):
    """
    params: policy network parameters
    apply_fn: policy network forward function, obs -> logits
    obs: [T, ...] observations
    actions: [T,] discrete actions
    advantages: [T,] from GAE
    returns: [T,] discounted returns
    old_log_probs: [T,] log probs of actions under old policy
    """
    logits = policy(params, data.obs)  # [T, num_actions]
    log_probs = jax.nn.log_softmax(logits)

    # reune os log probs das ações feitas
    logp_act = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(1)

    # razão entre as politicas antiga e nova
    ratio = jnp.exp(logp_act - old_log_probs)

    # clipped surrogate
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    # minimo entre o valor clipado e não clipado
    policy_loss = -jnp.mean(
        jnp.minimum(unclipped, clipped)
    )  # maximizar -> minimizar negativo

    # value loss
    values = critic(params, data.obs)
    value_loss = c1 * jnp.mean((returns - values) ** 2)

    # bonus entropia
    probs = jax.nn.softmax(logits)
    entropy = c2 * jnp.mean(-jnp.sum(probs * log_probs, axis=1))

    total_loss = policy_loss + value_loss - entropy
    return total_loss


##########################################################################

# Initialize
init_state = jnp.array(0)
num_steps = 5

# JIT compile the rollout
jit_rollout = jax.jit(rollout, static_argnums=(1, 2))

my_env = create_env()

final_state, trajectory = jit_rollout(init_state, num_steps, my_env)


print("Final state:", final_state)
print("Trajectory rewards:", trajectory.reward)
print("Trajectory dones:", trajectory.done)
print("Trajectory observations:", trajectory.obs)

"""

jaxpr = make_jaxpr(rollout, static_argnums=(1, 2))(init_state, num_steps, my_env)
print(jaxpr)

"""
