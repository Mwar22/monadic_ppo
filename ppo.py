import jax
import distrax
import optax
import jax.numpy as jnp
import flax.linen as nn
from enviroment import Env, Data, State, Action
from enviroment import create_env, my_step
from typing import Tuple


class Policy(nn.Module):
    action_dim: int
    discrete: bool

    """
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(2)(obs)
        x = nn.tanh(x)
        logits = nn.Dense(self.action_dim)(x)  # discrete actions
        return logits
    """

    def __call__(self, obs):
        return jnp.where(obs < 4, 1, 0)


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
        # action_value, logprob = disc_sample(output, rng1)

        action_value = output
        logprob = 1
        return State(rng2, state.data), Action(action_value, logprob)

    return Env(func)


def update_reward(data: Data):
    reward = jnp.where(data.done, 1.0, 0.0)
    return Data(data.obs, reward, data.done, data.info)


def update_done(data: Data):
    done = data.obs == 4
    return Data(data.obs, data.reward, done, data.info)


def step(action: Action) -> Env:
    def func(state: State):
        rng = state.rng
        last_obs = state.data["obs"]

        new_obs = jnp.clip(last_obs + action.value, 0, 4)
        new_state = State(rng, {"obs": new_obs})

        return new_state, Data(
            new_obs, jnp.array(0), jnp.array(0), info={"action": action}
        )

    return Env(func)


def compose_pipeline(rng: jax.Array, env: Env) -> Env:
    # Aplica o pipeline de transformações.
    # primeiro obtem as ações por meio da politica
    # depois obtem o proximo passo com a ação recebida
    policy = Policy(action_dim=1, discrete=True)

    dummy_obs = jnp.ones((1,))  # Example: batch of 1 with 1 features
    params = policy.init(rng, dummy_obs)

    return (
        env.bind(lambda _: get_action(policy, params))
        .bind(lambda action_data: step(action_data))
        .map(update_done)
        .map(update_reward)
    )


def rollout(init_state: State, num_steps: int, base_env: Env):
    rng, rng1 = jax.random.split(init_state.rng)
    pipeline = compose_pipeline(rng1, base_env)

    def scan_fn(state: State, _):
        new_state, data = pipeline.run(state)
        return new_state, data

    final_state, trajectory = jax.lax.scan(
        scan_fn, State(rng, init_state.data), None, length=num_steps
    )
    return final_state, trajectory


def general_advantage_estimator(data, critic, lam, gamma):
    obs, rewards, dones = data.obs, data.reward, data.done

    values = critic(obs)
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
    Parameters
    ----------
    policy:
        Rede da política (Actor)

    critic:
        Rede do crítico

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

    # valor experado do minimo entre o valor clipado e não clipado
    policy_loss = -jnp.mean(
        jnp.minimum(unclipped, clipped)
    )  # maximizar -> sinal de negativo, já que o otimizador minimiza

    # value loss, MSE entre os retornos e os valores previstos
    values = critic(params, data.obs)
    value_loss = c1 * jnp.mean((returns - values) ** 2)

    # bonus entropia
    probs = jax.nn.softmax(logits)
    entropy = c2 * jnp.mean(-jnp.sum(probs * log_probs, axis=1))

    total_loss = policy_loss + value_loss - entropy
    return total_loss


##########################################################################

# Initialize
rng = jax.random.PRNGKey(42)
data = {"obs": jnp.array(0)}

init_state = State(rng, data)
num_steps = 5

# JIT compile the rollout
jit_rollout = jax.jit(rollout, static_argnums=(1, 2))

my_env = create_env()

final_state, trajectory = jit_rollout(init_state, num_steps, my_env)


print("Final state:", final_state)
print("Trajectory rewards:", trajectory.reward)
print("Trajectory dones:", trajectory.done)
print("Trajectory observations:", trajectory.obs)

gae = general_advantage_estimator(
    trajectory, lambda obs: 0.5 * jnp.ones_like(obs), 0.1, 0.1
)
print("gae: ", gae)

"""

jaxpr = make_jaxpr(rollout, static_argnums=(1, 2))(init_state, num_steps, my_env)
print(jaxpr)

"""
