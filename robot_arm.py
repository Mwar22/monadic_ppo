import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from enviroment import EnvState, Data, State, Action
from ppo import ppo_train
from typing import Tuple

#######################################################################################
# Modelos
#######################################################################################


class Policy(nn.Module):
    action_dim: int
    discrete: bool

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(2)(obs)
        x = CauchyActivationModule()(x)
        logits = nn.Dense(self.action_dim)(x)  # discrete actions
        return logits


class Critic(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(3)(obs)
        x = CauchyActivationModule()(x)
        value = nn.Dense(1)(x)
        return value.squeeze(-1)


#######################################################################################

#######################################################################################
from jax import jit, custom_jvp

@custom_jvp
@jit
def cauchy_activation(x, lambda1= 0.01, lambda2= 0.01, d=1.0):
    #para estabilidade numerica
    eps = 1e-12
    
    x2_d2 = x**2 + d**2 + eps
    return (lambda1 * x + lambda2) / x2_d2

# gradiente customizado
@cauchy_activation.defjvp
def cauchy_activation_jvp(primals, tangents):
    x, lambda1, lambda2, d = primals
    x_dot, lambda1_dot, lambda2_dot, d_dot = tangents

    #para estabilidade numerica
    eps = 1e-12

    #cria uma versão "clipada" de d, diferenciavel, e evitando valores muito pequenos (explosão de gradiente)
    d_safe = jnp.sqrt(d**2 + 1e-6) # se |d|  << 1e-3, d_safe ≃ 1e-3. Caso contrário d_safe ≃ d
    
    #calcula valores que se repetem
    x2 = x**2
    d2 = d_safe**2
    x2_d2 = x2 + d2 + eps
    x2_d2_sq = x2_d2 ** 2
    inv = 1 / x2_d2_sq

    
    y = (lambda1 * x + lambda2) / x2_d2

    #derivadas parciais
    dy_dx = ((d2 - x2) * lambda1  - 2 * x * lambda2)*inv
    dy_dlambda1 = x / x2_d2
    dy_dlambda2 = 1 / x2_d2
    dy_dd = -2 * d_safe * (lambda1 *x + lambda2) * inv * (d / d_safe)

    tangent_out = dy_dx * x_dot + dy_dlambda1 * lambda1_dot + dy_dlambda2 * lambda2_dot + dy_dd * d_dot
    return y, tangent_out

class CauchyActivationModule(nn.Module):
    init_lambda1: float = 0.01
    init_lambda2: float = 0.01
    d: float = 1.0

    @nn.compact
    def __call__(self, x):
        # trainable params
        lambda1 = self.param("lambda1", lambda rng: jnp.array(self.init_lambda1))
        lambda2 = self.param("lambda2", lambda rng: jnp.array(self.init_lambda2))
        return cauchy_activation(x, lambda1, lambda2, self.d)

#######################################################################################

#######################################################################################

def get_action(policy: nn.Module, params) -> EnvState:
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

    return EnvState(func)


def update_reward(data: Data) -> Data:
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


def update_done(data: Data) -> Data:
    done = (data.obs == 200).squeeze(-1)
    return Data(data.obs, data.reward, done, data.info)


def step(action: Action) -> EnvState:
    def func(state: State):
        rng = state.rng
        last_obs = state.data["obs"]

        action_remap = action.value - 1.0
        new_obs = jnp.clip(last_obs + action_remap, 0, 200)

        new_state = State(rng, {"obs": new_obs})
        return new_state, Data(
            new_obs,
            jnp.array([0]),
            jnp.array([0]),
            info={"action": action, "last_obs": last_obs},
        )

    return EnvState(func)


def reset(state: State):
    return State(state.rng, {"obs": jnp.array([0])}), Data(
        jnp.array([0]), jnp.array([0]), jnp.array([0]), None
    )


def compose_pipeline(policy: nn.Module, policy_params) -> EnvState:
    env = EnvState(reset)

    return (
        env.bind(lambda _: get_action(policy, policy_params))
        .bind(lambda action: step(action))
        .map(update_done)
        .map(update_reward)
    )


#######################################################################################

#######################################################################################

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

#compõe o pipeline com as transformações
pipeline = compose_pipeline(policy, params["policy"])

# --- Executa o treinamento ---
print("JIT compiling and starting training...")
(final_params, _, _, _), metrics = ppo_train(
    pipeline,
    params,
    opt_state,
    env_states,
    rng,
    policy=policy,
    critic=critic,
    optimizer=optimizer,
    num_updates=NUM_UPDATES,
    num_steps_per_update=NUM_STEPS_PER_UPDATE,
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
