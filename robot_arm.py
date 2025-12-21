import jax
import os
import optax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from jax import config
from enviroment import StateMonad, Data, State, Action
from ppo import ppo_train, cont_sample_beta
from typing import Tuple, Any
from etils import epath
from joystick import create_joystick, create_reset, create_step
from mjx_base import EnviromentConfig, RangeConfig, ResetConfig, RewardConfig
from functools import partial
from jax.nn import initializers

#######################################################################################
# Modelos
#######################################################################################
activation_str = "sigmoid"
activation = lambda x: nn.sigmoid(x)

class Policy(nn.Module):
    action_dim: int
    discrete: bool

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(256)(obs)
        x = activation(x)
        #x = nn.LayerNorm()(x)
        
   
        x = nn.Dense(256)(obs)
        x = activation(x)
       
        logits = nn.Dense(self.action_dim)(x)
        return logits


class Critic(nn.Module):
    activation_name: str = "relu" #default

    @nn.compact
    def __call__(self, obs):
       
        x = nn.Dense(256)(obs)
        x = activation(x)
        #x = nn.LayerNorm()(x)
        
        x = nn.Dense(256)(obs)
        x = activation(x)
        
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

def get_action(policy: nn.Module, params) -> StateMonad:
    """
    :: p -> s -> EnvState s a
    """
    
    def func(state) -> Tuple[Any, Any]:
        last_obs = state["obs_history"]
        rng1, rng2 = jax.random.split(state["rng"])

        output = policy.apply(params, last_obs)
        action_value, logprob = cont_sample_beta(output, rng1)
        
        new_state = {**state, "rng": rng2, "action": action_value}
        return new_state, {"action": action_value, "logprob": logprob}

    return StateMonad(func)


def compose_pipeline(policy: nn.Module, policy_params, step_fn) -> StateMonad:
    env = StateMonad.pure({})

    def action_data_into_step(adata):
        """Combina dados da ação na saída """
        def fn(state):
            new_state, step_data = step_fn(state)
            return new_state, {**adata, **step_data}
        return StateMonad(fn)

    return (
        env.bind(lambda _: get_action(policy, policy_params))
        .bind(lambda adata: action_data_into_step(adata))
    )


#######################################################################################
# deve ser configurado antes de importar o jax ou TensorFlow/XLA
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
#xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

config.update("jax_enable_x64", False)
print(f"jax_enable_x64: {jax.config.read('jax_enable_x64')}")    

#env_states = {"rng":rng, "step":0, "goal":None, "obs_history":}
#######################################################################################
# --- Hyperparameters ---
NUM_UPDATES = 250#1000
NUM_ENVS = 512 #512
NUM_STEPS_PER_UPDATE = 250
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
ACTION_DIM = 6

# --- Inicialização ---
rng = jax.random.PRNGKey(42)
rng, policy_rng, critic_rng, env_rng = jax.random.split(rng, 4)

ARM_JOINTS = [
    "junta1",
    "junta2",
    "junta3",
    "junta4",
    "junta5",
    "junta6"
]

joystick = create_joystick(
    epath.Path("model/joystick_env.xml"),
    epath.Path("model"),
    epath.Path("model/meshes"),
    EnviromentConfig(),
    RewardConfig(),
    RangeConfig(),
    ResetConfig(),
    ARM_JOINTS
)

#controi-se as funções de step e reset para o ambiente
reset_jit = jax.jit(create_reset(joystick).run)
step_jit = jax.jit(create_step(joystick).run)


# Inicializa os modelos
policy = Policy(action_dim=ACTION_DIM, discrete=True)
critic = Critic()
dummy_obs_single = jnp.zeros((1, 15 * 45))
params = {
    "policy": policy.init(policy_rng, dummy_obs_single),
    "critic": critic.init(critic_rng, dummy_obs_single),
}

# Inicializa o otimizador
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(params)

# Inicializa os estados para os ambientes em paralelo
# (num_envs, features_dim)
envs_obs = jnp.zeros((NUM_ENVS, 1)) #dimensão extra do batch
envs_step = jnp.zeros((NUM_ENVS,))
envs_rng = jax.random.split(env_rng, NUM_ENVS)
envs_action = jnp.zeros((NUM_ENVS, ACTION_DIM))
envs_obs_history = jnp.zeros((NUM_ENVS, 15 * 45))

#estado inicial 
init_envs_states = {"rng":envs_rng, "step":envs_step, "goal":None, "obs_history":envs_obs_history, "action": envs_action, "mjx_data": None}
init_envs_states, _ = jax.vmap(reset_jit)(init_envs_states)

#compõe o pipeline com as transformações
pipeline = compose_pipeline(policy, params["policy"], step_jit)

# --- Executa o treinamento ---
#jax.config.update("jax_disable_jit", True)

print("JIT compiling and starting training...")
(final_params, _, _, _), metrics = ppo_train(
    pipeline,
    params,
    opt_state,
    init_envs_states,
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
print(f"min avg reward: {jnp.min(avg_episode_rewards)}")
print(f"max avg reward: {jnp.max(avg_episode_rewards)}")

grad_norm = metrics["grad_norm"]
grad_to_param_ratio = metrics["grad_to_param_ratio"]


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
ax1.plot(metrics["loss"])
ax1.set_title("Training Loss")
ax1.set_xlabel("Update Step")
ax1.set_ylabel("Loss")

ax2.plot(avg_episode_rewards)
ax2.set_title("Average Episode Reward")
ax2.set_xlabel("Update Step")
ax2.set_ylabel("Average Reward")

ax3.plot(grad_norm)
ax3.set_title("Gradient norm (Euclidian, L2)")
ax3.set_xlabel("Update Step")
ax3.set_ylabel("Norm")

ax4.plot(grad_to_param_ratio)
ax4.set_title("Gradient to parameters ratio")
ax4.set_xlabel("Update Step")
ax4.set_ylabel("Ratio")


plt.savefig(f"training_plots_{activation_str}.png")
print("\nTraining plots saved to training_plots.png")

import pandas as pd

df = pd.DataFrame({
    "step": range(len(metrics["loss"])),
    "loss": metrics["loss"],
    "avg_reward": avg_episode_rewards,
    "grad_norm": grad_norm,
    "grad_to_param_ratio": grad_to_param_ratio
})
df.to_csv(f"l1_{activation_str}.csv", index=False)
print("Done!")