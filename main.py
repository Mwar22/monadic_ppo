from __future__ import annotations
import jax
import os
import optax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from jax import config
from new_ppo import TrainingSettings, ppo_train
from etils import epath
from robot import create_step, create_rsd
from config import MujocoSimConfig, RangeConfig, RewardConfig
from mathutils import Scheduler
from dataclassutils import NetworksSettings, NetworkParameters

#######################################################################################
# Modelos
#######################################################################################
activation_str = "sigmoid"
activation = lambda x: nn.relu(x)

class Actor(nn.Module):
    action_dim: int
    discrete: bool

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(256)(obs)
        x = activation(x)
        x = nn.LayerNorm()(x)
        
   
        x = nn.Dense(256)(x)
        x = activation(x)
        x = nn.LayerNorm()(x)
       
        # 2 pois é uma distribuição, gerando metade para os parametros alfa e metade para beta
        logits = nn.Dense(2 * self.action_dim)(x)
        return logits


class Critic(nn.Module):
    activation_name: str = "relu" #default

    @nn.compact
    def __call__(self, obs):
       
        x = nn.Dense(256)(obs)
        x = activation(x)
        x = nn.LayerNorm()(x)
        
        x = nn.Dense(256)(x)
        x = activation(x)
        x = nn.LayerNorm()(x)
        
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

def create_networks(rng:jax.Array, obs_size:int, action_size:int):
    rng, rng_actor, rng_critic = jax.random.split(rng, 3)
    dummy_obs= jnp.zeros((1, obs_size)) 

    actor = Actor(action_size, discrete=True)
    critic = Critic()
    actor_params = actor.init(rng_actor, dummy_obs),
    critic_params = critic.init(rng_critic, dummy_obs),

    return rng, NetworksSettings(obs_size, action_size, actor, critic), NetworkParameters.init(actor_params, critic_params)


# --- Inicialização ---
rng = jax.random.PRNGKey(42)
rng, network_settings, network_params = create_networks(rng, obs_size=34, action_size=6)


robot_shared_data = create_rsd(
    epath.Path("model/joystick_env.xml"),
    epath.Path("model"),
    epath.Path("model/meshes"),
    MujocoSimConfig(),
    RewardConfig(),
    RangeConfig(),
    ["junta1", "junta2", "junta3", "junta4","junta5", "junta6"]
)

settings = TrainingSettings.init(
    network_settings,
    network_params,
    robot_shared_data,
    optimizer_creator  = lambda lr: optax.chain(
        optax.clip_by_global_norm(1.0), # Limits the "shock" of big reward changes
        optax.adam(lr)
    ),
    step_fn_creator = create_step,
    num_envs= 250,
    cycles_per_goal=20,
    epochs=30,
    rollout_steps=128,
    learning_rate=5e-4
)



# --- Executa o treinamento ---
disable_jit = False

if disable_jit:
    jax.config.update("jax_disable_jit", True)
    print("Debug: JIT disabled!")
else:
    print("JIT compiling and starting training...")

(rng, runpar, optim_state, network_params), metrics = ppo_train(rng, network_params, settings)

def metric_shape(metrics_dict, metric_str):
    print(f"{metric_str} shape: {metrics_dict[metric_str].shape}")


metric_shape(metrics, "avg_loss")
metric_shape(metrics, "avg_reward")
metric_shape(metrics, "avg_gradnorm")
metric_shape(metrics, "avg_entropy")
#metric_shape(metrics, "advantages_cycles")

loss = metrics["avg_loss"]
avg_episode_rewards =  metrics["avg_reward"]
grad_norm =  metrics["avg_gradnorm"]
entropy =  metrics["avg_entropy"]
#advantages_cycles = metrics["advantages_cycles"]


avg_loss = jnp.mean(loss[-20:])
print(f" Training finished! Average loss of last 20 steps: {avg_loss:.4f}")

# plotagem dos dados
print(f"min avg reward: {jnp.min(avg_episode_rewards)}")
print(f"max avg reward: {jnp.max(avg_episode_rewards)}")


fig, axs = plt.subplots(3, 2, figsize=(10, 8), tight_layout=True)
axs[0][0].plot(loss)
axs[0][0].set_title("Training Loss")
axs[0][0].set_xlabel("Epochs")
axs[0][0].set_ylabel("Loss")

#-------------------------------------------

axs[0][1].plot(avg_episode_rewards)
axs[0][1].set_title("Average Episode Reward")
axs[0][1].set_xlabel("Cycles per goal")
axs[0][1].set_ylabel("Average Reward")


axs[1][0].plot(grad_norm)
axs[1][0].set_title("Gradient norm (Euclidian, L2)")
axs[1][0].set_xlabel("Epochs")
axs[1][0].set_ylabel("Norm")

axs[1][1].plot(entropy)
axs[1][1].set_title("Entropy")
axs[1][1].set_xlabel("Epochs")
axs[1][1].set_ylabel("Entropy value")

#print(f"success_rate = {metrics["success_rate"]}")
axs[2][0].plot(metrics["sr_cycles"])
axs[2][0].set_xlabel("Cycles")
axs[2][0].set_ylabel(" success_rate %")

axs[2][1].plot(metrics["err"])
axs[2][1].set_xlabel("Cycles per Goal")
axs[2][1].set_ylabel("avg err")


plt.savefig(f"training_plots.png")
print("\nTraining plots saved to training_plots.png")

import pandas as pd

"""
df = pd.DataFrame({
    "step": range(len(metrics["loss"])),
    "loss": metrics["loss"],
    "avg_reward": avg_episode_rewards,
    "grad_norm": grad_norm,
    "success_count": avg_success_count
})
df.to_csv(f"l1_{activation_str}.csv", index=False)
"""
print("Done!")
