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
from config import MujocoSimConfig, RangeConfig, ResetConfig, RewardConfig
from mathutils import Scheduler
from dataclassutils import NetworksSettings

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
        #x = nn.LayerNorm()(x)
        
   
        x = nn.Dense(256)(x)
        x = activation(x)
       
        # 2 pois é uma distribuição, gerando metade para os parametros alfa e metade para beta
        logits = nn.Dense(2 * self.action_dim)(x)
        return logits


class Critic(nn.Module):
    activation_name: str = "relu" #default

    @nn.compact
    def __call__(self, obs):
       
        x = nn.Dense(256)(obs)
        x = activation(x)
        #x = nn.LayerNorm()(x)
        
        x = nn.Dense(256)(x)
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
    params = (
        actor.init(rng_actor, dummy_obs),
        critic.init(rng_critic, dummy_obs),
    )
    return rng, NetworksSettings(obs_size, action_size, actor, critic, params)


# --- Inicialização ---
rng = jax.random.PRNGKey(42)
rng, network_settings = create_networks(rng, obs_size=37, action_size=6)


robot_shared_data = create_rsd(
    epath.Path("model/joystick_env.xml"),
    epath.Path("model"),
    epath.Path("model/meshes"),
    MujocoSimConfig(),
    RewardConfig(),
    RangeConfig(),
    ["junta1", "junta2", "junta3", "junta4","junta5", "junta6"]
)

def adaptive_progress(current_p, success_rate, target_success = 0.10, step_size=0.005):
    # Só aumenta a dificuldade se mais de 30% dos robôs estão a ter sucesso
    # Se o sucesso for baixo, o progresso "congela" ou até recua um pouco
  
    delta = jnp.where(success_rate > target_success, 0.002, -0.0001)
    new_p = jnp.clip(current_p + delta, 0.0, 1.0)
    return new_p

settings = TrainingSettings.init(
    network_settings,
    robot_shared_data,
    optimizer_creator  = lambda lr: optax.adam(lr),
    step_fn_creator = create_step,
    progress_fn = adaptive_progress,
    num_envs= 500,
    epochs=600,
    steps_per_episode=30,
    learning_rate=5e-4
)



# --- Executa o treinamento ---
#jax.config.update("jax_disable_jit", True)

print("JIT compiling and starting training...")
(rng, runpar, params, optimizer_state), metrics = ppo_train(rng, settings)


avg_loss = jnp.mean(metrics["loss"][-20:])
print(f" Training finished! Average loss of last 20 steps: {avg_loss:.4f}")

# plotagem dos dados
avg_rewards_per_update = metrics["avg_reward"]
avg_episode_rewards = jnp.mean(avg_rewards_per_update, axis=1)
print(f"min avg reward: {jnp.min(avg_episode_rewards)}")
print(f"max avg reward: {jnp.max(avg_episode_rewards)}")

grad_norm = metrics["grad_norm"]
entropy = metrics["entropy"]

avg_ptr = jnp.mean(metrics["ptr"])
print(f"avg ptr per episode: {avg_ptr}")

fig, axs = plt.subplots(3, 2, figsize=(10, 8), tight_layout=True)
axs[0][0].plot(metrics["loss"])
axs[0][0].set_title("Training Loss")
axs[0][0].set_xlabel("Update Step")
axs[0][0].set_ylabel("Loss")

#-------------------------------------------

axs[0][1].plot(avg_episode_rewards)
axs[0][1].set_title("Average Episode Reward")
axs[0][1].set_xlabel("Update Step")
axs[0][1].set_ylabel("Average Reward")

mean_err = jnp.mean(metrics["err"], axis=1)
axs[1][0].plot(mean_err)
axs[1][0].set_title("Mean pos err")
axs[1][0].set_xlabel("Update Step")
axs[1][0].set_ylabel("Pos Err")

"""
axs[1][0].plot(grad_norm)
axs[1][0].set_title("Gradient norm (Euclidian, L2)")
axs[1][0].set_xlabel("Update Step")
axs[1][0].set_ylabel("Norm")
"""
axs[1][1].plot(entropy)
axs[1][1].set_title("Entropy")
axs[1][1].set_xlabel("Update Step")
axs[1][1].set_ylabel("Entropy value")

#print(f"success_rate = {metrics["success_rate"]}")
axs[2][0].plot(metrics["success_rate"])
axs[2][0].set_xlabel("Update Step")
axs[2][0].set_ylabel(" success_rate %")

axs[2][1].plot(metrics["progress"])
axs[2][1].set_xlabel("Update Step ")
axs[2][1].set_ylabel(" Progress %")


plt.savefig(f"training_plots.png")
print("\nTraining plots saved to training_plots.png")

"""
import pandas as pd

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
