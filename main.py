"""
Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------
Arquivo com o código principal de treinamento
"""

import os

# deve ser configurado antes de importar o jax ou TensorFlow/XLA
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')

#xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# evita do jax prealocar a gpu inteira
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# limite, pois tbm precisamos de um pouco de vram para o sistema
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import jax
from jax import config
config.update("jax_enable_x64", False)
print(f"jax_enable_x64: {jax.config.read('jax_enable_x64')}")    


import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import config
from new_ppo import TrainingSettings, ppo_train
from etils import epath
from robot import create_step, create_rsd
from config import MujocoSimConfig, RangeConfig, RewardConfig
from networks import save_params, create_networks


################################################FUNÇÕES AUXILIARES DE CONFIGURAÇÂO ###################################
#cria o otimizazor
def create_optimizer(steps):
    
    lr_scheduler = optax.schedules.linear_schedule(
        init_value=5e-4,
        end_value=1e-5,
        transition_steps=steps
    )

    return optax.chain(
        optax.clip_by_global_norm(1.0), # gradient clipping
        optax.adam(lr_scheduler)
    )


################################################### INICIALIZAÇÂO #####################################################
rng = jax.random.PRNGKey(42)
rng, network_settings, network_params = create_networks(rng, obs_size=34, action_size=6)


range_cfg = RangeConfig.init(
    numberof_goals=100,
    position_min_values = jnp.array([-0.468, -0.468, 0]),
    position_max_values = jnp.array([0.468, 0.468, 0.664]),
    position_velocities_min_values = jnp.array([1e-2, 1e-2, 1e-2]),
    position_velocities_max_values = jnp.array([0.1, 0.1, 0.1]),
    orientation_min_values = jnp.array([-3.14, -3.14, -3.14]),
    orientation_max_values = jnp.array([3.14, 3.14, 3.14])
)

robot_shared_data = create_rsd(
    epath.Path("model/joystick_env.xml"),
    epath.Path("model"),
    epath.Path("model/meshes"),
    MujocoSimConfig(),
    RewardConfig(),
    range_cfg,
    ["junta1", "junta2", "junta3", "junta4","junta5", "junta6"]
)

settings = TrainingSettings.init(
    network_settings,
    network_params,
    robot_shared_data,
    optimizer_creator  = create_optimizer,
    step_fn_creator = create_step,
    num_envs= 512,
    cycles_per_goal=30,
    epochs=30,
    rollout_steps=128,
    target_success=0.75,
)


################################################### TREINAMENTO #######################################################
disable_jit = False

if disable_jit:
    jax.config.update("jax_disable_jit", True)
    print("Debug: JIT disabled!")
else:
    print("JIT compiling and starting training...")

(rng, runpar, optim_state, network_params), metrics = ppo_train(rng, network_params, settings)


############################################### PLOTAGEM / SALVAMENTOS ###############################################

#salva os parametros treinados da rede
save_params(network_params)

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
axs[2][0].set_xlabel("Cycles per goal")
axs[2][0].set_ylabel(" success_rate %")

axs[2][1].plot(metrics["avg_err"])
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
