"""
Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------
Arquivo com o código principal de treinamento
"""

import os
import sys
sys.stdout.flush()

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")

xla_flags += " --xla_gpu_triton_gemm_any=True --xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
os.environ["XLA_FLAGS"] = xla_flags

# alocação dinamica
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# evita do jax prealocar a gpu inteira
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# limite, pois tbm precisamos de um pouco de vram para o sistema
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60"

import jax
from jax import config

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", False)
print(f"jax_enable_x64: {jax.config.read('jax_enable_x64')}")


import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import config
from new_ppo import TrainingSettings, ppo_train
from etils import epath
from robot import create_training_step, RobotSharedData
from config import MujocoSimConfig, RangeConfig, RewardConfig
from networks import create_networks
from utils import save


################################################FUNÇÕES AUXILIARES DE CONFIGURAÇÂO ###################################
# cria o otimizazor
def create_optimizer(steps):
    lr_scheduler = optax.schedules.cosine_onecycle_schedule(
        peak_value=5e-4,        
        pct_start=0.3,            # 30% do treino subindo (warm-up), 70% descendo
        div_factor=5.0,          # LR inicial = peak_value / div_factor
        final_div_factor=50.0,    # LR final = LR inicial / final_div_factor para o ajuste fino,
        transition_steps=steps
    )

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_scheduler, eps=1e-8),
    )


################################################### INICIALIZAÇÂO #####################################################
rng = jax.random.PRNGKey(777)
rng, rng2 = jax.random.split(rng)
rng, network_settings, network_params = create_networks(rng, obs_size=13, action_size=6)


range_cfg = RangeConfig.init(
    pos_min=jnp.array([-0.468, -0.468, 0]),
    pos_max=jnp.array([0.468, 0.468, 0.664]),
    posvel_min=jnp.array([1e-2, 1e-2, 1e-2]),
    posvel_max=jnp.array([0.1, 0.1, 0.1]),
    ori_min=jnp.array([-3.14, -3.14, -3.14]),
    ori_max=jnp.array([3.14, 3.14, 3.14]),
)

sim_cfg = MujocoSimConfig.init(
    ctrl_freq=50.0,
    sim_freq=1000.0,
)

robot_shared_data = RobotSharedData.init(
    epath.Path("model/joystick_env.xml"),
    epath.Path("model"),
    epath.Path("model/meshes"),
    sim_cfg,
    RewardConfig(),
    range_cfg,
    robot_name="thor",
)

if robot_shared_data.value is None:
    raise RuntimeError("RSD is None")

settings = TrainingSettings.init(
    network_settings,
    network_params,
    robot_shared_data.value,
    optimizer_creator=create_optimizer,
    step_fn_creator=create_training_step,
    num_envs=1500,
    epochs=10,
    action_scale=0.01,
    obs_noise_scale=0.001,
    numberof_goals=20,
    rollout_steps=256,
    target_success=0.6,
)


################################################### TREINAMENTO #######################################################
disable_jit = False

if disable_jit:
    jax.config.update("jax_disable_jit", True)
    print("Debug: JIT disabled!")
else:
    print("JIT compiling and starting training...")


(runpar, optim_state, network_params, state), metrics = ppo_train(
    rng2, network_params, settings
)


############################################### PLOTAGEM / SALVAMENTOS ###############################################

# salva os parametros treinados da rede
save(network_params, "trained_params.msgpack")

# salva as estatísticas de observação acumuladas
save(runpar, "trained_runpar.msgpack")

loss = metrics["avg_loss"]
mean_rewards_vs_timestamp = metrics["mean_rewards_vs_timestamp"]
mean_rewards_vs_goals = metrics["mean_rewards_vs_goals"]
grad_norm = metrics["avg_gradnorm"]
entropy = metrics["avg_entropy"]
success_rate = metrics["success_rate"]
err_tol = metrics["err_tol"]
avg_kl_div = metrics["avg_kl_div"]
avg_ctrl_norm = metrics["avg_ctrl_norm"]

avg_loss = jnp.mean(loss[-20:])
print(f" Training finished! Average loss of last 20 steps: {avg_loss:.4f}")

# plotagem dos dados
print(
    f"mean_rewards_vs_goals: min = {jnp.min(mean_rewards_vs_goals)}, max = {jnp.max(mean_rewards_vs_goals)}"
)
print(
    f"mean_rewards_vs_timestamp: min = {jnp.min(mean_rewards_vs_timestamp)}, max = {jnp.max(mean_rewards_vs_timestamp)}"
)

fig, axs = plt.subplots(3, 3, figsize=(10, 8), tight_layout=True)
axs[0][0].plot(loss)
axs[0][0].set_title("Training Loss")
axs[0][0].set_xlabel("Epochs")
axs[0][0].set_ylabel("Loss")
axs[0][0].grid(True)

axs[0][1].plot(grad_norm)
axs[0][1].set_title("Gradient norm (Euclidian, L2)")
axs[0][1].set_xlabel("Epochs")
axs[0][1].set_ylabel("Norm")
axs[0][1].grid(True)

axs[0][2].plot(avg_kl_div)
axs[0][2].set_title("Mean KL divergence")
axs[0][2].set_xlabel("Epochs°")
axs[0][2].set_ylabel("Value")
axs[0][2].grid(True)


axs[1][0].plot(mean_rewards_vs_timestamp)
axs[1][0].set_title("Mean sum of rewards (ac. goals)")
axs[1][0].set_xlabel("Rollout timestamp")
axs[1][0].set_ylabel("Average Reward")
axs[1][0].grid(True)

axs[1][1].plot(mean_rewards_vs_goals)
axs[1][1].set_title("Mean sum of rewards (ac. rollouts)")
axs[1][1].set_xlabel("Goal n°")
axs[1][1].set_ylabel("Average Reward")
axs[1][1].grid(True)


dual = axs[1][2].twinx()
axs[1][2].set_title("Err tol")
axs[1][2].plot(err_tol)
dual.plot(avg_ctrl_norm, color="red")

axs[1][2].set_xlabel("Goal n°")
axs[1][2].set_ylabel("Tol value")
dual.set_ylabel("Avg ctrl norm")
axs[1][2].grid(True)

axs[2][0].plot(entropy)
axs[2][0].set_title("Entropy")
axs[2][0].set_xlabel("Epochs")
axs[2][0].set_ylabel("Entropy value")
axs[2][0].grid(True)
dual.grid(True)


axs[2][1].plot(metrics["avg_err"])
axs[2][1].set_title("Average err")
axs[2][1].set_xlabel("Goal n°")
axs[2][1].set_ylabel("avg err")
axs[2][1].grid(True)

axs[2][2].plot(success_rate)
axs[2][2].set_title("Mean (across envs) success rate")
axs[2][2].set_xlabel("Goal n°")
axs[2][2].set_ylabel("%")
axs[2][2].grid(True)


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
