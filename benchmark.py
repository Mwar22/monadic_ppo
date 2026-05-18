"""
Copyright (c) 2025 Lucas de Jesus
Licensed under CC BY-ND 4.0 with additional commercial use restrictions.
See the LICENSE file in the project root for full license details.
------------------------------------------------------------------------
Arquivo com o código principal de treinamento
"""

import os
import sys
import time
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
        peak_value=8e-3, transition_steps=steps
    )

    return optax.chain(
        optax.clip_by_global_norm(1.0),  # gradient clipping
        optax.adam(lr_scheduler),
    )


################################################### INICIALIZAÇÂO #####################################################
rng = jax.random.PRNGKey(42)
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
    num_envs=1280,
    epochs=10,
    action_scale=2.0,
    obs_noise_scale=0.001,
    numberof_goals=10, 
    rollout_steps=128,  #256 vs 512
    target_success=0.4,
)


################################################### TREINAMENTO #######################################################
disable_jit = False

if disable_jit:
    jax.config.update("jax_disable_jit", True)
    print("Debug: JIT disabled!")
else:
    print("JIT compiling and starting training...")

# 1. Calcular a quantidade total de passos que o grafo vai processar
passos_por_iteração = settings.num_envs * settings.rollout_steps
total_passos_treino = settings.active_numberof_goals * passos_por_iteração

print("==================================================")
print(f"Configuração do Treino:")
print(f"  Ambientes em paralelo: {settings.num_envs}")
print(f"  Passos de Rollout:     {settings.rollout_steps}")
print(f"  Total de Objetivos/Iterações: {settings.active_numberof_goals}")
print(f"  Total de passos no Grafo: {total_passos_treino:,}")
print("==================================================")

# =====================================================================
# CHAMADA 1: Compilação JIT + Execução Inicial (Warmup)
# =====================================================================
print("\nExecução 1: Compilando o grafo XLA e executando (isto vai demorar)...")
tempo_inicio_compilacao = time.perf_counter()

# Dispara a função (o @jax.jit vai compilar aqui se for a primeira vez)
final_carry, metrics = ppo_train(rng, network_params, settings)

# CRÍTICO: Força o Python a esperar a GPU terminar 100% da computação
metrics["avg_loss"].block_until_ready()

tempo_total_compilacao = time.perf_counter() - tempo_inicio_compilacao
print(f"-> Execução 1 concluída em {tempo_total_compilacao:.2f} segundos (Compilação + Treino).")

# =====================================================================
# CHAMADA 2: Benchmark de Velocidade Pura (Usando o Grafo em Cache)
# =====================================================================
print("\nExecução 2: Iniciando Benchmark de Velocidade Pura (Grafo em Cache)...")

# Avança o RNG para a segunda corrida não ser uma cópia idêntica de dados
rng, subkey = jax.random.split(rng)

tempo_inicio = time.perf_counter()

# Roda exatamente a mesma função com os mesmos formatos de inputs
final_carry_bench, metrics_bench = ppo_train(subkey, network_params, settings)

# Bloqueia novamente até que a GPU termine o treino completo
metrics_bench["avg_loss"].block_until_ready()

tempo_puro_execucao = time.perf_counter() - tempo_inicio

# =====================================================================
# 2. CÁLCULO DO SPS REAL
# =====================================================================
sps = total_passos_treino / tempo_puro_execucao

print("\n" + "="*50)
print("               RESULTADOS DO BENCHMARK            ")
print("="*50)
print(f"Tempo de Execução de Hardware:  {tempo_puro_execucao:.3f} segundos")
print(f"Total de Passos Simulados:       {total_passos_treino:,}")
print(f"Passos por Segundo (SPS Real):   {sps:,.0f}")
print("="*50)