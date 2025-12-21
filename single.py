import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from enviroment import StateMonad, Data, State, Action
from ppo import ppo_train
from typing import Tuple
from etils import epath
from joystick import create_joystick, create_reset, create_step
from mjx_base import EnviromentConfig, RangeConfig, ResetConfig, RewardConfig



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

script_dir = epath.Path(__file__).parent.resolve()
print(script_dir)
joystick = create_joystick(
    script_dir / 'model/joystick_env.xml',
    script_dir / 'model',
    script_dir / 'model/meshes',
    EnviromentConfig(),
    RewardConfig(),
    RangeConfig(),
    ResetConfig(),
    ARM_JOINTS
)

#teste reset
state = {"rng":rng, "step":0, "goal":None, "obs_history":None, "action": jnp.zeros(6), "mjx_data": None}
reset = create_reset(joystick)
reset_jit = jax.jit(reset.run)
state, r = reset_jit(state)

#print("---reset test---")
#print(f"state: {state} r: {r}")
#print(f"state: {state}")

#teste step
step = create_step(joystick)
step_jit = jax.jit(step.run)
state, r = step_jit(state)

print("---step test---")
state, r = step_jit(state)
state, r = step_jit(state)
state, r = step_jit(state)
state, r = step_jit(state)
state, r = step_jit(state)
state, r = step_jit(state)
state, r = reset_jit(state)
state, r = reset_jit(state)
state, r = reset_jit(state)
state, r = reset_jit(state)
#print(f"state: {state} r: {r}")
#print(f"state: {state}")

