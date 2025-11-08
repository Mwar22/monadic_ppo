import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from enviroment import EnvState, Data, State, Action
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

env_states = {"rng":rng, "step":0, "goal":None, "obs_history":None}
reset = create_reset(joystick)
reset.run(env_states)
