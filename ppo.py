import jax
import jax.numpy as jnp
from jax import make_jaxpr
from flax import struct
from typing import Callable, Tuple, NewType, Any

State = NewType("State", jax.Array)


@struct.dataclass
class Data:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


@struct.dataclass
class Env:
    """
    Immutable dataclass, where the enviroment by itself is a state monad
    step:: s -> (s', r)
    """

    exec: Callable[[jax.Array], Tuple[jax.Array, Any]]

    def run(self, state):
        return self.exec(state)

    def bind(self, f):
        """
        :: Env s x -> (x -> Env s y)-> Env s y
        """

        def new_exec(state):
            new_state, ret_data = self.exec(state)
            return f(ret_data).exec(new_state)

        return Env(new_exec)

    def map(self, f):
        """
        :: Env s x -> (x -> y) -> Env s y
        """

        def new_exec(state):
            new_state, ret_data = self.exec(state)
            return new_state, f(ret_data)

        return Env(new_exec)


"""
Tiny env for testing.
----------------------
State: an integer s representing position on a 1D line.
Range: 0 to 4.
Actions: +1 (move right) or -1 (move left).
Reward: +1 when the agent reaches the goal state 4, 0 otherwise.
Done: True when s == 4.
Max episode length: 5 steps (to keep it short for debugging).
"""


def my_step(action):
    """
    :: a -> (s -> (s', d))
    """

    def step(state):
        new_state = jnp.clip(state + action, 0, 4)
        reward = jnp.where(new_state == 4, 1.0, 0.0)
        done = new_state == 4
        return new_state, Data(new_state, reward, done)

    return Env(step)


def create_env():
    return my_step(jnp.array(0))


def my_reset():
    def reset(state):
        return jnp.array(0), Data(jnp.array(0), jnp.array(0), jnp.array(0))

    return Env(reset)


def policy(data):
    return jnp.where(data.obs < 4, 1, 0)


def show_obs(data):
    print(f"ran show_obs: {data.obs}")
    return data


"""
my_env = (
    my_reset().map(lambda data: policy(data)).bind(lambda action: my_step(action))
    # .map(lambda data: show_obs(data))
    # my_env.map(lambda _, data: show_state(data))
    # my_env.map(lambda _, data: policy(data.obs))
    # my_env.bind(lambda state, action: my_step(state, action))
    # .run(jnp.array(0))
)
"""


def compose_pipeline(env: Env) -> Env:
    # Compose pipeline once
    return env.map(policy).bind(my_step)


def rollout(init_state: jax.Array, num_steps: int, base_env: Env):
    pipeline = compose_pipeline(base_env)

    def scan_fn(state, _):
        new_state, data = pipeline.run(state)
        return new_state, data

    final_state, traj = jax.lax.scan(scan_fn, init_state, None, length=num_steps)
    return final_state, traj


# Initialize
init_state = jnp.array(0)
num_steps = 5

# JIT compile the rollout
jit_rollout = jax.jit(rollout, static_argnums=(1, 2))

my_env = create_env()
"""
final_state, trajectory = jit_rollout(init_state, num_steps, my_env)


print("Final state:", final_state)
print("Trajectory rewards:", trajectory.reward)
print("Trajectory dones:", trajectory.done)
print("Trajectory observations:", trajectory.obs)

"""

jaxpr = make_jaxpr(rollout, static_argnums=(1, 2))(init_state, num_steps, my_env)
print(jaxpr)
