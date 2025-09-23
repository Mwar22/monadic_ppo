import jax
import jax.numpy as jnp
from flax import struct
from typing import Callable, Tuple, Any, Dict


@struct.dataclass
class Data:
    obs: jax.Array = jnnp.array(0)
    reward: jax.Array = jnp.array(0)
    done: jax.Array = jnp.array(0)
    info: Dict[str, Any] | None = None

    def clone(self):
        return Data(self.obs, self.reward, self.done, self.info)


@struct.dataclass
class Env:
    """
    Immutable dataclass, where the enviroment by itself is a state monad
    step:: s -> (s', r)
    """

    exec: Callable[
        [
            Any,
        ],
        Tuple[Any, Any],
    ]

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


########################################################################################################

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

    def step(state: Any):
        last_obs, rng = state

        new_state = (jnp.clip(last_obs + action, 0, 4), rng)
        reward = jnp.where(new_state[0] == 4, 1.0, 0.0)
        done = new_state[0] == 4

        return new_state, Data(obs=new_state[0], reward=reward, done=done, info=None)

    return Env(step)


def create_env():
    return my_step(jnp.array(0))


def my_reset():
    def reset(state):
        _, rng = state
        return (jnp.array(0), rng), Data(jnp.array(0), jnp.array(0), jnp.array(0), None)

    return Env(reset)


def policy(data):
    return jnp.where(data.obs < 4, 1, 0)


def show_obs(data):
    print(f"ran show_obs: {data.obs}")
    return data
