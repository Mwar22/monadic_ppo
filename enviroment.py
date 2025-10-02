import jax
import jax.numpy as jnp
from flax import struct
from typing import Callable, Tuple, Any, Dict, Optional


@struct.dataclass
class State:
    rng: jax.Array
    data: Dict[str, Any]


@struct.dataclass()
class Action:
    value: jax.Array
    logprob: jax.Array


@struct.dataclass
class Data:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: Optional[Dict[str, Any]]

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
            State,
        ],
        Tuple[State, Any],
    ]

    def run(self, state: State) -> Tuple[State, Any]:
        return self.exec(state)

    def bind(self, f):
        """
        :: Env s x -> (x -> Env s y)-> Env s y
        """

        def new_exec(state: State):
            new_state, ret_data = self.exec(state)
            return f(ret_data).exec(new_state)

        return Env(new_exec)

    def map(self, f):
        """
        :: Env s x -> (x -> y) -> Env s y
        """

        def new_exec(state: State):
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

    def step(state: State):
        rng = state.rng
        last_obs = state.data["obs"]

        new_obs = jnp.clip(last_obs + action, 0, 4)
        done = new_obs == 4
        reward = jnp.where(done, 1.0, 0.0)

        new_state = State(rng, {"obs": new_obs})

        return new_state, Data(obs=new_obs, reward=reward, done=done, info=None)

    return Env(step)


def create_env():
    return my_step(jnp.array(0))


def my_reset():
    def reset(state: State):
        rng = state.rng
        return State(rng, {"obs": jnp.array(0)}), Data(
            jnp.array(0), jnp.array(0), jnp.array(0), None
        )

    return Env(reset)


def policy(data: Data):
    return jnp.where(data.obs < 4, 1, 0)


def show_obs(data: Data):
    print(f"ran show_obs: {data.obs}")
    return data
