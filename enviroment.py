import jax
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
    info: Dict[str, Any]

    def clone(self):
        return Data(self.obs, self.reward, self.done, self.info)


@struct.dataclass
class EnvState:
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

        return EnvState(new_exec)

    def map(self, f):
        """
        :: Env s x -> (x -> y) -> Env s y
        """

        def new_exec(state: State):
            new_state, ret_data = self.exec(state)
            return new_state, f(ret_data)

        return EnvState(new_exec)
