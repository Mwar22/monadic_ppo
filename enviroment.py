import jax
from flax import struct
from typing import Callable, Tuple, Any, Dict, Optional


@struct.dataclass
class State:
    rng: jax.Array
    data: Dict[str, Any]


@struct.dataclass
class Action:
    value: jax.Array
    logprob: jax.Array


@struct.dataclass
class Data:
    obs: jax.Array
    reward: jax.Array
    info: Dict[str, Any] | None

    def clone(self):
        return Data(self.obs, self.reward,  self.info)


@struct.dataclass
class StateMonad:
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

    def run(self, state: Any) -> Tuple[Any, Any]:
        return self.exec(state)

    def bind(self, f):
        """
        :: Env s x -> (x -> Env s y)-> Env s y
        """

        def new_exec(state: Any):
            new_state, ret_data = self.exec(state)
            return f(ret_data).exec(new_state)

        return StateMonad(new_exec)


    def map(self, f):
        """
        :: Env s x -> (x -> y) -> Env s y
        """

        def new_exec(state: Any):
            new_state, ret_data = self.exec(state)
            return new_state, f(ret_data)

        return StateMonad(new_exec)
    
    @staticmethod
    def pure(value: Any):
        return StateMonad(lambda s: (s, value))
