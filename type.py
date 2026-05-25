import jax
import jax.numpy as jnp
from typing import Protocol
from flax import struct
from typing import TYPE_CHECKING, Self, Any, Dict, Tuple, TypeVar, Generic, Callable
from mujoco import mjx

if TYPE_CHECKING:
    from dataclassutils import (
        NetworksSettings,
        NetworkParameters,
        RunningParameters,
        RunningAvg,
        TrainingSettings,
        RangeConfig
    )

class Goals(Protocol):
    def sample(self, progress: float)->Self:
        ...


class SingleObservation(Protocol):
    def as_array(self)->jax.Array:
        ...

    def normalize(self, mean: jax.Array, var: jax.Array) -> Self: 
        ...
  
    def sample(self, mjx_data: mjx.Data, obs_noise=0.0)->Self:
        ...

class Observations(struct.PyTreeNode):
    data: Dict[str, SingleObservation] = struct.field(pytree_node=True)

    def as_array(self)->jax.Array:
        return jnp.concatenate([obs.as_array() for obs in self.data.values()])

    def normalize(self, stats_dict:Dict[str, Any])->Self:
        normalized_data = {
            name: obs.normalize(stats_dict[name].mean, stats_dict[name].var)
            for name, obs in self.data.items()
        }
        return self.replace(data = normalized_data)

    def sample(self, mjx_data: mjx.Data, obs_noise=0.0)->Self:
        sampled_data = {
            name: obs.sample(mjx_data, obs_noise)
            for name, obs in self.data.items()
        }
        return self.replace(data = sampled_data)
    
    def __getitem__(self, key: str) -> SingleObservation:
        return self.data[key]
    

class SingleReward(Protocol):
    gain = 0.6
    def value(self, obs: Observations, progress:jax.Array)->jax.Array:
        ...

class Rewards(struct.PyTreeNode):
    data: Tuple[SingleReward, ...]

    def value(self, obs: Observations, progress:jax.Array)->jax.Array:
        rewards = jnp.array([reward.value(obs, progress) for reward in self.data])
        return jnp.sum(rewards)
    

class RngBound(struct.PyTreeNode):
    comp: Callable[[jax.Array], Tuple[jax.Array, Any]]

    @classmethod
    def init(cls, comp: Callable[[jax.Array], Any])->Self:
        """
        comp:: rng -> rng, r
        """
        return cls(comp)
    
    def run(self, rng: jax.Array):
        return self.comp(rng)
    
    def map(self, f: Callable[[Any], Any]):
        """ 
        ::RngBoundComp a -> x -> y -> RngBoundComp b
        """

        def func(rng:jax.Array):
            new_rng, result = self.comp(rng)
            return new_rng, f(result)
        
        return RngBound(func)
    
    def bind(self, f: Callable[[Any], Self]):
        """ 
        ::RngBoundComp a -> x -> RngBoundComp y -> RngBoundComp b
        f:: x -> RngBoundComp y
        """
        def func(rng):
            new_rng, result = self.comp(rng)
            return f(result).run(new_rng)
        
        return RngBound(func)
