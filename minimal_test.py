import jax
import jax.numpy as jnp
from config import VectorRange


rng = jax.random.PRNGKey(441)
range = VectorRange.init(5, jnp.array([0, 0, 0]), jnp.array([10, 10, 10]))
rng , value = range.sample_normal(rng, 0.8)
print(value)
