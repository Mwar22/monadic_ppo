import jax
import jax.numpy as jnp
import time

# Polynomial approximation for 1/x^2 over [0.1, 10]
# Coefficients obtained via fitting (example)
coeffs = jnp.array([10.0, -8.5, 2.5, -0.2], dtype=jnp.float32)

def fast_inv_square_poly(x):
    x1 = x
    x2 = x*x
    x3 = x*x*x
    return coeffs[0] + coeffs[1]*x1 + coeffs[2]*x2 + coeffs[3]*x3

# JIT compile
fast_inv_square_poly_jit = jax.jit(fast_inv_square_poly)
normal_inv_square_jit = jax.jit(lambda x: 1.0/(x*x))

# Test input
N = 10_000_000
x = jnp.linspace(0.1, 10.0, N, dtype=jnp.float32)

# Warmup
fast_inv_square_poly_jit(x).block_until_ready()
normal_inv_square_jit(x).block_until_ready()

# Timing
start = time.time()
res_poly = fast_inv_square_poly_jit(x).block_until_ready()
end = time.time()
print("Polynomial approx time:", end - start)

start = time.time()
res_normal = normal_inv_square_jit(x).block_until_ready()
end = time.time()
print("Normal 1/x^2 time:", end - start)

# Relative error
rel_error = jnp.max(jnp.abs(res_poly - res_normal)/res_normal)
print("Max relative error:", rel_error)