import jax.numpy as np
from jax import jacfwd, grad, jit, jvp, custom_jvp

# Motivation:
# https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#numerical-stability
@custom_jvp
def log1pexp(x):
	return np.log(1. + np.exp(x))

@log1pexp.defjvp
def log1pexp_jvp(primals, tangents):
	x, = primals
	x_dot, = tangents
	ans = log1pexp(x)
	ans_dot = (1 - 1/(1 + np.exp(x))) * x_dot
	return ans, ans_dot