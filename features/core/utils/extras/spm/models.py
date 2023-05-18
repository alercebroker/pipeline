import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax.nn import sigmoid as jsigmoid
from numba import njit


@njit
def v1(times, ampl, t0, gamma, beta, t_rise, t_fall):
    """Direct usage of the model"""
    sigmoid_factor = 1 / 3
    t1 = t0 + gamma

    sigmoid = 1 / (1 + np.exp(-sigmoid_factor * (times - t1)))
    den = 1 + np.exp(-(times - t0) / t_rise)
    temp = (1 - beta) * np.exp(-(times - t1) / t_fall) * sigmoid + (1. - beta * (times - t0) / gamma) * (1 - sigmoid)
    return temp * ampl / den


@njit
def v2(times, ampl, t0, gamma, beta, t_rise, t_fall):
    """Uses constrains to provide higher stability with respect to v1"""
    sigmoid_factor = 0.5
    t1 = t0 + gamma

    sigmoid_arg = sigmoid_factor * (times - t1)
    sigmoid = 1 / (1 + np.exp(-np.clip(sigmoid_arg, -10, 10)))
    sigmoid *= sigmoid_arg > -10  # set to zero  below -10
    sigmoid = np.where(sigmoid_arg < 10, sigmoid, 1)  # set to one above 10

    # TODO: Different from v2_jax, but comes from original code. Check before changing
    stable = t_fall >= t_rise  # Early times diverge in unstable case
    sigmoid *= stable + (1 - stable) * (times > t1)

    raise_arg = -(times - t0) / t_rise
    den = 1 + np.exp(np.clip(raise_arg, -20, 20))

    fall_arg = np.clip(-(times - t1) / t_fall, -20, 20)
    temp = (1 - beta) * np.exp(fall_arg) * sigmoid + (1 - beta * (times - t0) / gamma) * (1 - sigmoid)
    return np.where(raise_arg < 20, temp * ampl / den, 0)


@jjit
def v2_jax(times, ampl, t0, gamma, beta, t_rise, t_fall):
    """Uses constrains to provide higher stability with respect to v1"""
    sigmoid_factor = 0.5
    t1 = t0 + gamma

    sigmoid_arg = sigmoid_factor * (times - t1)
    sigmoid = jsigmoid(sigmoid_arg)
    sigmoid *= sigmoid_arg > -10  # set to zero  below -10
    sigmoid = jnp.where(sigmoid_arg < 10, sigmoid, 1)  # set to one above 10

    # TODO: This is apparently a bug carried over (stable should be >=). Check before fixing
    stable = t_fall < t_rise  # Early times diverge in unstable case
    sigmoid *= stable + (1 - stable) * (times > t1)

    raise_arg = -(times - t0) / t_rise
    den = 1 + jnp.exp(np.clip(raise_arg, -20, 20))

    fall_arg = jnp.clip(-(times - t1) / t_fall, -20, 20)
    temp = (1 - beta) * jnp.exp(fall_arg) * sigmoid + (1 - beta * (times - t0) / gamma) * (1 - sigmoid)
    return jnp.where(raise_arg < 20, temp * ampl / den, 0)
