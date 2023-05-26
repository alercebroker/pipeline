from jax import grad as jgrad
from jax import jit as jjit
from jax import numpy as jnp

from . import models


@jjit
def v1(params, time, flux, error, band, fids, smooth):
    negative = (flux + error) < 0
    # Give lower weights in square error to negative detections, based on how negative it is
    weight = jnp.exp(-(((flux + error) * negative / (error + 1)) ** 2))

    params = params.reshape((-1, 6))
    sq_err = 0.0

    for i in fids:
        ampl, t0, gamma, beta, t_rise, t_fall = params[i]

        sim = models.v1_jax(time, ampl, t0, gamma, beta, t_rise, t_fall)
        sq_err_i = ((sim - flux) / (error + smooth)) ** 2

        sq_err += jnp.dot(sq_err_i * (band == i), weight)

    var = jnp.var(params, axis=0) + jnp.array([1, 5e-2, 5e-2, 5e-3, 5e-2, 5e-2])
    lambdas = jnp.array([0, 1, 0.1, 20, 0.7, 0.01])

    regularization = jnp.dot(lambdas, jnp.sqrt(var))
    return regularization + sq_err


v1_grad = jjit(jgrad(v1))


@jjit
def v2(params, time, flux, error, band, fids, smooth):
    negative = (flux + error) < 0
    # Give lower weights in square error to negative detections, based on how negative it is
    weight = jnp.exp(-(((flux + error) * negative / (error + 1)) ** 2))

    params = params.reshape((-1, 6))
    sq_err = 0.0

    for i in fids:
        ampl, t0, gamma, beta, t_rise, t_fall = params[i]

        sim = models.v2_jax(time, ampl, t0, gamma, beta, t_rise, t_fall)
        sq_err_i = ((sim - flux) / (error + smooth)) ** 2

        sq_err += jnp.dot(sq_err_i * (band == i), weight)

    var = jnp.var(params, axis=0) + jnp.array([1, 5e-2, 5e-2, 5e-3, 5e-2, 5e-2])
    lambdas = jnp.array([0, 1, 0.1, 20, 0.7, 0.01])

    regularization = jnp.dot(lambdas, jnp.sqrt(var))
    return regularization + sq_err


v2_grad = jjit(jgrad(v2))
