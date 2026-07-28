"""NumPy/CVXPY implementation of the FairProjection numerical core.

Kept separate from ``coreMP`` so selecting ``method='np'`` does not import or
execute TensorFlow.
"""

from __future__ import annotations

from itertools import islice
import multiprocessing
from multiprocessing import Pool

import cvxpy as cp
import numpy as np
import scipy as sp


def _solver_value(problem, variable, context):
    problem.solve(warm_start=True)
    accepted = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if problem.status not in accepted or variable.value is None:
        raise RuntimeError(
            f"FairProjection {context} solver failed with status {problem.status!r}."
        )
    value = np.asarray(variable.value)
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"FairProjection {context} solver returned non-finite values.")
    return value


def admm(G, y, rho=2, div="kl", tol=1e-6, max_iter=1000, report=False):
    """Run the upstream ADMM algorithm using NumPy and CVXPY only."""
    del tol
    if div not in {"kl", "cross-entropy"}:
        raise ValueError("div must be 'kl' or 'cross-entropy'.")
    n, c, k = G.shape
    logy = np.log(y)
    x = np.zeros((n, c, 1))
    dual = np.ones((k, 1))
    v = np.ones((n, c, 1))
    mu = np.ones((n, c, 1))
    G_t = np.transpose(G, axes=[0, 2, 1])
    Q = np.sum(G_t @ G, axis=0) / n

    dual_variable = cp.Variable(shape=(k, 1), nonneg=True)
    linear_part = cp.Parameter(shape=(k, 1))
    cost = (rho / 2) * cp.quad_form(dual_variable, cp.psd_wrap(Q))
    problem = cp.Problem(cp.Minimize(cost + linear_part.T @ dual_variable))

    for _ in range(max_iter):
        cv = mu + rho * (G @ dual)
        inner_tol = 1e-13
        if div == "kl":
            a = cv - rho * logy
            x = np.zeros((n, c, 1))
            for _ in range(50):
                previous = x
                x = -(sp.special.softmax(x, axis=1) + a) / rho
                if np.abs(x - previous).max() < inner_tol:
                    break
            v = x - logy
        else:
            z = np.zeros((n, 1, 1))
            a = 4 * rho * y
            for _ in range(50):
                cpz = cv + z
                root = np.sqrt(a + cpz * cpz)
                numerator = -cpz + root
                gradient = (-1 + numerator.sum(axis=1) * 0.5).reshape(n, 1, 1)
                derivative = -0.5 * (numerator / root).sum(axis=1).reshape(n, 1, 1)
                increment = gradient / derivative
                z -= increment
                if np.abs(increment).max() < inner_tol:
                    break
            cpz = cv + z
            x = 0.5 * (-cpz + np.sqrt(a + cpz * cpz))
            v = -(x + cv) / rho

        linear_part.value = np.sum(G_t @ (mu + rho * v), axis=0) / n
        dual = _solver_value(problem, dual_variable, "NumPy ADMM")
        mu += rho * (v + (G @ dual))

    if report:
        projected = predict(dual, G, y, div=div)
        if not np.all(np.isfinite(projected)):
            raise RuntimeError("FairProjection NumPy ADMM produced invalid probabilities.")
    return dual


def predict(dual, G, y, div="kl"):
    """Compute projected probabilities from a fitted dual parameter."""
    n, _, _ = G.shape
    v = G @ dual
    if div == "kl":
        return sp.special.softmax(-v + np.log(y), axis=1)
    if div != "cross-entropy":
        raise ValueError("div must be 'kl' or 'cross-entropy'.")
    if n < 5000:
        return predict_cross(v, y)

    cores = max(1, multiprocessing.cpu_count() - 1)
    iterator = iter(range(n))
    batches = list(iter(lambda: tuple(islice(iterator, 100)), ()))
    with Pool(cores) as pool:
        values = pool.starmap(
            predict_cross,
            [(v[index, :, :], y[index, :, :]) for index in batches],
        )
    return np.concatenate(values, axis=0)


def predict_cross(
    v,
    y,
    tol=1e-10,
    alpha=0.3,
    beta=0.5,
    max_iter=100,
    return_obj=False,
):
    """Interior-point prediction for the cross-entropy divergence."""
    y_inverse = 1 / (y + tol)
    n, _, _ = y.shape

    def newton_step(h):
        a = h * y_inverse
        b = h * a
        gradient = v - (1 / a)
        offset = (-np.sum(gradient * b, axis=1) / np.sum(b, axis=1)).reshape(
            n, 1, 1
        )
        return -(gradient + offset) * b

    def objective(h):
        return np.sum(v * h, axis=1) - np.sum(y * np.log(h), axis=1)

    def gradient(h):
        return v - y / h

    def newton_decrement(h, step):
        return np.sqrt(np.sum(step * step * y / (h * h), axis=1)).max()

    def line_search(h, step):
        scale = np.ones((n, 1, 1))
        while True:
            candidate = h + step * scale
            invalid = candidate.min(axis=1) < 0
            if not np.any(invalid):
                break
            scale[invalid] *= beta

        delta = (gradient(h) * step).sum(axis=1).reshape(n, 1, 1)
        current = objective(h).reshape(n, 1, 1)
        while True:
            candidate = h + step * scale
            insufficient = (
                objective(candidate).reshape(n, 1, 1)
                > current + alpha * scale * delta
            )
            if not np.any(insufficient):
                return candidate
            scale[insufficient] *= beta

    h = y.copy()
    objective_value = objective(h)
    for _ in range(max_iter):
        step = newton_step(h)
        if newton_decrement(h, step) ** 2 / 2 < tol:
            break
        h = line_search(h, step)
        objective_value = objective(h) + np.sum(y * np.log(y), axis=1)
    if return_obj:
        return h, objective_value
    return h
