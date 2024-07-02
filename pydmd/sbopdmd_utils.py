"""
Utilities module for sbopdmd.py.
"""

from typing import Callable
import numpy as np


def L0_norm(X: np.ndarray):
    """
    Applies the L0 norm to all columns of X and returns the sum.
    """
    return np.sum(X != 0.0)


def L1_norm(X: np.ndarray):
    """
    Applies the L1 norm to all columns of X and returns the sum.
    """
    return np.sum(np.abs(X))


def L2_norm(X: np.ndarray):
    """
    Applies the L2 norm to all columns of X and returns the sum.
    """
    return np.sum(np.linalg.norm(X, 2, axis=0))


def L2_norm_squared(X: np.ndarray):
    """
    Applies the squared L2 norm to all columns of X and returns the sum.
    """
    return np.sum(np.abs(X) ** 2)


def hard_threshold(X: np.ndarray, gamma: float):
    """
    Hard thresholding for the L0 norm.
    """
    X[np.abs(X) ** 2 < 2 * gamma] = 0.0
    return X


def soft_threshold(X: np.ndarray, gamma: float):
    """
    Soft thresholding for the L1 norm.
    """
    return np.sign(X) * np.maximum(np.abs(X) - gamma, 0.0)


def group_lasso(X: np.ndarray, gamma: float):
    """
    Proximal operator for the L2 norm, applied column-wise.
    """
    col_norms = np.linalg.norm(X, 2, axis=0)
    inds_small = col_norms <= gamma
    col_norms[inds_small] = 1.0
    X[:, inds_small] = 0.0
    X = X.dot(np.diag(1.0 - (gamma / col_norms)))
    return X


def scaled_hard_threshold(
    X: np.ndarray,
    gamma: float,
    alpha: float,
    beta: float,
):
    """
    Scaled hard thresholding for the L0 norm and L2 norm squared.
    """
    scale = 1 / (1 + (2 * gamma * beta))
    X = hard_threshold(X, (gamma * alpha) / scale)
    return X * scale


def scaled_soft_threshold(
    X: np.ndarray,
    gamma: float,
    alpha: float,
    beta: float,
):
    """
    Scaled soft thresholding for the L1 norm and L2 norm squared.
    """
    scale = 1 / (1 + (2 * gamma * beta))
    X = soft_threshold(X, gamma * alpha)
    return X * scale


def split_B(B: np.ndarray):
    """
    Split the given amplitude-scaled mode matrix into a normalized mode
    matrix and an array of mode amplitudes.
    """
    b = np.linalg.norm(B, axis=1)

    # Remove extremely small-amplitude modes.
    inds_small = np.abs(b) < (10 * np.finfo(float).eps * np.max(b))
    b[inds_small] = 1.0
    B = np.diag(1 / b).dot(B)
    B[inds_small] = 0.0
    b[inds_small] = 0.0

    return B, b


def get_nonzero_cols(X: np.ndarray, tol=1e-16):
    """
    Return the indices of the columns of X that are not the zero vector.
    """
    X[np.abs(X) < tol] = 0.0
    return np.nonzero(np.any(X, axis=0))[0]


def backtracking_line_search(
    x0: np.ndarray,
    dx: np.ndarray,
    func_f: Callable,
    grad_f: Callable,
    t0: float,
    t_up: float,
    grad_scale: float,
):
    """
    Backtracking line search.
    """
    t = t0
    f0 = func_f(x0)
    f_new = func_f(x0 - dx)
    df0 = grad_scale * np.vdot(dx, grad_f(x0))
    while f_new > f0 - t * df0:
        t *= t_up
        f_new = func_f(x0 - t * dx)
    return t


def accelerated_prox_grad(
    X0: np.ndarray,
    func_f: Callable,
    func_g: Callable,
    grad_f: Callable,
    prox_g: Callable,
    beta_f: float,
    tol: float,
    max_iter: int,
    use_restarts: bool,
):
    """
    Accelerated Proximal Gradient Descent for
        min_X f(X) + g(X)
    where f is beta smooth and g is proxable.

    :param X0: Initial value for the solver.
    :type X0: np.ndarray
    :param func_f: Smooth portion of the objective function.
    :type func_f: function
    :param func_g: Regularizer portion of the objective function.
    :type func_g: function
    :param grad_f: Gradient of f with respect to X.
    :type grad_f: function
    :param prox_g: Proximal operator of g given X and a constant float.
    :type prox_g: function
    :param beta_f: Beta smoothness constant for f.
    :type beta_f: float
    :param tol: Tolerance for terminating the solver.
    :type tol: float
    :param max_iter: Maximum number of iterations for the solver.
    :type max_iter: int
    :param use_restarts: Whether or not to reset t when the objective
        function value worsens.
    :type use_restarts: bool

    :return: Tuple consisting of the following components:
        1. Final optimal solution.
        2. Objective value history across iterations.
        3. Convergece history across iterations.
    :rtype: Tuple[np.ndarray, list, list]
    """
    # Set initial values.
    X = X0.copy()
    Y = X0.copy()
    t = 1.0

    step_size = 1.0 / beta_f
    obj_hist = np.empty(max_iter)
    err_hist = np.empty(max_iter)

    # Start iteration.
    iter_count = 0
    err = tol + 1.0

    while err >= tol:
        # Proximal gradient descent step.
        X_new = prox_g(Y - step_size * grad_f(Y), step_size)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
        Y_new = X_new + ((t - 1.0) / t_new) * (X_new - X)

        # Get new objective and error values.
        obj = func_f(X_new) + func_g(X_new)
        err = np.linalg.norm(X - X_new)
        obj_hist[iter_count] = obj
        err_hist[iter_count] = err

        # Update information.
        np.copyto(X, X_new)
        np.copyto(Y, Y_new)
        t = t_new

        # Reset t if objective function value is getting worse.
        if use_restarts and iter_count > 1:
            if obj_hist[iter_count - 1] < obj_hist[iter_count]:
                t = 1.0

        # Check if exceed maximum number of iterations.
        iter_count += 1
        if iter_count >= max_iter:
            return X, obj_hist[:iter_count], err_hist[:iter_count]

    return X, obj_hist[:iter_count], err_hist[:iter_count]
