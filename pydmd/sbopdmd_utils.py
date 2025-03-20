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


def sign(X: np.ndarray):
    """
    Returns the sign of the entires of X, element-wise.
    Entries may be real, complex, or zero.
    """
    signs = np.zeros(X.shape, dtype="complex")
    inds_nonzero = X != 0.0
    signs[inds_nonzero] = np.divide(X[inds_nonzero], np.abs(X[inds_nonzero]))

    return signs


def hard_threshold(X: np.ndarray, gamma: float):
    """
    Hard thresholding for the L0 norm.
    """
    X_thres = np.copy(X)
    X_thres[np.abs(X_thres) ** 2 < 2 * gamma] = 0.0

    return X_thres


def soft_threshold(X: np.ndarray, gamma: float):
    """
    Soft thresholding for the L1 norm.
    """
    X_thres = np.multiply(sign(X), np.maximum(np.abs(X) - gamma, 0.0))

    return X_thres


def group_lasso(X: np.ndarray, gamma: float):
    """
    Proximal operator for the L2 norm, applied column-wise.
    """
    # Get the column indices at which the L2 norm is small.
    col_norms = np.linalg.norm(X, 2, axis=0)
    inds_small = col_norms < gamma
    col_norms[inds_small] = 1.0

    # Threshold the given matrix.
    X_thres = np.copy(X)
    X_thres[:, inds_small] = 0.0
    X_thres = X_thres.dot(np.diag(1.0 - (gamma / col_norms)))

    return X_thres


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
    X_thres = scale * hard_threshold(X, (gamma * alpha) / scale)

    return X_thres


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
    X_thres = scale * soft_threshold(X, gamma * alpha)

    return X_thres


def support_lstsq(
    S: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
):
    """
    Least-squares for solving
        B = AX,
    for the matrices A, B over the given support.

    :param S: (r, n) support matrix.
    :type S: np.ndarray
    :param A: (m, r) right-hand side matrix.
    :type A: np.ndarray
    :param B: (m, n) left-hand side matrix.
    :type B: np.ndarray

    :return: (r, n) solution matrix X.
    :rtype: np.ndarray
    """
    # Initialize X with zeros.
    # Note: nonzero entries will be updated later.
    X = np.zeros(S.shape, dtype="complex")

    # Regress again, but only on features given by the support.
    for j in range(X.shape[1]):
        support_inds = S[:, j] != 0.0
        X[support_inds, j] = np.linalg.lstsq(
            A[:, support_inds], B[:, j], rcond=None
        )[0]

    return X


def FISTA(
    X0: np.ndarray,
    func_f: Callable,
    func_g: Callable,
    grad_f: Callable,
    prox_g: Callable,
    beta_f: float,
    tol: float = 1e-6,
    max_iter: int = 1000,
    use_restarts: bool = True,
    sparse_inds: np.ndarray = None,
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
    :param sparse_inds: Indices at which to threshold the solution matrix.
        All indices of the solution matrix are thresholded by default.
    :type sparse_inds: np.ndarray

    :return: Tuple consisting of the following components:
        1. Final optimal solution.
        2. Objective value history across iterations.
        3. Convergece history across iterations.
    :rtype: Tuple[np.ndarray, list, list]
    """
    # Set the indices of X at which to threshold.
    if sparse_inds is None:
        sparse_inds = np.ones(X0.shape, dtype=bool)

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
        # X_new = prox_g(Y - step_size * grad_f(Y), step_size)
        X_new = Y - step_size * grad_f(Y)
        X_new[sparse_inds] = prox_g(X_new[sparse_inds], step_size)
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


def SR3(
    nu: float,
    W0: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    func_g: Callable,
    prox_g: Callable,
    tol: float = 1e-6,
    max_iter: int = 1000,
    sparse_inds: np.ndarray = None,
):
    """
    Sparse Relaxed Regularized Regression (SR3) for
        min_X (1 / 2) * ||AX - B||^2 + (1 / (2 * nu)) * ||X - W||^2 + g(W)
    where g is proxable. Uses alternating updates.

    :param nu: Regularization strength parameter.
    :type nu: float
    :param W0: Initial value for the solver.
    :type W0: np.ndarray
    :param A: (m, r) matrix.
    :type A: np.ndarray
    :param B: (m, n) matrix.
    :type B: np.ndarray
    :param func_g: Regularizer portion of the objective function.
    :type func_g: function
    :param prox_g: Proximal operator of g given X and a constant float.
    :type prox_g: function
    :param tol: Tolerance for terminating the solver.
    :type tol: float
    :param max_iter: Maximum number of iterations for the solver.
    :type max_iter: int
    :param sparse_inds: Indices at which to threshold the solution matrix.
        All indices of the solution matrix are thresholded by default.
    :type sparse_inds: np.ndarray

    :return: Tuple consisting of the following components:
        1. Final (r, n) optimal sparse support matrix.
        2. Objective value history across iterations.
        3. Convergece history across iterations.
    :rtype: Tuple[np.ndarray, list, list]
    """
    # Set the indices of X at which to threshold.
    if sparse_inds is None:
        sparse_inds = np.ones(W0.shape, dtype=bool)

    def obj_func(X, W):
        obj = 0.5 * np.linalg.norm(A.dot(X) - B) ** 2
        obj += (1 / (2 * nu)) * np.linalg.norm(X - W) ** 2
        obj += func_g(W)
        return obj

    # Set initial values.
    X = W0.copy()
    W = W0.copy()

    _, r = A.shape
    L = A.conj().T.dot(A) + (1 / nu) * np.eye(r)
    R = A.conj().T.dot(B)

    obj_hist = np.empty(max_iter)
    err_hist = np.empty(max_iter)

    # Start iteration.
    iter_count = 0
    err = tol + 1.0

    while err >= tol:
        # Perform SR3 step.
        X_new = np.linalg.lstsq(L, R + (1 / nu) * W, rcond=None)[0]
        # W_new = prox_g(X_new, nu)
        W_new = X_new.copy()
        W_new[sparse_inds] = prox_g(W_new[sparse_inds], nu)

        # Get new objective and error values.
        obj = obj_func(X_new, W_new)
        err = np.linalg.norm(W - W_new)
        obj_hist[iter_count] = obj
        err_hist[iter_count] = err

        # Update information.
        np.copyto(X, X_new)
        np.copyto(W, W_new)

        # Check if exceed maximum number of iterations.
        iter_count += 1
        if iter_count >= max_iter:
            return W, obj_hist[:iter_count], err_hist[:iter_count]

    return W, obj_hist[:iter_count], err_hist[:iter_count]


def STLSQ(
    A: np.ndarray,
    B: np.ndarray,
    prox_func: Callable,
    tol: float = 1e-6,
    max_iter: int = 20,
    X0: np.ndarray = None,
    sparse_inds: np.ndarray = None,
    apply_final_prox: bool = False,
):
    """
    Sequential thresholded least-squares for solving
        B = AX
    for fixed matrices A, B. Produces sparse solution X.

    :param A: (m, r) right-hand side matrix.
    :type A: np.ndarray
    :param B: (m, n) left-hand side matrix.
    :type B: np.ndarray
    :param prox_func: Function used to threshold the X matrix.
    :type prox_func: function
    :param tol: Convergence tolerance for the matrix X. If specified, stlsq
        will terminate either once the computed X matrix ceases to change
        between iterations according to the tolerance, or once the maximum
        number of iterations is reached.
    :type tol: float
    :param max_iter: Maximum number of iterations to perform when using a
        convergence tolerance.
    :type max_iter: int
    :param X0: Initial (r, n) matrix for X. If not given, least-squares is used
        to generate an initial matrix.
    :type X0: np.ndarray
    :param sparse_inds: Indices at which to threshold the solution matrix. All
        indices of the solution matrix are thresholded by default.
    :type sparse_inds: np.ndarray
    :param apply_final_prox: Whether or not to apply the given prox function
        one last time prior to returning the output matrix.
    :type apply_final_prox: bool

    :return: Sparse (r, n) solution matrix X.
    :rtype: np.ndarray
    """
    # Initialize the output matrix X.
    if X0 is None:
        X = np.linalg.lstsq(A, B, rcond=None)[0]
    else:
        X = X0.copy()

    # Set the indices of X at which to threshold.
    if sparse_inds is None:
        sparse_inds = np.ones(X.shape, dtype=bool)

    # Perform at most max_iter iterations until X converges.
    error = np.inf
    count = 0
    while error > tol and count < max_iter:
        # Apply thresholding, but only at specified indices.
        X_new = X.copy()
        X_new[sparse_inds] = prox_func(X_new[sparse_inds])

        # Regress again, but only on features that are sufficiently large.
        for j in range(X.shape[1]):
            big_inds = X_new[:, j] != 0.0
            X_new[big_inds, j] = np.linalg.lstsq(
                A[:, big_inds], B[:, j], rcond=None
            )[0]

        # Get the change in X between iterations and update.
        error = np.linalg.norm(X - X_new)
        np.copyto(X, X_new)
        count += 1

    # Apply the prox function one last time if requested.
    if apply_final_prox:
        return prox_func(X)

    return X
