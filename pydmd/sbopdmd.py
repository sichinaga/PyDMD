"""
Derived module from bopdmd.py for BOP-DMD with sparse modes.
"""

from numbers import Number
from typing import Callable, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

from .bopdmd import BOPDMD, BOPDMDOperator
from .snapshots import Snapshots
from .utils import compute_rank, compute_svd


def split_B(B):
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
    normalize_rows: bool,
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
        if normalize_rows:
            X_new = split_B(X_new)[0]

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
            print("Proximal gradient descent reached maximum iterations.")
            return X, obj_hist[:iter_count], err_hist[:iter_count]

    return X, obj_hist[:iter_count], err_hist[:iter_count]


class sBOPDMDOperator(BOPDMDOperator):
    """
    BOP-DMD operator with sparse modes.
    """

    def __init__(
        self,
        mode_regularizer,
        mode_prox,
        split_mode_matrix,
        compute_A,
        use_proj,
        init_alpha,
        proj_basis,
        num_trials,
        trial_size,
        eig_sort,
        eig_constraints,
        remove_bad_bags,
        bag_warning,
        bag_maxfail,
        init_lambda=1.0,
        maxlam=52,
        lamup=2.0,
        maxiter=30,
        tol=1e-6,
        eps_stall=1e-12,
        verbose=False,
        prox_grad_tol=1e-6,
        prox_grad_maxiter=1000,
        prox_grad_restart=True,
    ):
        super().__init__(
            compute_A=compute_A,
            use_proj=use_proj,
            init_alpha=init_alpha,
            proj_basis=proj_basis,
            num_trials=num_trials,
            trial_size=trial_size,
            eig_sort=eig_sort,
            eig_constraints=eig_constraints,
            mode_prox=mode_prox,
            remove_bad_bags=remove_bad_bags,
            bag_warning=bag_warning,
            bag_maxfail=bag_maxfail,
            init_lambda=init_lambda,
            maxlam=maxlam,
            lamup=lamup,
            maxiter=maxiter,
            tol=tol,
            eps_stall=eps_stall,
            verbose=verbose,
        )
        self._mode_regularizer = mode_regularizer
        self._split_mode_matrix = split_mode_matrix

        # Set the parameters of accelerated prox gradient descent.
        self._prox_grad_params = {}
        self._prox_grad_params["tol"] = prox_grad_tol
        self._prox_grad_params["max_iter"] = prox_grad_maxiter
        self._prox_grad_params["use_restarts"] = prox_grad_restart
        self._prox_grad_params["normalize_rows"] = split_mode_matrix

    def _variable_projection(self, H, t, init_alpha, Phi, dPhi):
        """
        Variable projection routine for multivariate data with regularization.
        Note: The B matrix always contains the amplitude-scaled modes.
        """
        # Define M, IS, and IA.
        M, IS = H.shape
        IA = len(init_alpha)

        # Unpack all variable projection parameters stored in varpro_opts.
        (
            init_lambda,
            maxlam,
            lamup,
            _,
            maxiter,
            tol,
            eps_stall,
            _,
            verbose,
        ) = self._varpro_opts

        def compute_objective(B, alpha):
            """
            Compute the current objective.
            """
            residual = H - Phi(alpha, t).dot(B)
            objective = 0.5 * np.linalg.norm(residual, "fro") ** 2
            if self._split_mode_matrix:
                B_normalized = split_B(B)[0]
                objective += self._mode_regularizer(B_normalized)
            else:
                objective += self._mode_regularizer(B)

            return objective

        def compute_B(B0, alpha):
            """
            Use accelerated prox gradient to update B for the current alpha.
            """
            if self._split_mode_matrix:
                B0_normalized, b = split_B(B0)
                A = Phi(alpha, t).dot(np.diag(b))
                init_B = B0_normalized
            else:
                A = Phi(alpha, t)
                init_B = B0

            beta_f = np.linalg.norm(A, 2) ** 2

            def func_f(Z):
                return 0.5 * np.linalg.norm(H - A.dot(Z), "fro") ** 2

            def grad_f(Z):
                return A.conj().T.dot(A.dot(Z) - H)

            B_updated, obj_hist, err_hist = accelerated_prox_grad(
                init_B,
                func_f,
                self._mode_regularizer,
                grad_f,
                self._mode_prox,
                beta_f,
                **self._prox_grad_params,
            )

            # B_updated has normalized rows -- reincorporate the amplitudes.
            if self._split_mode_matrix:
                # Use the updated B to compute updated amplitudes.
                b_updated = np.diag(
                    np.linalg.multi_dot(
                        [
                            np.linalg.pinv(Phi(alpha, t)),
                            H,
                            np.linalg.pinv(B_updated),
                        ]
                    )
                )
                # Hard threshold the amplitudes and make them real.
                b_updated[np.abs(b_updated) ** 2 < 2 * 1e-6] = 0.0
                b_updated = np.abs(b_updated)

                B_updated = np.diag(b_updated).dot(B_updated)

            if verbose:
                print("Prox Gradient Results:")
                plt.figure(figsize=(8, 2))
                plt.subplot(1, 2, 1)
                plt.title("Objective")
                plt.plot(obj_hist, "-o")
                plt.semilogy()
                plt.subplot(1, 2, 2)
                plt.title("Error")
                plt.plot(err_hist, "-o")
                plt.semilogy()
                plt.tight_layout()
                plt.show()

            return B_updated

        def compute_alpha(B, alpha_0):
            """
            Use Levenberg-Marquardt to step alpha for the current B.
            """
            djac_matrix = np.zeros((M * IS, IA), dtype="complex")
            rjac = np.zeros((2 * IA, IA), dtype="complex")
            scales = np.zeros(IA)
            objective = compute_objective(B, alpha_0)
            residual = H - Phi(alpha_0, t).dot(B)

            for i in range(IA):
                # Build the Jacobian matrix by looping over all alpha indices.
                djac_matrix[:, i] = dPhi(alpha_0, t, i).dot(B).ravel(order="F")
                # Scale for the Levenberg-Marquardt algorithm.
                scales[i] = min(np.linalg.norm(djac_matrix[:, i]), 1)
                scales[i] = max(scales[i], 1e-6)

            # Loop to determine lambda (the step-size parameter).
            rhs_temp = residual.ravel(order="F")[:, None]
            q_out, djac_out, j_pvt = qr(
                djac_matrix, mode="economic", pivoting=True
            )
            ij_pvt = np.arange(IA)
            ij_pvt = ij_pvt[j_pvt]
            rjac[:IA] = np.triu(djac_out[:IA])
            rhs_top = q_out.conj().T.dot(rhs_temp)
            scales_pvt = scales[j_pvt[:IA]]
            rhs = np.concatenate(
                (rhs_top[:IA], np.zeros(IA, dtype="complex")), axis=None
            )

            def step(_lambda, scales_pvt=scales_pvt, rhs=rhs, ij_pvt=ij_pvt):
                """
                Helper function that, when given a step size _lambda,
                computes and returns the updated step and alpha vectors.
                """
                # Compute the step delta.
                rjac[IA:] = _lambda * np.diag(scales_pvt)
                delta = np.linalg.lstsq(rjac, rhs, rcond=None)[0]
                delta = delta[ij_pvt]
                # Compute the updated alpha vector.
                alpha_updated = alpha_0.ravel() + delta.ravel()
                alpha_updated = self._push_eigenvalues(alpha_updated)
                return alpha_updated

            for j in range(maxlam + 1):
                # Scale lambda up every iteration.
                lam = init_lambda * (lamup**j)

                # Take a step using our step size lam.
                alpha_new = step(lam)
                objective_new = compute_objective(B, alpha_new)

                # If the objective improved, terminate.
                if objective_new < objective:
                    return alpha_new

            # Terminate if no appropriate step length was found...
            if verbose:
                print(
                    "Failed to find appropriate step length. "
                    "Consider increasing maxlam or changing lamup."
                )

            return alpha_0

        # Initialize values.
        alpha = self._push_eigenvalues(init_alpha)
        B = np.linalg.lstsq(Phi(alpha, t), H, rcond=None)[0]

        # Initialize storage for objective values and error.
        # Note: "error" refers to differences in iterations.
        all_obj = np.empty(maxiter)
        all_err = np.empty(maxiter)

        # Initialize termination flags.
        converged = False
        stalled = False

        for itr in range(maxiter):
            # Get the new optimal matrix B.
            B_new = compute_B(B, alpha)

            # Take a Levenberg-Marquardt step to update alpha.
            alpha_new = compute_alpha(B_new, alpha)

            # Get new objective and error values.
            err_alpha = np.linalg.norm(alpha - alpha_new)
            err_B = np.linalg.norm(B - B_new)
            all_obj[itr] = compute_objective(B_new, alpha_new)
            all_err[itr] = err_alpha + err_B

            # Update information.
            np.copyto(alpha, alpha_new)
            np.copyto(B, B_new)

            # Print iterative progress if the verbose flag is turned on.
            if verbose:
                print(
                    f"Iteration {itr + 1}: "
                    f"Objective = {np.round(all_obj[itr], decimals=3)} "
                    f"Error (alpha) = {np.round(err_alpha, decimals=3)} "
                    f"Error (B) = {np.round(err_B, decimals=3)}.\n"
                )

            # Update termination status and terminate if converged or stalled.
            converged = all_err[itr] < tol
            # Note: we may need to change to abs if this stall condition causes
            # too many early terminations due to worsenig objectives.
            stalled = (itr > 0) and (
                all_obj[itr - 1] - all_obj[itr] < eps_stall * all_obj[itr - 1]
            )

            if converged:
                if verbose:
                    print("Convergence reached!")
                return B, alpha, converged

            if stalled:
                if verbose:
                    msg = (
                        "Stall detected: objective reduced by less than {} "
                        "times the objective at the previous step. "
                        "Iteration {}. Current objective {}. "
                        "Consider decreasing eps_stall."
                    )
                    print(msg.format(eps_stall, itr + 1, all_obj[itr]))
                return B, alpha, converged

        # Failed to meet tolerance in maxiter steps.
        if verbose:
            msg = (
                "Failed to reach tolerance after maxiter = {} iterations. "
                "Current error {}."
            )
            print(msg.format(maxiter, all_err[itr]))

        return B, alpha, converged


class SparseBOPDMD(BOPDMD):
    """
    Dynamic Mode Decomposition with sparse modes.

    :param mode_regularizer: Regularizer portion of the objective function
        given matrix input X.
    :type mode_regularizer: function
    :param mode_prox: Proximal operator of the given mode_regularizer function
        given matrix input X and a constant float.
    :type mode_prox: function
    :param split_mode_matrix: Whether or not to split the amplitudes from the
        mode matrix when performing the optimization. Default behavior is to
        combine the modes and the amplitudes during the optimization.
    :type split_mode_matrix: bool
    """

    def __init__(
        self,
        mode_regularizer: Callable = None,
        mode_prox: Callable = None,
        split_mode_matrix: bool = False,
        svd_rank: Number = 0,
        compute_A: bool = False,
        init_alpha: np.ndarray = None,
        num_trials: int = 0,
        trial_size: Number = 0.8,
        eig_sort: str = "auto",
        eig_constraints: Union[set, Callable] = None,
        remove_bad_bags: bool = False,
        bag_warning: int = 100,
        bag_maxfail: int = 200,
        varpro_opts_dict: dict = None,
    ):
        super().__init__(
            svd_rank=svd_rank,
            compute_A=compute_A,
            use_proj=False,  # don't project the data
            init_alpha=init_alpha,
            proj_basis=None,  # ignore since we don't project the data
            num_trials=num_trials,
            trial_size=trial_size,
            eig_sort=eig_sort,
            eig_constraints=eig_constraints,
            mode_prox=mode_prox,
            remove_bad_bags=remove_bad_bags,
            bag_warning=bag_warning,
            bag_maxfail=bag_maxfail,
            varpro_opts_dict=varpro_opts_dict,
        )
        self._mode_regularizer = mode_regularizer
        self._split_mode_matrix = split_mode_matrix

    def fit(self, X, t):
        """
        Compute the Optimized Dynamic Mode Decomposition with regularization.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param t: the input time vector.
        :type t: numpy.ndarray or iterable
        """
        # Process the input data and convert to numpy.ndarrays.
        self._reset()
        self._snapshots_holder = Snapshots(X)
        self._time = np.array(t).squeeze()

        # Check that input time vector is one-dimensional.
        if self._time.ndim > 1:
            raise ValueError("Input time vector t must be one-dimensional.")

        # Check that the number of snapshots in the data matrix X matches the
        # number of time points in the time vector t.
        if self.snapshots.shape[1] != len(self._time):
            msg = (
                "The number of columns in the data matrix X must match "
                "the number of time points in the time vector t."
            )
            raise ValueError(msg)

        # Compute the rank of the fit.
        self._svd_rank = int(compute_rank(self.snapshots, self._svd_rank))

        # Set/check the projection basis.
        self._proj_basis = compute_svd(self.snapshots, -1)[0]

        # Set/check the initial guess for the continuous-time DMD eigenvalues.
        if self._init_alpha is None:
            self._init_alpha = self._initialize_alpha()
        elif (
            not isinstance(self._init_alpha, np.ndarray)
            or self._init_alpha.ndim > 1
            or len(self._init_alpha) != self._svd_rank
        ):
            msg = "init_alpha must be a 1D np.ndarray with {} entries."
            raise ValueError(msg.format(self._svd_rank))

        # Build the operator now that the initial alpha has been defined.
        self._Atilde = sBOPDMDOperator(
            self._mode_regularizer,
            self._mode_prox,
            self._split_mode_matrix,
            self._compute_A,
            self._use_proj,
            self._init_alpha,
            self._proj_basis,
            self._num_trials,
            self._trial_size,
            self._eig_sort,
            self._eig_constraints,
            self._remove_bad_bags,
            self._bag_warning,
            self._bag_maxfail,
            **self._varpro_opts_dict,
        )

        # Fit the data.
        self._b = self.operator.compute_operator(self.snapshots.T, self._time)

        return self
