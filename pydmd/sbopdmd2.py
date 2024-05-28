"""
Derived module from bopdmd.py for BOP-DMD with sparse modes.
"""

from numbers import Number
from typing import Callable, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy.sparse import csr_matrix

from .bopdmd import BOPDMDOperator
from .sbopdmd import SparseBOPDMD
from .snapshots import Snapshots
from .utils import compute_rank, compute_svd
from .sbopdmd_utils import accelerated_prox_grad, split_B


class sBOPDMDOperator2(BOPDMDOperator):
    """
    BOP-DMD operator with sparse modes.
    """

    def __init__(
        self,
        mode_regularizer,
        mode_prox,
        compute_A,
        use_proj,
        init_alpha,
        init_B,
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
        self._init_B = init_B

        # Set the parameters of accelerated prox gradient descent.
        self._prox_grad_params = {}
        self._prox_grad_params["tol"] = prox_grad_tol
        self._prox_grad_params["max_iter"] = prox_grad_maxiter
        self._prox_grad_params["use_restarts"] = prox_grad_restart
        self._prox_grad_params["normalize_rows"] = True

    def _exp_function2(self, alpha, t, i):
        """
        Derivatives of the matrix of exponentials.

        :param alpha: Vector of time scalings in the exponent.
        :type alpha: numpy.ndarray
        :param t: Vector of time values.
        :type t: numpy.ndarray
        :param i: Index in alpha of the derivative variable.
        :type i: int
        :return: Derivatives of Phi(alpha, t) with respect to alpha[i].
        :rtype: scipy.sparse.csr_matrix
        """
        m = len(t)
        n = len(alpha)
        A = np.exp(alpha[i] * t)
        return csr_matrix(
            (A, (np.arange(m), np.full(m, fill_value=i))), shape=(m, n)
        )

    def _variable_projection(self, H, t, init_alpha, Phi, dPhi):
        """
        Variable projection routine for multivariate data with regularization.
        Note: The B matrix is no longer amplitude-scaled.
        """
        # Define M, IS, and IA.
        M, IS = H.shape
        IA = 2 * len(init_alpha)
        ia = len(init_alpha)

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

        def compute_objective(B, alpha, b):
            """
            Compute the current objective.
            """
            residual = H - np.linalg.multi_dot([Phi(alpha, t), np.diag(b), B])
            objective = 0.5 * np.linalg.norm(residual, "fro") ** 2
            objective += self._mode_regularizer(B)

            return objective

        def compute_B(B0, alpha, b):
            """
            Use accelerated prox gradient to update B for the current alpha.
            """
            A = Phi(alpha, t).dot(np.diag(b))
            beta_f = np.linalg.norm(A, 2) ** 2

            def func_f(Z):
                return 0.5 * np.linalg.norm(H - A.dot(Z), "fro") ** 2

            def grad_f(Z):
                return A.conj().T.dot(A.dot(Z) - H)

            B_updated, obj_hist, err_hist = accelerated_prox_grad(
                B0,
                func_f,
                self._mode_regularizer,
                grad_f,
                self._mode_prox,
                beta_f,
                **self._prox_grad_params,
            )

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

        def compute_alpha(B, alpha_0, b_0):
            """
            Use Levenberg-Marquardt to step alpha for the current B.
            """
            djac_matrix = np.zeros((M * IS, IA), dtype="complex")
            rjac = np.zeros((2 * IA, IA), dtype="complex")
            scales = np.zeros(IA)
            objective = compute_objective(B, alpha_0, b_0)
            residual = H - np.linalg.multi_dot([Phi(alpha_0, t), np.diag(b_0), B])

            for i in range(ia):
                djac_matrix[:, i] = dPhi(alpha_0, t, i).dot(np.diag(b_0)).dot(B).ravel(order="F")
                # Scale for the Levenberg-Marquardt algorithm.
                scales[i] = min(np.linalg.norm(djac_matrix[:, i]), 1)
                scales[i] = max(scales[i], 1e-6)

            for i in range(ia):
                djac_matrix[:, i + ia] = self._exp_function2(alpha_0, t, i).dot(B).ravel(order="F")
                scales[i + ia] = min(np.linalg.norm(djac_matrix[:, i + ia]), 1)
                scales[i + ia] = max(scales[i + ia], 1e-6)

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
                # Compute the updated gamma vector.
                gamma_updated = np.concatenate([alpha_0.ravel(), b_0.ravel()]) + delta.ravel()
                alpha_updated = gamma_updated[:ia]
                alpha_updated = self._push_eigenvalues(alpha_updated)
                b_updated = gamma_updated[ia:]
                return alpha_updated, b_updated

            for j in range(maxlam + 1):
                # Scale lambda up every iteration.
                lam = init_lambda * (lamup**j)

                # Take a step using our step size lam.
                alpha_new, b_new = step(lam)
                objective_new = compute_objective(B, alpha_new, b_new)

                # If the objective improved, terminate.
                if objective_new < objective:
                    return alpha_new, b_new

            # Terminate if no appropriate step length was found...
            if verbose:
                print(
                    "Failed to find appropriate step length. "
                    "Consider increasing maxlam or changing lamup."
                )

            return alpha_0

        # Initialize values.
        alpha = self._push_eigenvalues(init_alpha)
        if self._init_B is None:
            B = np.linalg.lstsq(Phi(alpha, t), H, rcond=None)[0]
        else:
            B = np.copy(self._init_B)
        B, b = split_B(B)

        # Initialize storage for objective values and error.
        # Note: "error" refers to differences in iterations.
        all_obj = np.empty(maxiter)
        all_err = np.empty(maxiter)

        # Initialize termination flags.
        converged = False
        stalled = False

        for itr in range(maxiter):
            # Get the new optimal matrix B.
            B_new = compute_B(B, alpha, b)

            # Take a Levenberg-Marquardt step to update alpha.
            alpha_new, b_new = compute_alpha(B_new, alpha, b)

            # Get new objective and error values.
            err_alpha = np.linalg.norm(alpha - alpha_new)
            err_B = np.linalg.norm(B - B_new)
            all_obj[itr] = compute_objective(B_new, alpha_new, b_new)
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


class SparseBOPDMD2(SparseBOPDMD):
    """
    Dynamic Mode Decomposition with sparse modes.

    :param mode_regularizer: Regularizer portion of the objective function
        given matrix input X. Can be a function, or one of the following preset
        regularizers. Note that if a preset regularizer is used, the mode_prox
        function will be precomputed and will not need to be provided by the
        user. Use regularizer_params instead if using a preset.
        - "l0" (scaled L0 norm)
        - "l1" (scaled L1 norm)
        - "l2" (scaled L2 norm)
        - "l02" (scaled L0 norm + scaled L2 norm squared)
        - "l12" (scaled L1 norm + scaled L2 norm squared)
    :type mode_regularizer: str or function
    :param regularizer_params: Dictionary of parameters for the mode
        regularizer to be used if a preset regularizer is requested.
        - "lambda" - Scaling for the first norm term (used by all presets).
            Defaults to 1.0.
        - "lambda_2" - Scaling for the L2 norm squared (used by "l02", "l12").
            Defaults to 1e-6.
    :type regularizer_params: dict
    :param mode_prox: Proximal operator of the given mode_regularizer function
        given matrix input X and a constant float.
    :type mode_prox: function
    """

    def __init__(
        self,
        mode_regularizer: Union[str, Callable] = None,
        regularizer_params: dict = None,
        mode_prox: Callable = None,
        svd_rank: Number = 0,
        compute_A: bool = False,
        init_alpha: np.ndarray = None,
        init_B: np.ndarray = None,
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
            mode_regularizer=mode_regularizer,
            regularizer_params=regularizer_params,
            mode_prox=mode_prox,
            svd_rank=svd_rank,
            compute_A=compute_A,
            init_alpha=init_alpha,
            init_B=init_B,
            num_trials=num_trials,
            trial_size=trial_size,
            eig_sort=eig_sort,
            eig_constraints=eig_constraints,
            remove_bad_bags=remove_bad_bags,
            bag_warning=bag_warning,
            bag_maxfail=bag_maxfail,
            varpro_opts_dict=varpro_opts_dict,
        )

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
        self._Atilde = sBOPDMDOperator2(
            self.mode_regularizer,
            self.mode_prox,
            self._compute_A,
            self._use_proj,
            self._init_alpha,
            self._init_B,
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
