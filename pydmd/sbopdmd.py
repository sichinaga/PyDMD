"""
Derived module from bopdmd.py for BOP-DMD with sparse modes.
"""

from numbers import Number
from typing import Callable, Union
from inspect import isfunction

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

from .bopdmd import BOPDMD, BOPDMDOperator
from .snapshots import Snapshots
from .utils import compute_rank, compute_svd

from .sbopdmd_utils import (
    L0_norm,
    L1_norm,
    L2_norm,
    L2_norm_squared,
    hard_threshold,
    soft_threshold,
    group_lasso,
    scaled_hard_threshold,
    scaled_soft_threshold,
    accelerated_prox_grad,
    get_nonzero_cols,
    split_B,
)


class sBOPDMDOperator(BOPDMDOperator):
    """
    BOP-DMD operator with sparse modes.
    """

    def __init__(
        self,
        mode_regularizer,
        compute_A,
        use_proj,
        use_mask,
        init_alpha,
        init_B,
        proj_basis,
        num_trials,
        trial_size,
        eig_sort,
        eig_constraints,
        mode_prox,
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

        # Information for pixel masking.
        self._use_mask = use_mask
        self._unmasked = []

        # Set the parameters of accelerated prox gradient descent.
        self._prox_grad_params = {}
        self._prox_grad_params["tol"] = prox_grad_tol
        self._prox_grad_params["max_iter"] = prox_grad_maxiter
        self._prox_grad_params["use_restarts"] = prox_grad_restart

    @property
    def unmasked_indices(self):
        """
        Get the indices of the nonzero variables.
        """
        return self._unmasked

    def _variable_projection(self, H, t, init_alpha, Phi, dPhi):
        """
        Variable projection routine for multivariate data with regularization.
        Note: The B matrix always contains the amplitude-scaled modes.
        """

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

        def compute_objective(B, alpha, H):
            """
            Compute the current objective.
            """
            residual = H - Phi(alpha, t).dot(B)
            objective = 0.5 * np.linalg.norm(residual, "fro") ** 2
            objective += self._mode_regularizer(B)

            return objective

        def compute_B(B0, alpha, H):
            """
            Use accelerated prox gradient to update B for the current alpha.
            """
            A = Phi(alpha, t)
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

        def compute_alpha(B, alpha_0, H):
            """
            Use Levenberg-Marquardt to step alpha for the current B.
            """

            # Define M, IS, and IA.
            M, IS = H.shape
            IA = len(alpha_0)

            djac_matrix = np.zeros((M * IS, IA), dtype="complex")
            rjac = np.zeros((2 * IA, IA), dtype="complex")
            scales = np.zeros(IA)
            objective = compute_objective(B, alpha_0, H)
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
                objective_new = compute_objective(B, alpha_new, H)

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

        # Set the projected data if requested.
        if self._use_proj:
            H_proj = H.dot(self._proj_basis.conj())

        # Initialize alpha.
        alpha = self._push_eigenvalues(init_alpha)

        # Initialize B.
        if self._init_B is None:
            B = np.linalg.lstsq(Phi(alpha, t), H, rcond=None)[0]
        else:
            B = np.copy(self._init_B)

        # Initialize storage for the global alpha and B outputs.
        alpha_global = np.empty(alpha.shape, dtype="complex")
        B_global = np.empty(B.shape, dtype="complex")
        all_global_found = False
        itr_global = 0
        k = 1

        # Initialize storage for objective values and error.
        # Note: "error" refers to differences in iterations.
        all_obj = np.empty(maxiter)
        all_err = np.empty(maxiter)

        # Initialize termination flags.
        converged = False
        stalled = False

        for itr in range(maxiter):
            # Get the new optimal matrix B.
            if self._use_mask:
                B_new = np.zeros(B.shape, dtype="complex")
                unmasked_inds = get_nonzero_cols(B)
                B_new[:, unmasked_inds] = compute_B(
                    B[:, unmasked_inds], alpha, H[:, unmasked_inds]
                )
                if itr > 0 and not all_global_found:
                    self._unmasked.append(unmasked_inds)
            else:
                B_new = compute_B(B, alpha, H)

            # Take a Levenberg-Marquardt step to update alpha.
            if self._use_proj:
                B_new_proj = B_new.dot(self._proj_basis.conj())
                alpha_new = compute_alpha(B_new_proj, alpha, H_proj)
            else:
                alpha_new = compute_alpha(B_new, alpha, H)

            # Get new objective and error values.
            err_alpha = np.linalg.norm(alpha - alpha_new)
            err_B = np.linalg.norm(B - B_new)
            all_obj[itr] = compute_objective(B_new, alpha_new, H)
            all_err[itr] = err_alpha + err_B

            # Update information.
            np.copyto(alpha, alpha_new)
            np.copyto(B, B_new)

            # Update the global matrices.
            if self._use_mask:
                # If enough features are active, we have a global mode.
                # Note: On the first iteration, noise etc. often make the mask
                # non-existent -- begin this process after the first iteration.
                if not all_global_found and itr > 0:
                    M = B.shape[1]
                    M_active = len(self._unmasked[-1])
                    mode_found = M_active > 0.9 * M and itr % k == 0
                    if mode_found:
                        # Find global mode and remove it from alpha, B.
                        global_eig, global_mode, alpha, B = self._get_global_mode(B, alpha)
                        # Remove the global mode from the data also.
                        H, H_proj = self._remove_mode(H, t, global_eig, global_mode)
                        # Reserve global information for the final output.
                        alpha_global[itr_global] = global_eig
                        B_global[itr_global] = global_mode
                        itr_global += 1
                    else:
                        alpha_global[itr_global:] = alpha
                        B_global[itr_global:] = B
                        all_global_found = True
                        # del self._unmasked[-1]
                else:
                    alpha_global[itr_global:] = alpha
                    B_global[itr_global:] = B
            else:
                alpha_global = alpha
                B_global = B

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
                return B_global, alpha_global, converged

            if stalled:
                if verbose:
                    msg = (
                        "Stall detected: objective reduced by less than {} "
                        "times the objective at the previous step. "
                        "Iteration {}. Current objective {}. "
                        "Consider decreasing eps_stall."
                    )
                    print(msg.format(eps_stall, itr + 1, all_obj[itr]))
                return B_global, alpha_global, converged

        # Failed to meet tolerance in maxiter steps.
        if verbose:
            msg = (
                "Failed to reach tolerance after maxiter = {} iterations. "
                "Current error {}."
            )
            print(msg.format(maxiter, all_err[itr]))

        return B_global, alpha_global, converged

    def _single_trial_compute_operator(self, H, t, init_alpha):
        """
        Helper function that computes the standard optimized dmd operator.
        Returns the resulting DMD modes, eigenvalues, amplitudes, reduced
        system matrix, full system matrix, and whether or not convergence
        of the variable projection routine was reached.
        """
        B, alpha, converged = self._variable_projection(
            H, t, init_alpha, self._exp_function, self._exp_function_deriv
        )
        # Save the modes, eigenvalues, and amplitudes respectively.
        B, b = split_B(B)
        w = B.T
        e = alpha

        # Compute the projected propagator Atilde.
        w_proj = self._proj_basis.conj().T.dot(w)
        Atilde = np.linalg.multi_dot(
            [w_proj, np.diag(e), np.linalg.pinv(w_proj)]
        )

        # Compute the full system matrix A.
        if self._compute_A:
            A = np.linalg.multi_dot([w, np.diag(e), np.linalg.pinv(w)])
        else:
            A = None

        return w, e, b, Atilde, A, converged

    def _get_global_mode(self, B, alpha):
        """
        Finds and returns
        """
        # We consider the global mode to be the first row of B that contains
        # the largest number of nonzero features, assuming B is thresholded.
        ind_global = np.argmax(np.count_nonzero(B, axis=1))
        global_eig = alpha[ind_global]
        global_mode = B[ind_global]

        # Remove the global component from B and alpha.
        all_ind = np.arange(len(alpha))
        alpha = alpha[all_ind != ind_global]
        B = B[all_ind != ind_global]

        return global_eig, global_mode, alpha, B

    def _remove_mode(self, H, t, global_eig, global_mode):
        """
        Removes 
        """
        global_spatiotemporal = np.outer(np.exp(global_eig * t), global_mode)
        H = H - global_spatiotemporal
        if self._use_proj:
            H_proj = H.dot(self._proj_basis.conj())
        else:
            H_proj = None

        return H, H_proj


class SparseBOPDMD(BOPDMD):
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
        svd_rank: Number = 0,
        compute_A: bool = False,
        use_proj: bool = True,
        use_mask: bool = True,
        init_alpha: np.ndarray = None,
        init_B: np.ndarray = None,
        proj_basis: np.ndarray = None,
        num_trials: int = 0,
        trial_size: Number = 0.8,
        eig_sort: str = "auto",
        eig_constraints: Union[set, Callable] = None,
        mode_prox: Callable = None,
        remove_bad_bags: bool = False,
        bag_warning: int = 100,
        bag_maxfail: int = 200,
        varpro_opts_dict: dict = None,
    ):
        super().__init__(
            svd_rank=svd_rank,
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
            varpro_opts_dict=varpro_opts_dict,
        )
        self._mode_regularizer = mode_regularizer
        self._regularizer_params = regularizer_params
        self._use_mask = use_mask
        self._init_B = init_B

        if self._regularizer_params is None:
            self._regularizer_params = {}
        if "lambda" not in self._regularizer_params:
            self._regularizer_params["lambda"] = 1.0
        if "lambda_2" not in self._regularizer_params:
            self._regularizer_params["lambda_2"] = 1e-6

    def mode_regularizer(self, X):
        """
        Apply the mode regularizer to the matrix X.
        """
        if isfunction(self._mode_regularizer):
            return self._mode_regularizer(X)

        if self._mode_regularizer == "l0":
            return self._regularizer_params["lambda"] * L0_norm(X)

        if self._mode_regularizer == "l1":
            return self._regularizer_params["lambda"] * L1_norm(X)

        if self._mode_regularizer == "l2":
            return self._regularizer_params["lambda"] * L2_norm(X)

        if self._mode_regularizer == "l02":
            return self._regularizer_params["lambda"] * L0_norm(
                X
            ) + self._regularizer_params["lambda_2"] * L2_norm_squared(X)

        if self._mode_regularizer == "l12":
            return self._regularizer_params["lambda"] * L1_norm(
                X
            ) + self._regularizer_params["lambda_2"] * L2_norm_squared(X)

        raise ValueError("Invalid mode_regularizer provided.")

    def mode_prox(self, X, t):
        """
        Apply the proximal operator to the matrix X with scaling t.
        """
        if isfunction(self._mode_prox):
            return self._mode_prox(X, t)

        if self._mode_regularizer == "l0":
            return hard_threshold(X, self._regularizer_params["lambda"] * t)

        if self._mode_regularizer == "l1":
            return soft_threshold(X, self._regularizer_params["lambda"] * t)

        if self._mode_regularizer == "l2":
            return group_lasso(X, self._regularizer_params["lambda"] * t)

        if self._mode_regularizer == "l02":
            return scaled_hard_threshold(
                X,
                t,
                self._regularizer_params["lambda"],
                self._regularizer_params["lambda_2"],
            )

        if self._mode_regularizer == "l12":
            return scaled_soft_threshold(
                X,
                t,
                self._regularizer_params["lambda"],
                self._regularizer_params["lambda_2"],
            )

        raise ValueError("Invalid mode_regularizer provided.")

    @property
    def mask(self):
        """
        Get the mask used to cover zero variables.
        """
        all_masks = []
        for indices in self.operator.unmasked_indices:
            M = np.ones(self.snapshots.shape[0])
            M[indices] = 0.0
            all_masks.append(M.reshape(self.snapshots_shape))

        return all_masks

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
        if self._proj_basis is None and self._use_proj:
            self._proj_basis = compute_svd(self.snapshots, self._svd_rank)[0]
        elif self._proj_basis is None and not self._use_proj:
            self._proj_basis = compute_svd(self.snapshots, -1)[0]
        elif (
            not isinstance(self._proj_basis, np.ndarray)
            or self._proj_basis.ndim != 2
            or self._proj_basis.shape[1] != self._svd_rank
        ):
            msg = "proj_basis must be a 2D np.ndarray with {} columns."
            raise ValueError(msg.format(self._svd_rank))

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
            self.mode_regularizer,
            self._compute_A,
            self._use_proj,
            self._use_mask,
            self._init_alpha,
            self._init_B,
            self._proj_basis,
            self._num_trials,
            self._trial_size,
            self._eig_sort,
            self._eig_constraints,
            self.mode_prox,
            self._remove_bad_bags,
            self._bag_warning,
            self._bag_maxfail,
            **self._varpro_opts_dict,
        )

        # Fit the data.
        self._b = self.operator.compute_operator(self.snapshots.T, self._time)

        return self
