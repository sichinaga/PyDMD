"""
Derived module from bopdmd.py for BOP-DMD with sparse modes.
"""

import warnings
from numbers import Number
from typing import Callable, Union
from inspect import isfunction

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy.sparse import csr_matrix

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
    support_lstsq,
    FISTA,
    SR3,
)


class SparseBOPDMDOperator(BOPDMDOperator):
    """
    BOP-DMD operator with sparse modes.

    :param feature_tol: Tolerance for terminating the mode solver for
        individual features of the data, as features and their respective
        optimizations are treated as independent. Once a mode feature's
        tolerance is reached, it is no longer updated across variable
        projection iterations.
    :type feature_tol: float
    :param use_optdmd_eigs: Whether or not to compute the Jacobian using the
        formulation used by Optimized DMD. By default, this formulation is not
        used and the Jacobian expression for a generic mode matrix is used.
    :type use_optdmd_eigs: bool
    :param mode_opt_params: Dictionary of optional parameters for either the
        `SR3` or `FISTA` functions. See the documentation for these functions
        found in `sbopdmd_utils.py` for more information.
    :type mode_opt_params: dict
    :param global_mode_params: Dictionary of optional parameters for the
        `_get_global_modes` function in `bopdmd.py`. Accounts for parameters
        `active_thresh` (default = 0.1) and `global_thresh` (default = 0.5).
    :type global_mode_params: dict
    """

    def __init__(
        self,
        mode_regularizer,
        SR3_step,
        apply_debias,
        compute_A,
        use_proj,
        init_alpha,
        init_B,
        proj_basis,
        num_trials,
        trial_size,
        eig_sort,
        eig_constraints,
        mode_prox,
        index_global,
        remove_bad_bags,
        bag_warning,
        bag_maxfail,
        real_eig_limit,
        init_lambda=1.0,
        maxlam=52,
        lamup=2.0,
        maxiter=30,
        tol=1e-6,
        eps_stall=1e-12,
        verbose=False,
        varpro_flag=True,
        feature_tol=0.0,
        use_optdmd_eigs=False,
        mode_opt_params=None,
        global_mode_params=None,
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
            index_global=index_global,
            remove_bad_bags=remove_bad_bags,
            bag_warning=bag_warning,
            bag_maxfail=bag_maxfail,
            real_eig_limit=real_eig_limit,
            init_lambda=init_lambda,
            maxlam=maxlam,
            lamup=lamup,
            maxiter=maxiter,
            tol=tol,
            eps_stall=eps_stall,
            verbose=verbose,
            varpro_flag=varpro_flag,
        )
        self._mode_regularizer = mode_regularizer
        self._SR3_step = SR3_step
        self._apply_debias = apply_debias
        self._init_B = init_B

        # General variable projection parameters.
        self._maxiter = maxiter
        self._tol = tol
        self._eps_stall = eps_stall
        self._verbose = verbose
        self._feature_tol = feature_tol
        self._unconverged_features = None

        # Set the parameters of Levenberg-Marquardt.
        self._init_lambda = init_lambda
        self._lev_marq_params = {}
        self._lev_marq_params["maxlam"] = maxlam
        self._lev_marq_params["lamup"] = lamup
        self._lev_marq_params["use_optdmd_eigs"] = use_optdmd_eigs

        if self._apply_debias:
            self._lev_marq_params["use_optdmd_eigs"] = True

        # Set the parameters of SR3 or FISTA.
        if mode_opt_params is None:
            self._mode_opt_params = {}
        else:
            self._mode_opt_params = mode_opt_params

        self._obj_history = []
        self._err_history = []

        # Set the parameters of get_global_modes.
        if global_mode_params is None:
            self._global_mode_params = {}
        else:
            self._global_mode_params = global_mode_params

    def _compute_B(self, B0, alpha, H, t, Phi):
        """
        Use accelerated prox-gradient to update B for the current alpha.
        """
        A = Phi(alpha, t)

        # Get the indices at which to apply sparsity.
        index_local = np.ones(len(B0), dtype=bool)

        if self._index_global is not None:
            if self._index_global == "auto":
                index_global = self._get_global_modes(
                    B0, **self._global_mode_params
                )
            else:
                index_global = self._index_global.copy()

            index_local[index_global] = False

        if self._SR3_step > 0.0:
            # Apply Sparse Relaxed Regularized Regression (SR3).
            B_updated, obj_hist, err_hist = SR3(
                nu=self._SR3_step,
                W0=B0,
                A=A,
                B=H,
                func_g=self._mode_regularizer,
                prox_g=self._mode_prox,
                sparse_inds=index_local,
                **self._mode_opt_params,
            )

            if self._apply_debias:
                B_updated = support_lstsq(S=B_updated, A=A, B=H)

        else:
            # Apply proximal gradient directly to the B matrix (FISTA).
            beta_f = np.linalg.norm(A, 2) ** 2

            def func_f(Z):
                return 0.5 * np.linalg.norm(H - A.dot(Z), "fro") ** 2

            def grad_f(Z):
                return A.conj().T.dot(A.dot(Z) - H)

            B_updated, obj_hist, err_hist = FISTA(
                X0=B0,
                func_f=func_f,
                func_g=self._mode_regularizer,
                grad_f=grad_f,
                prox_g=self._mode_prox,
                beta_f=beta_f,
                sparse_inds=index_local,
                **self._mode_opt_params,
            )

        # Save the objective and error history.
        self._obj_history.append(obj_hist)
        self._err_history.append(err_hist)

        # Plot the progress of the B update.
        if self._verbose and len(self._obj_history) <= 3:
            if self._SR3_step > 0.0:
                print("SR3 Results:")
            else:
                print("FISTA Results:")

            plt.figure(figsize=(8, 2))
            plt.subplot(1, 2, 1)
            plt.title("Objective")
            plt.plot(obj_hist, ".-", c="k", mec="b", mfc="b")
            plt.semilogy()
            plt.subplot(1, 2, 2)
            plt.title("Error")
            plt.plot(err_hist, ".-", c="k", mec="r", mfc="r")
            plt.semilogy()
            plt.tight_layout()
            plt.show()

        return B_updated

    def _compute_levmarq(
        self,
        alpha,
        B,
        H,
        init_lambda,
        t,
        Phi,
        dPhi,
        maxlam,
        lamup,
        use_optdmd_eigs,
    ):
        """
        Use Levenberg-Marquardt to step alpha and its corresponding B matrix.
        """

        def compute_residual(alpha, B):
            """
            Helper function that, when given the current alpha (eigenvalues)
            and B matrix (modes), computes and returns the residual.
            Distinguishes between the compressed and uncompressed cases.
            """
            if self._use_proj:
                B_proj = B.dot(self._proj_basis.conj())
                return H_proj - Phi(alpha, t).dot(B_proj)

            return H - Phi(alpha, t).dot(B)

        # Set the damping parameter.
        _lambda = init_lambda

        # Define M, IS, and IA.
        if self._use_proj:
            H_proj = H.dot(self._proj_basis.conj())
            M, IS = H_proj.shape
        else:
            M, IS = H.shape

        IA = len(alpha)

        # Compute the SVD of Phi(alpha, t) if using OptDMD approximation.
        if use_optdmd_eigs:
            U = self._compute_irank_svd(
                Phi(alpha, t),
                tolrank=M * np.finfo(float).eps,
            )[0]

        # Initialize storage for Jacobian computations.
        djac_matrix = np.zeros((M * IS, IA), dtype="complex")
        rjac = np.zeros((2 * IA, IA), dtype="complex")
        scales = np.zeros(IA)

        # Initialize the current objective and residual.
        residual = compute_residual(alpha, B)
        objective = np.linalg.norm(residual, "fro") ** 2
        objective += self._mode_regularizer(B)

        for i in range(IA):
            # Build the Jacobian matrix by looping over all alpha indices.
            if use_optdmd_eigs:
                dphi_temp = dPhi(alpha, t, i)
                uut_dphi = csr_matrix(U @ csr_matrix(U.conj().T @ dphi_temp))
                if self._use_proj:
                    B_proj = B.dot(self._proj_basis.conj())
                    djac_approx = (dphi_temp - uut_dphi) @ B_proj
                else:
                    djac_approx = (dphi_temp - uut_dphi) @ B
                djac_matrix[:, i] = djac_approx.ravel(order="F")
            elif self._use_proj:
                B_proj = B.dot(self._proj_basis.conj())
                djac_matrix[:, i] = (
                    dPhi(alpha, t, i).dot(B_proj).ravel(order="F")
                )
            else:
                djac_matrix[:, i] = dPhi(alpha, t, i).dot(B).ravel(order="F")

            # Scale for the Levenberg-Marquardt algorithm.
            scales[i] = min(np.linalg.norm(djac_matrix[:, i]), 1)
            scales[i] = max(scales[i], 1e-6)

        # Loop to determine lambda (the step-size parameter).
        rhs_temp = residual.ravel(order="F")[:, None]
        q_out, djac_out, j_pvt = qr(djac_matrix, mode="economic", pivoting=True)
        if not self._varpro_flag:
            # The original python, which is a "mistake" that makes bopdmd
            # behave more like exact DMD but also keeps the solver from
            # wandering into bad states.
            ij_pvt = np.arange(IA)
            ij_pvt = ij_pvt[j_pvt]
        else:  # self._varpro_flag
            # Use the true variable projection.
            ij_pvt = np.zeros(IA, dtype=int)
            ij_pvt[j_pvt] = np.arange(IA, dtype=int)

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
            alpha_updated = alpha.ravel() + delta.ravel()
            alpha_updated = self._push_eigenvalues(alpha_updated)

            return alpha_updated, delta

        # Take a step using our initial step size init_lambda.
        alpha_new, _ = step(_lambda)

        # Compute the updated (sparse, full-dimensional) B matrix.
        B_new = np.copy(B)
        B_new[:, self._unconverged_features] = self._compute_B(
            B[:, self._unconverged_features],
            alpha_new,
            H[:, self._unconverged_features],
            t,
            Phi,
        )

        # Compute the corresponding residual for the new update.
        residual_new = compute_residual(alpha_new, B_new)
        objective_new = np.linalg.norm(residual_new, "fro") ** 2
        objective_new += self._mode_regularizer(B_new)

        if objective_new < objective:
            # # Rescale lambda based on the improvement ratio.
            # actual_improvement = objective - objective_new
            # pred_improvement = (_lambda**2) * np.linalg.multi_dot(
            #     [delta.conj().T, np.diag(scales_pvt**2), delta]
            # ).real
            # pred_improvement -= np.linalg.multi_dot(
            #     [delta.conj().T, djac_matrix.conj().T, rhs_temp]
            # )[0].real
            # improvement_ratio = actual_improvement / pred_improvement
            # _lambda *= max(1 / 3, 1 - (2 * improvement_ratio - 1) ** 3)
            _lambda /= 3
            return alpha_new, B_new, _lambda

        # Increase lambda until something works.
        for _ in range(maxlam):
            _lambda *= lamup
            alpha_new, _ = step(_lambda)
            B_new = np.copy(B)
            B_new[:, self._unconverged_features] = self._compute_B(
                B[:, self._unconverged_features],
                alpha_new,
                H[:, self._unconverged_features],
                t,
                Phi,
            )
            residual_new = compute_residual(alpha_new, B_new)
            objective_new = np.linalg.norm(residual_new, "fro") ** 2
            objective_new += self._mode_regularizer(B_new)

            # If the objective improved, terminate.
            if objective_new < objective:
                return alpha_new, B_new, _lambda

            # If failure, remove the recorded B update information.
            del self._obj_history[-1]
            del self._err_history[-1]

        # Terminate if no appropriate step length was found...
        if self._verbose:
            print(
                "Failed to find appropriate LM step length. "
                "Consider increasing maxlam or changing lamup.\n"
            )

        return alpha, B, _lambda

    def _variable_projection(self, H, t, init_alpha, Phi, dPhi):
        """
        Variable projection routine for multivariate data with regularization.
        """

        def get_objective(alpha, B):
            """
            Compute the current objective.
            """
            residual = H - Phi(alpha, t).dot(B)
            objective = 0.5 * np.linalg.norm(residual, "fro") ** 2
            objective += self._mode_regularizer(B)

            return objective

        # Set the Levenberg-Marquardt parameters.
        _lambda = self._init_lambda
        self._lev_marq_params["t"] = t
        self._lev_marq_params["Phi"] = Phi
        self._lev_marq_params["dPhi"] = dPhi

        # Initialize alpha.
        alpha = self._push_eigenvalues(init_alpha)

        # Initialize B.
        if self._init_B is None:
            B = np.linalg.lstsq(Phi(alpha, t), H, rcond=None)[0]
            B = self._compute_B(B, alpha, H, t, Phi)
        else:
            B = np.copy(self._init_B)

        # Initialize storage for objective values and error.
        # Note: "error" refers to differences in iterations.
        all_obj = np.empty(self._maxiter)
        all_err = np.empty(self._maxiter)

        # Initialize termination flags.
        self._unconverged_features = np.ones(H.shape[1], dtype=bool)
        converged = False
        stalled = False

        for itr in range(self._maxiter):

            # Take a Levenberg-Marquardt step to update alpha.
            alpha_new, B_new, _lambda = self._compute_levmarq(
                alpha, B, H, _lambda, **self._lev_marq_params
            )

            # Get new objective and error values.
            err_alpha = np.linalg.norm(alpha - alpha_new)
            err_B = np.linalg.norm(B - B_new)
            all_obj[itr] = get_objective(alpha_new, B_new)
            all_err[itr] = err_alpha + err_B

            # Get feature indices at which very little change occured.
            self._unconverged_features = (
                np.linalg.norm(B - B_new, axis=0) ** 2 > self._feature_tol
            )

            # Update information.
            np.copyto(alpha, alpha_new)
            np.copyto(B, B_new)

            # Print iterative progress if the verbose flag is turned on.
            if self._verbose:
                print(
                    f"Iteration {itr + 1}: "
                    f"Objective = {np.round(all_obj[itr], decimals=4)} "
                    f"Error (alpha) = {np.round(err_alpha, decimals=4)} "
                    f"Error (B) = {np.round(err_B, decimals=4)}.\n"
                )

            # Update termination status and terminate if converged or stalled.
            converged = all_err[itr] < self._tol
            # Note: we may need to change to abs if this stall condition causes
            # too many early terminations due to worsening objectives.
            stalled = (itr > 0) and (
                all_obj[itr - 1] - all_obj[itr]
                < self._eps_stall * all_obj[itr - 1]
            )

            if converged:
                if self._verbose:
                    print("Convergence reached!")
                return B, alpha, converged

            if stalled:
                if self._verbose:
                    msg = (
                        "Stall detected: objective reduced by less than {} "
                        "times the objective at the previous step. "
                        "Iteration {}. Current objective {}. "
                        "Consider decreasing eps_stall."
                    )
                    print(msg.format(self._eps_stall, itr + 1, all_obj[itr]))
                return B, alpha, converged

        # Failed to meet tolerance in maxiter steps.
        if self._verbose:
            msg = (
                "Failed to reach tolerance after maxiter = {} iterations. "
                "Current error {}."
            )
            print(msg.format(self._maxiter, all_err[itr]))

        return B, alpha, converged

    def _single_trial_compute_operator(self, H, t, init_alpha):
        """
        Helper function that computes the standard optimized dmd operator.
        Returns the resulting DMD modes, eigenvalues, amplitudes, reduced
        system matrix, full system matrix, and whether or not convergence
        of the variable projection routine was reached.
        """
        B, alpha, converged = self._variable_projection(
            H,
            t,
            init_alpha,
            self._exp_function,
            self._exp_function_deriv,
        )

        # Save the modes, eigenvalues, and amplitudes.
        B, b = self._split_B(B)
        e = alpha
        w = B.T

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


class SparseBOPDMD(BOPDMD):
    """
    Bagging, Optimized Dynamic Mode Decomposition with sparse modes.

    :param mode_regularizer: Regularizer portion of the objective function
        given matrix input X. May be a function, or one of the following preset
        regularizer options. Note that if a preset regularizer is used, the
        `mode_prox` function will be precomputed based on the chosen preset and
        will not need to be provided by the user. Use the `regularizer_params`
        option to set regularizer parameters if using a preset.
        - "l0": scaled L0 norm
        - "l1": scaled L1 norm
        - "l2": scaled L2 norm
        - "l02": scaled L0 norm + scaled L2 norm squared
        - "l12": scaled L1 norm + scaled L2 norm squared
    :type mode_regularizer: str or function
    :param regularizer_params: Dictionary of parameters for the mode
        regularizer, to be used if a preset regularizer is requested.
        Accounts for the following parameters:
        - "lambda": Scaling for the first norm term (used by all presets).
            Defaults to 1.0.
        - "lambda_2": Scaling for the second norm term (used by "l02", "l12").
            Defaults to 1e-6.
    :type regularizer_params: dict
    :param index_global: DMD mode indices at which modes are assumed to be
        global. Global modes are not sparsified when applying the Sparse-Mode
        DMD pipeline via mode_prox, hence this parameter is not used if
        mode_prox is not provided. By default, all modes are sparsified.
    :type index_global: list
    :param SR3_step: Relaxation parameter for the Sparse Relaxed Regularized
        Regression (SR3) routine. Smaller values lead to a smaller gap between
        the computed modes and the auxiliary SR3 variable. If not given (or if
        less than or equal to zero), proximal gradient is applied directly to
        the modes and SR3 is not used. By default, SR3 is used.
    :type SR3_step: float
    :param apply_debias: Whether or not to apply a de-biasing step following
        the application of the sparse mode update. This entails re-computing
        least squares on the optimal support computed by SR3.
    :type apply_debias: bool
    :param init_B: Initial guess for the amplitude-scaled DMD modes.
        Defaults to using the relationship H = Phi(init_alpha)init_B.
    :type init_B: numpy.ndarray
    :param mode_prox: Proximal operator associated with the `mode_regularizer`
        function, which takes matrix input X and a constant float. Note that
        this parameter must be provided if `mode_regularizer` is given as a
        custom function.
    :type mode_prox: function
    """

    def __init__(
        self,
        mode_regularizer: Union[str, Callable] = "l1",
        regularizer_params: dict = None,
        index_global: Union[list, str] = None,
        SR3_step: float = 1.0,
        apply_debias: bool = False,
        svd_rank: Number = 0,
        compute_A: bool = False,
        use_proj: bool = True,
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
        real_eig_limit: float = None,
        varpro_flag: bool = True,
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
            index_global=index_global,
            remove_bad_bags=remove_bad_bags,
            bag_warning=bag_warning,
            bag_maxfail=bag_maxfail,
            varpro_opts_dict=varpro_opts_dict,
            real_eig_limit=real_eig_limit,
            varpro_flag=varpro_flag,
        )
        self._mode_regularizer = mode_regularizer
        self._regularizer_params = regularizer_params
        self._SR3_step = SR3_step
        self._apply_debias = apply_debias
        self._init_B = init_B

        # Ensure the validity of the given mode regularizer.
        supported_regularizers = ("l0", "l1", "l2", "l02", "l12")
        if (
            isinstance(self._mode_regularizer, str)
            and self._mode_regularizer not in supported_regularizers
        ):
            raise ValueError(
                "Invalid mode_regularizer preset provided. "
                f"Please choose from one of {supported_regularizers}."
            )
        if isfunction(self._mode_regularizer) and self._mode_prox is None:
            raise ValueError(
                "Custom mode_regularizer was provided without mode_prox. "
                "Please provide the corresponding proximal operator function."
            )

        # Set the parameters of the preset regularizer.
        if self._regularizer_params is None:
            self._regularizer_params = {}
        if "lambda" not in self._regularizer_params:
            self._regularizer_params["lambda"] = 1.0
        if "lambda_2" not in self._regularizer_params:
            self._regularizer_params["lambda_2"] = 1e-6
        if self._regularizer_params.keys() - ["lambda", "lambda_2"]:
            warnings.warn(
                "Parameters other than 'lambda' and 'lambda_2' were provided. "
                "These extra parameters will be ignored, so please be sure to "
                "set the parameters 'lambda' and/or 'lambda_2'."
            )

    def mode_regularizer(self, X: np.ndarray):
        """
        Apply the mode regularizer function to the matrix X.

        :param X: (n, m) numpy array.
        :type X: numpy.ndarray
        :return: the value of the regularizer function applied to X.
        :rtype: float
        """
        # Simply use mode_regularizer if it was given as a function.
        if isfunction(self._mode_regularizer):
            return self._mode_regularizer(X)

        # Define the mode regularizer function using a preset.
        _lambda = self._regularizer_params["lambda"]
        _lambda_2 = self._regularizer_params["lambda_2"]

        if self._mode_regularizer == "l0":
            return _lambda * L0_norm(X)

        if self._mode_regularizer == "l1":
            return _lambda * L1_norm(X)

        if self._mode_regularizer == "l2":
            return _lambda * L2_norm(X)

        if self._mode_regularizer == "l02":
            return _lambda * L0_norm(X) + _lambda_2 * L2_norm_squared(X)

        if self._mode_regularizer == "l12":
            return _lambda * L1_norm(X) + _lambda_2 * L2_norm_squared(X)

    def mode_prox(self, X: np.ndarray, t: float):
        """
        Apply the proximal operator function to the matrix X with scaling t.

        :param X: (n, m) numpy array.
        :type X: numpy.ndarray
        :param t: proximal operator scaling.
        :type t: float
        :return: (n, m) numpy array of thresholded values.
        :rtype: numpy.ndarray
        """
        # Simply use mode_prox if it was given as a function.
        if isfunction(self._mode_prox):
            return self._mode_prox(X, t)

        # Define the proximal operator function using a preset.
        _lambda = self._regularizer_params["lambda"]
        _lambda_2 = self._regularizer_params["lambda_2"]

        if self._mode_regularizer == "l0":
            return hard_threshold(X, _lambda * t)

        if self._mode_regularizer == "l1":
            return soft_threshold(X, _lambda * t)

        if self._mode_regularizer == "l2":
            return group_lasso(X, _lambda * t)

        if self._mode_regularizer == "l02":
            return scaled_hard_threshold(X, t, _lambda, _lambda_2)

        if self._mode_regularizer == "l12":
            return scaled_soft_threshold(X, t, _lambda, _lambda_2)

    def fit(self, X, t):
        """
        Compute the Optimized Dynamic Mode Decomposition with sparse modes.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param t: the input time vector.
        :type t: numpy.ndarray or iterable
        """
        # Process the input data and convert to numpy.ndarrays.
        self._reset()
        X = X.astype(complex)  # use complex data types
        self._snapshots_holder = Snapshots(X)
        self._time = np.array(t).squeeze()

        # Check that input time vector is one-dimensional.
        if self._time.ndim > 1:
            raise ValueError("Input time vector t must be one-dimensional.")

        # Check that the number of snapshots in the data matrix X matches the
        # number of time points in the time vector t.
        if self.snapshots.shape[1] != len(self._time):
            raise ValueError(
                "The number of columns in the data matrix X must match "
                "the number of time points in the time vector t."
            )

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

        # Build the Sparse-Mode BOP-DMD operator now that the initial alpha and
        # the projection basis have been defined.
        self._Atilde = SparseBOPDMDOperator(
            self.mode_regularizer,
            self._SR3_step,
            self._apply_debias,
            self._compute_A,
            self._use_proj,
            self._init_alpha,
            self._init_B,
            self._proj_basis,
            self._num_trials,
            self._trial_size,
            self._eig_sort,
            self._eig_constraints,
            self.mode_prox,
            self._index_global,
            self._remove_bad_bags,
            self._bag_warning,
            self._bag_maxfail,
            self._real_eig_limit,
            varpro_flag=self._varpro_flag,
            **self._varpro_opts_dict,
        )

        # Fit the data.
        self._b = self.operator.compute_operator(self.snapshots.T, self._time)

        return self

    def fit_econ(self, s, V, t):
        raise NotImplementedError("This function has not been implemented yet.")
