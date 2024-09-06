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


class SparseBOPDMDOperator(BOPDMDOperator):
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
        sampling_ratio,
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
        sampling_rng=None,
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
        self._unmasked_features = None

        # Parameters for random feature sampling.
        self._sampling_ratio = sampling_ratio
        if sampling_rng is None:
            self._sampling_rng = np.random.default_rng()
        else:
            self._sampling_rng = sampling_rng

        # General variable projection parameters.
        self._maxiter = maxiter
        self._tol = tol
        self._eps_stall = eps_stall
        self._verbose = verbose

        # Set the parameters of Levenberg-Marquardt.
        self._lev_marq_params = {}
        self._lev_marq_params["init_lambda"] = init_lambda
        self._lev_marq_params["maxlam"] = maxlam
        self._lev_marq_params["lamup"] = lamup

        # Set the parameters of accelerated prox-gradient descent.
        self._prox_grad_params = {}
        self._prox_grad_params["tol"] = prox_grad_tol
        self._prox_grad_params["max_iter"] = prox_grad_maxiter
        self._prox_grad_params["use_restarts"] = prox_grad_restart

    @property
    def unmasked_features(self):
        """
        Get the indices of the active features.

        :return: the indices of the active features.
        :rtype: numpy.ndarray
        """
        return self._unmasked_features

    def _compute_B(self, B0, alpha, H, t, Phi):
        """
        Use accelerated prox-gradient to update B for the current alpha.
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

        if self._verbose:
            print("Prox Gradient Results:")
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

    def _compute_alpha_levmarq(
        self, B, alpha_0, H, t, Phi, dPhi, init_lambda, maxlam, lamup
    ):
        """
        Use Levenberg-Marquardt to step alpha for the current B.
        """
        # Define M, IS, and IA.
        M, IS = H.shape
        IA = len(alpha_0)

        # Initialize storage for Jacobian computations.
        djac_matrix = np.zeros((M * IS, IA), dtype="complex")
        rjac = np.zeros((2 * IA, IA), dtype="complex")
        scales = np.zeros(IA)

        # Initialize the current objective and residual.
        # Note: Here, we only use the objective to compare the quality of
        # different alpha results, hence we omit the regularizer portion.
        residual = H - Phi(alpha_0, t).dot(B)
        objective = np.linalg.norm(residual, "fro") ** 2

        for i in range(IA):
            # Build the Jacobian matrix by looping over all alpha indices.
            djac_matrix[:, i] = dPhi(alpha_0, t, i).dot(B).ravel(order="F")
            # Scale for the Levenberg-Marquardt algorithm.
            scales[i] = min(np.linalg.norm(djac_matrix[:, i]), 1)
            scales[i] = max(scales[i], 1e-6)

        # Loop to determine lambda (the step-size parameter).
        rhs_temp = residual.ravel(order="F")[:, None]
        q_out, djac_out, j_pvt = qr(djac_matrix, mode="economic", pivoting=True)
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
            residual_new = H - Phi(alpha_new, t).dot(B)
            objective_new = np.linalg.norm(residual_new, "fro") ** 2

            # If the objective improved, terminate.
            if objective_new < objective:
                return alpha_new

        # Terminate if no appropriate step length was found...
        if self._verbose:
            print(
                "Failed to find appropriate LM step length. "
                "Consider increasing maxlam or changing lamup.\n"
            )

        return alpha_0

    def _variable_projection(self, H, t, init_alpha, Phi, dPhi):
        """
        Variable projection routine for multivariate data with regularization.
        """

        def get_objective(B, alpha):
            """
            Compute the current objective.
            """
            residual = H - Phi(alpha, t).dot(B)
            objective = 0.5 * np.linalg.norm(residual, "fro") ** 2
            objective += self._mode_regularizer(B)

            # Scale by the number of features.
            objective /= N

            return objective

        # Set additional Levenberg-Marquardt parameters.
        self._lev_marq_params["t"] = t
        self._lev_marq_params["Phi"] = Phi
        self._lev_marq_params["dPhi"] = dPhi

        # Record the original data set size (number of features).
        N = H.shape[1]

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

        # Initialize storage for objective values and error.
        # Note: "error" refers to differences in iterations.
        all_obj = np.empty(self._maxiter)
        all_err = np.empty(self._maxiter)

        # Initialize termination flags.
        converged = False
        stalled = False

        for itr in range(self._maxiter):
            # Obtain a sample of the feature indices for speed-up.
            if self._use_mask:
                valid_feature_inds = get_nonzero_cols(B)
                self._unmasked_features = valid_feature_inds
            else:
                valid_feature_inds = np.arange(N)

            # Number of features that will be updated this iteration.
            sample_size = min(
                len(valid_feature_inds),
                int(N * self._sampling_ratio),
            )

            # Indices of the features that will be updated this iteration.
            sampled_inds = self._sampling_rng.choice(
                valid_feature_inds,
                size=sample_size,
                replace=False,
            )

            # Get the new optimal matrix B.
            B_new = np.copy(B)
            B_new[:, sampled_inds] = self._compute_B(
                B[:, sampled_inds], alpha, H[:, sampled_inds], t, Phi
            )

            # Take a Levenberg-Marquardt step to update alpha.
            if self._use_proj:
                B_new_proj = B_new.dot(self._proj_basis.conj())
                alpha_new = self._compute_alpha_levmarq(
                    B_new_proj, alpha, H_proj, **self._lev_marq_params
                )
            else:
                alpha_new = self._compute_alpha_levmarq(
                    B_new, alpha, H, **self._lev_marq_params
                )

            # Get new objective and error values.
            err_alpha = np.linalg.norm(alpha - alpha_new)
            err_B = np.linalg.norm(B - B_new)
            all_obj[itr] = get_objective(B_new, alpha_new)
            all_err[itr] = err_alpha + err_B

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
    :param use_mask: Flag that determines whether or not to ignore features
        that are deemed inactive during the variable projection routine. If
        `True`, features that are eliminated due to sparsity are ignored during
        future variable projection iterations. If `False`, all features are
        updated at every iteration of variable projection.
    :type use_mask: bool
    :param init_B: Initial guess for the amplitude-scaled DMD modes.
        Defaults to using the relationship H = Phi(init_alpha)init_B.
    :type init_B: numpy.ndarray
    :param sampling_ratio: Size of the subset of mode features to update at
        each iteration of variable projection. Must be a value within the range
        (0.0, 1.0], in which case int(sampling_ratio * n) random features will
        be updated per iteration, where n denotes the total number of features.
        If `use_mask=True` and the number of unmasked features becomes smaller
        than int(sampling_ratio * n), then all of the unmasked features will be
        updated at each iteration. Defaults to 1.0.
    :type sampling_ratio: float
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
        svd_rank: Number = 0,
        compute_A: bool = False,
        use_proj: bool = True,
        use_mask: bool = True,
        init_alpha: np.ndarray = None,
        init_B: np.ndarray = None,
        proj_basis: np.ndarray = None,
        sampling_ratio: float = 1.0,
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
        self._sampling_ratio = sampling_ratio

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

    @property
    def mask(self):
        """
        Get the mask used to cover inactive features. Inactive features are
        denoted with 1.0 and active features are denoted with 0.0.

        :return: the mask used to cover inactive features.
        :rtype: numpy.ndarray
        """
        M = np.ones(self.snapshots.shape[0])
        M[self.operator.unmasked_features] = 0.0
        M = M.reshape(self.snapshots_shape)

        return M

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
            self._compute_A,
            self._use_proj,
            self._use_mask,
            self._init_alpha,
            self._init_B,
            self._proj_basis,
            self._sampling_ratio,
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
