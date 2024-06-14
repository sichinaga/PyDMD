"""
Derived module for space-hierarchical sparse-mode BOP-DMD.
"""

from numbers import Number
from typing import Callable, Union

import numpy as np

from .sbopdmd import SparseBOPDMD
from .sbopdmd_utils import get_nonzero_cols
from .utils import compute_rank

class HSparseBOPDMD:
    """
    Dynamic Mode Decomposition with sparse modes and spatial hierarchy.
    """

    def __init__(
        self,
        global_thres: float = 0.8,
        global_sampling_ratio: float = 0.2,
        local_sampling_ratio: float = 0.5,
        mode_regularizer: Union[str, Callable] = None,
        regularizer_params: dict = None,
        svd_rank: Number = 0,
        use_proj: bool = True,
        use_mask: bool = True,
        eig_constraints: Union[set, Callable] = None,
        mode_prox: Callable = None,
        bopdmd_opts_dict: dict = None,
        varpro_opts_dict: dict = None,
    ):
        self._gobal_thres = global_thres
        self._global_sampling_ratio = global_sampling_ratio
        self._local_sampling_ratio = local_sampling_ratio
        self._svd_rank = svd_rank

        # Build the dictionary of universal SparseBOPDMD parameters.
        self._sparse_opts_dict = {}
        self._sparse_opts_dict["mode_regularizer"] = mode_regularizer
        self._sparse_opts_dict["regularizer_params"] = regularizer_params
        self._sparse_opts_dict["use_proj"] = use_proj
        self._sparse_opts_dict["use_mask"] = use_mask
        self._sparse_opts_dict["eig_constraints"] = eig_constraints
        self._sparse_opts_dict["mode_prox"] = mode_prox

        if bopdmd_opts_dict is None:
            # This dictionary accounts for the following:
            # - compute_A
            # - init_alpha
            # - init_B
            # - proj_basis
            # - num_trials
            # - trial_size
            # - eig_sort
            # - remove_bad_bags
            # - bag_warning
            # - bag_maxfail
            self._bopdmd_opts_dict = {}

        if varpro_opts_dict is None:
            # This dictionary accounts for the following:
            # - init_lambda
            # - maxlam
            # - lamup
            # - maxiter
            # - tol
            # - eps_stall
            # - verbose
            # - prox_grad_tol
            # - prox_grad_maxiter
            # - prox_grad_restart
            # - sampling_rng
            self._varpro_opts_dict = {}

    def fit(self, X, t):
        r = compute_rank(X, self._svd_rank)

        # PHASE 1: Fit a SparseBOPDMD model to the full, original data set.
        sbopdmd = SparseBOPDMD(
            svd_rank=r,
            sampling_ratio=self._global_sampling_ratio,
            varpro_opts_dict=self._varpro_opts_dict,
            **self._sparse_opts_dict,
            **self._bopdmd_opts_dict,
        )
        sbopdmd.fit(X, t)

        # PHASE 2: Check the fitted model for global/local modes.
        local_eigs, local_modes, local_features = self._get_local_info(sbopdmd)

        if local_eigs:
            # PHASE 3: If local features are found, train a second model that
            # narrows in on the local features only.
            # - Use a larger sampling ratio.
            # - Initialize with the eigs and modes already computed.
            # TODO: Change up the dicts -- this will cause an error.
            sbopdmd_l = SparseBOPDMD(
                svd_rank=r,
                sampling_ratio=self._local_sampling_ratio,
                init_alpha=sbopdmd.eigs,
                init_B=(sbopdmd.modes.dot(np.diag(sbopdmd.amplitudes))).T,
                varpro_opts_dict=self._varpro_opts_dict,
                **self._sparse_opts_dict,
                **self._bopdmd_opts_dict,
            )
            sbopdmd_l.fit(X[local_features], t)

            # TODO: Update the local modes.
            # TODO: Subtract the local modes from the data.
            # TODO: Fit a regular BOPDMD model to the global data.

        else:
            


    def _get_local_info(self, dmd):
        """
        :param dmd:
        """
        all_modes = dmd.modes
        all_eigs = dmd.eigs

        N, r = all_modes.shape
        N_active_all = np.count_nonzero(all_modes, axis=0)
        inds_local = np.arange(r)[N_active_all / N < self._gobal_thres]
        local_eigs = all_eigs[inds_local]
        local_modes = all_modes[:, inds_local]
        local_features = get_nonzero_cols(local_modes.T)

        return local_eigs, local_modes, local_features

    def _remove_modes(self, H, t, global_eigs, global_modes):
        """
        Removes
        """
        for eig, mode in zip(global_eigs, global_modes):
            spatiotemporal_mode = np.outer(np.exp(eig * t), mode)
            H = H - spatiotemporal_mode

        if self._use_proj:
            H_proj = H.dot(self._proj_basis.conj())
        else:
            H_proj = None

        return H, H_proj