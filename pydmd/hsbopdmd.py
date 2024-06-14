"""
Derived module for space-hierarchical sparse-mode BOP-DMD.
"""

import warnings
from numbers import Number
from typing import Callable, Union

import numpy as np

from .bopdmd import BOPDMD
from .sbopdmd import SparseBOPDMD
from .sbopdmd_utils import get_nonzero_cols
from .utils import compute_rank


class HSparseBOPDMD:
    """
    Dynamic Mode Decomposition with sparse modes and spatial hierarchy.

    :param global_thres:
    :param global_sampling_ratio:
    :param local_sampling_ratio:
    """

    def __init__(
        self,
        global_thres: float = 0.8,
        global_sampling_ratio: float = 0.1,
        local_sampling_ratio: float = 0.5,
        mode_regularizer: Union[str, Callable] = None,
        regularizer_params: dict = None,
        svd_rank: Number = 0,
        use_proj: bool = True,
        use_mask: bool = True,
        init_alpha: np.ndarray = None,
        init_B: np.ndarray = None,
        eig_constraints: Union[set, Callable] = None,
        mode_prox: Callable = None,
        bopdmd_opts_dict: dict = None,
        varpro_opts_dict: dict = None,
    ):
        self._gobal_thres = global_thres
        self._svd_rank = svd_rank

        self._eigs_global = None
        self._modes_global = None
        self._amplitudes_global = None

        self._eigs_local = None
        self._modes_local = None
        self._amplitudes_local = None

        self._sbopdmd_global = None
        self._sbopdmd_local = None
        self._bopdmd_global = None

        # Dictionary of universal SparseBOPDMD parameters.
        sparse_opts_dict = {}
        sparse_opts_dict["mode_regularizer"] = mode_regularizer
        sparse_opts_dict["regularizer_params"] = regularizer_params
        sparse_opts_dict["mode_prox"] = mode_prox
        sparse_opts_dict["use_mask"] = use_mask
        sparse_opts_dict["varpro_opts_dict"] = varpro_opts_dict

        # Dictionary of SparseBOPDMD parameters for the global level.
        self._global_opts_dict = {}
        self._global_opts_dict["init_alpha"] = init_alpha
        self._global_opts_dict["init_B"] = init_B
        self._global_opts_dict["sampling_ratio"] = global_sampling_ratio
        self._global_opts_dict.update(sparse_opts_dict)

        # Dictionary of SparseBOPDMD parameters for the local level.
        self._local_opts_dict = {}
        self._local_opts_dict["sampling_ratio"] = local_sampling_ratio
        self._local_opts_dict.update(sparse_opts_dict)

        # Dictionary of universal BOPDMD parameters. This includes...
        # - compute_A
        # - use_proj (model parameter)
        # - proj_basis
        # - num_trials
        # - trial_size
        # - eig_sort
        # - eig_constraints (model parameter)
        # - remove_bad_bags
        # - bag_warning
        # - bag_maxfail

        if bopdmd_opts_dict is None:
            self._bopdmd_opts_dict = {}
        else:
            self._bopdmd_opts_dict = bopdmd_opts_dict

        self._bopdmd_opts_dict["use_proj"] = use_proj
        self._bopdmd_opts_dict["eig_constraints"] = eig_constraints

    def fit(self, X, t):
        # Compute the integer rank of the fit.
        r = compute_rank(X, self._svd_rank)

        # PHASE 1: Fit a SparseBOPDMD model to the full, original data set.
        self._sbopdmd_global = SparseBOPDMD(
            svd_rank=r,
            **self._global_opts_dict,
            **self._bopdmd_opts_dict,
        )
        self._sbopdmd_global.fit(X, t)

        # PHASE 2: Check the fitted model for local modes.
        r_local, local_features = self._get_local_info()

        # PHASE 3: If local modes are found, train a second SparseBOPDMD
        # model that narrows in on the local features only.
        if r_local > 0:
            # Features of the local model:
            # - Uses a pre-computed feature mask.
            # - Often uses a larger sampling ratio.
            # - Initialize with the modes already computed.
            self._local_opts_dict["unmasked_features"] = local_features
            self._local_opts_dict["init_alpha"] = self._sbopdmd_global.eigs
            self._local_opts_dict["init_B"] = (
                self._sbopdmd_global.modes.dot(
                    np.diag(self._sbopdmd_global.amplitudes)
                )
            ).T
            self._sbopdmd_local = SparseBOPDMD(
                svd_rank=r,
                **self._local_opts_dict,
                **self._bopdmd_opts_dict,
            )
            self._sbopdmd_local.fit(X, t)

            # The most active modes are the local modes.
            inds_local = np.argsort(-np.abs(self._sbopdmd_local.amplitudes))[
                :r_local
            ]
            self._eigs_local = self._sbopdmd_local.eigs[inds_local]
            self._modes_local = self._sbopdmd_local.modes[:, inds_local]
            self._amplitudes_local = self._sbopdmd_local.amplitudes[inds_local]

            # Subtract the local modes from the data.
            X_global = self._remove_local_modes(X, t)

            # Fit a regular BOPDMD model to the global data.
            r_global = r - r_local
            # Idea: add ability to provide a mode proxy function.
            self._bopdmd_global = BOPDMD(
                svd_rank=r_global,
                **self._bopdmd_opts_dict,
            )
            self._bopdmd_global.fit(X_global, t)

        else:
            warnings.warn(
                "No local modes found. Consider decreasing global_thres, "
                "or consider fitting data to a BOPDMD model instead."
            )
            self._bopdmd_global = BOPDMD(
                svd_rank=r,
                **self._bopdmd_opts_dict,
            )
            self._bopdmd_global.fit(X, t)

    def _get_local_info(self):
        """
        :param dmd:
        """
        all_modes = self._sbopdmd_global.modes
        N, r = all_modes.shape
        N_active_all = np.count_nonzero(all_modes, axis=0)
        inds_local = np.arange(r)[N_active_all / N < self._gobal_thres]
        r_local = len(inds_local)
        local_features = get_nonzero_cols(all_modes[:, inds_local].T)

        return r_local, local_features

    def _remove_local_modes(self, X, t):
        """
        Removes
        """
        for eig, mode, b in zip(
            self._eigs_local, self._modes_local, self._amplitudes_local
        ):
            X = X - np.outer(b * mode, np.exp(eig * t))

        return X
