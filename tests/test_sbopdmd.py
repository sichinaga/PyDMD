import time
import numpy as np
from pytest import raises, warns

from pydmd.bopdmd import BOPDMD
from pydmd.sbopdmd import SparseBOPDMD
from pydmd.sbopdmd_utils import L0_norm, hard_threshold, soft_threshold


def generate_toy_data(
    n: int = 50,
    m: int = 1000,
    f1: float = 2.5,
    f2: float = 6.0,
    dt: float = 0.01,
    sigma: float = 0.5,
):
    """
    Method for generating testing data. Data consists of a Gaussian that
    oscillates with frequency f1 and a step function that oscillates with
    frequency f2. Data is n-dimensional and contains m snapshots.
    """
    time_vals = np.arange(m) * dt

    # Add noise to the data.
    noise_1 = sigma * np.random.default_rng(seed=1234).standard_normal((n, m))
    noise_2 = sigma * np.random.default_rng(seed=5678).standard_normal((n, m))

    # Build the slow Gaussian mode.
    u1 = np.exp(-((np.arange(n) - (n / 2)) ** 2) / (n / 2))
    data_1_clean = np.outer(u1, np.exp(1j * f1 * time_vals))
    data_1 = data_1_clean + noise_1

    # Build the fast square mode.
    u2 = np.zeros(n)
    u2[3 * (n // 10) : 4 * (n // 10)] = 1.0
    data_2_clean = np.outer(u2, np.exp(1j * f2 * time_vals))
    data_2 = data_2_clean + noise_2

    # Build the combined data matrix.
    frequencies = 1j * np.array([f1, f2])
    data_clean = data_1_clean + data_2_clean
    data = data_1 + data_2

    return data, time_vals, frequencies, data_1, data_2, data_clean


# BUILD THE TEST DATA SET:
X, t, true_eigs, X1, X2, X_clean = generate_toy_data()
X_big = generate_toy_data(n=1000)[0]


# DUMMY ARRAY FOR TESTING THRESHOLDING FUNCTIONS:
A = np.array([[-100, -1, 0, 1, 100], [-100j, -1j, 0, 1j, 100j]])


def relative_error(actual: np.ndarray, truth: np.ndarray):
    """Compute relative error."""
    return np.linalg.norm(actual - truth) / np.linalg.norm(truth)


def sort_imag(a: np.ndarray):
    """Sorts the entries of a by imaginary and then real component."""
    sorted_inds = np.argsort(a.imag + 1j * a.real)
    return a[sorted_inds]


def assert_model_accurate(model, eig_tol=0.01, recon_tol=0.1):
    """
    Check that the given model accurately computes system eigenvalues and that
    it accurately reconstructs the noise-free version of the data.
    """
    np.testing.assert_allclose(sort_imag(model.eigs), true_eigs, rtol=eig_tol)
    assert relative_error(model.reconstructed_data, X_clean) < recon_tol


def assert_model_similar(model_1, model_2, mode_tol=0.01, eig_tol=0.01):
    """
    Check that the given models are approximately the same. Checks the
    similarity of the modes and the eigenvalues of the models.
    """
    assert relative_error(model_1.modes, model_2.modes) < mode_tol
    np.testing.assert_allclose(
        sort_imag(model_1.eigs),
        sort_imag(model_2.eigs),
        rtol=eig_tol,
    )


def assert_modes_sparse(model, index_modes=None, sparsity_ratio=0.8):
    """
    Check that the modes specified by index_modes are actually sparse.
    """
    n, r = model.modes.shape

    if index_modes is None:
        index_modes = np.arange(r)

    num_active_features = np.count_nonzero(model.modes[:, index_modes])
    max_active_features = sparsity_ratio * n * len(index_modes)
    assert num_active_features < max_active_features


def test_l0():
    """
    See that "l0" preset uses the expected regularizer and proximal operator.
    """
    s_optdmd = SparseBOPDMD(
        mode_regularizer="l0",
        regularizer_params={"lambda": 5.0},
    )
    # Test the output of the mode regularizer function.
    assert s_optdmd.mode_regularizer(A) == 5.0 * 8

    # Test the output of the proximal operator function.
    np.testing.assert_array_equal(
        s_optdmd.mode_prox(A, 1.0),
        100.0 * np.array([[-1, 0, 0, 0, 1], [-1j, 0, 0, 0, 1j]]),
    )


def test_l1():
    """
    See that "l1" preset uses the expected regularizer and proximal operator.
    """
    s_optdmd = SparseBOPDMD(
        mode_regularizer="l1",
        regularizer_params={"lambda": 5.0},
    )
    # Test the output of the mode regularizer function.
    assert s_optdmd.mode_regularizer(A) == 5.0 * 404

    # Test the output of the proximal operator function.
    np.testing.assert_array_equal(
        s_optdmd.mode_prox(A, 1.0),
        95.0 * np.array([[-1, 0, 0, 0, 1], [-1j, 0, 0, 0, 1j]]),
    )


def test_l2():
    """
    See that "l2" preset uses the expected regularizer and proximal operator.
    """
    s_optdmd = SparseBOPDMD(
        mode_regularizer="l2",
        regularizer_params={"lambda": 5.0},
    )
    # Test the output of the mode regularizer function.
    assert s_optdmd.mode_regularizer(A) == 5.0 * np.sum(
        np.linalg.norm(A, 2, axis=0)
    )

    # Test the output of the proximal operator function.
    a = 100 * (1 - (5.0 / np.linalg.norm([100, 100j], 2)))
    np.testing.assert_array_equal(
        s_optdmd.mode_prox(A, 1.0),
        a * np.array([[-1, 0, 0, 0, 1], [-1j, 0, 0, 0, 1j]]),
    )


def test_l02():
    """
    See that "l02" preset uses the expected regularizer and proximal operator.
    """
    s_optdmd = SparseBOPDMD(
        mode_regularizer="l02",
        regularizer_params={"lambda": 5.0, "lambda_2": 2.0},
    )
    # Test the output of the mode regularizer function.
    assert s_optdmd.mode_regularizer(A) == 5.0 * 8 + 2.0 * 40004

    # Test the output of the proximal operator function.
    np.testing.assert_array_equal(
        s_optdmd.mode_prox(A, 1.0),
        20.0 * np.array([[-1, 0, 0, 0, 1], [-1j, 0, 0, 0, 1j]]),
    )


def test_l12():
    """
    See that "l12" preset uses the expected regularizer and proximal operator.
    """
    s_optdmd = SparseBOPDMD(
        mode_regularizer="l12",
        regularizer_params={"lambda": 5.0, "lambda_2": 2.0},
    )
    # Test the output of the mode regularizer function.
    assert s_optdmd.mode_regularizer(A) == 5.0 * 404 + 2.0 * 40004

    # Test the output of the proximal operator function.
    np.testing.assert_array_equal(
        s_optdmd.mode_prox(A, 1.0),
        19.0 * np.array([[-1, 0, 0, 0, 1], [-1j, 0, 0, 0, 1j]]),
    )


def test_custom_regularizer():
    """
    See that requesting a custom regularizer and proximal operator function
    works as expected.
    """

    # Note: this is functionally the same as using "l0".
    def custom_regularizer(Y):
        return 5.0 * L0_norm(Y)

    def custom_prox(Z, c):
        return hard_threshold(Z, gamma=5.0 * c)

    # Test that an error occurs if both functions aren't provided.
    with raises(ValueError):
        s_optdmd = SparseBOPDMD(mode_regularizer=custom_regularizer)

    # Test functionality of using custom functions.
    s_optdmd = SparseBOPDMD(
        mode_regularizer=custom_regularizer,
        mode_prox=custom_prox,
    )
    # Test the output of the mode regularizer function.
    assert s_optdmd.mode_regularizer(A) == 5.0 * 8

    # Test the output of the proximal operator function.
    np.testing.assert_array_equal(
        s_optdmd.mode_prox(A, 1.0),
        100.0 * np.array([[-1, 0, 0, 0, 1], [-1j, 0, 0, 0, 1j]]),
    )


def test_regularizer_errors():
    """
    See that the appropriate errors and warnings are thrown when invalid
    regularizer parameters are requested.
    """
    # Error should be thrown if an unrecognized preset is given.
    with raises(ValueError):
        _ = SparseBOPDMD(mode_regularizer="l_0")

    # Warning should be thrown if an unrecognized regularizer param is given.
    with warns():
        _ = SparseBOPDMD(
            mode_regularizer="l0",
            regularizer_params={"lambda": 5.0, "lambda2": 2.0},
        )


def test_fit_SR3():
    """
    Test that basic sparse-mode DMD with SR3 updates can accurately compute
    eigenvalues and reconstruct the data (i.e. it produces accurate models).
    Additionally confirm that the modes are actually sparsified.
    """
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 0.001},
    )
    s_optdmd.fit(X, t)
    assert_model_accurate(s_optdmd)
    assert_modes_sparse(s_optdmd)


def test_fit_FISTA():
    """
    Test that basic sparse-mode DMD with prox-gradient can accurately compute
    eigenvalues and reconstruct the data (i.e. it produces accurate models).
    Additionally confirm that the modes are actually sparsified.
    """
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 1.0}, # lambda needs to be updated
        SR3_step=0, # don't use SR3
    )
    s_optdmd.fit(X, t)
    assert_model_accurate(s_optdmd)
    assert_modes_sparse(s_optdmd)


def test_fit_thresh():
    """
    Test that basic sparse-mode DMD with thresholding can accurately compute
    eigenvalues and reconstruct the data (i.e. it produces accurate models).
    Additionally confirm that the modes are actually sparsified.
    """

    def mode_prox(Z):
        return hard_threshold(Z, gamma=0.001)

    s_optdmd = BOPDMD(svd_rank=2, mode_prox=mode_prox)
    s_optdmd.fit(X, t)
    assert_model_accurate(s_optdmd)
    assert_modes_sparse(s_optdmd)


def test_sparsity_SR3():
    """
    Test that increasing the sparsity parameter actually results in sparser
    and sparser modes. This test uses the SR3 model and fits to the Gaussian
    mode of the data, polluted by noise. Tests L0 and L1 norm sparsity.
    """
    # Test for various parameters of the L0 norm.
    n = 50
    for _lambda in [1e-4, 1e-3, 1e-2]:
        s_optdmd = SparseBOPDMD(
            svd_rank=1,
            mode_regularizer="l0",
            regularizer_params={"lambda": _lambda},
        )
        s_optdmd.fit(X1, t)
        assert n > np.count_nonzero(s_optdmd.modes)
        n = np.count_nonzero(s_optdmd.modes)

    # Test for various parameters of the L1 norm.
    n = 50
    for _lambda in [0.01, 0.02, 0.03]:
        s_optdmd = SparseBOPDMD(
            svd_rank=1,
            mode_regularizer="l1",
            regularizer_params={"lambda": _lambda},
        )
        s_optdmd.fit(X1, t)
        assert n > np.count_nonzero(s_optdmd.modes)
        n = np.count_nonzero(s_optdmd.modes)


def test_sparsity_FISTA():
    """
    Test that increasing the sparsity parameter actually results in sparser
    and sparser modes. This test uses the prox-gradient model and fits to the
    Gaussian mode of the data, polluted by noise. Tests L0 and L1 norm.
    """
    # Test for various parameters of the L0 norm.
    n = 50
    for _lambda in [0.1, 1.0, 10.0]:
        s_optdmd = SparseBOPDMD(
            svd_rank=1,
            mode_regularizer="l0",
            regularizer_params={"lambda": _lambda},
            SR3_step=0,
        )
        s_optdmd.fit(X1, t)
        assert n > np.count_nonzero(s_optdmd.modes)
        n = np.count_nonzero(s_optdmd.modes)

    # Test for various parameters of the L1 norm.
    n = 50
    for _lambda in [10.0, 20.0, 30.0]:
        s_optdmd = SparseBOPDMD(
            svd_rank=1,
            mode_regularizer="l1",
            regularizer_params={"lambda": _lambda},
            SR3_step=0,
        )
        s_optdmd.fit(X1, t)
        assert n > np.count_nonzero(s_optdmd.modes)
        n = np.count_nonzero(s_optdmd.modes)


def test_sparsity_thresh():
    """
    Test that increasing the sparsity parameter actually results in sparser
    and sparser modes. This test uses the thresholding model and fits to the
    Gaussian mode of the data, polluted by noise. Tests L0 and L1 norm.
    """
    # Test for various parameters of hard thresholding.
    n = 50
    for _gamma in [1e-4, 1e-3, 1e-2]:

        def mode_prox_hard(Z):
            return hard_threshold(Z, gamma=_gamma)

        s_optdmd = BOPDMD(svd_rank=1, mode_prox=mode_prox_hard)
        s_optdmd.fit(X1, t)
        assert n > np.count_nonzero(s_optdmd.modes)
        n = np.count_nonzero(s_optdmd.modes)

    # Test for various parameters of soft thresholding.
    n = 50
    for _gamma in [0.01, 0.02, 0.03]:

        def mode_prox_soft(Z):
            return soft_threshold(Z, gamma=_gamma)

        s_optdmd = BOPDMD(svd_rank=1, mode_prox=mode_prox_soft)
        s_optdmd.fit(X1, t)
        assert n > np.count_nonzero(s_optdmd.modes)
        n = np.count_nonzero(s_optdmd.modes)


def test_eig_constraints():
    """
    Test eigenvalue constraint functionality and correctness.
    """
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 0.001},
        eig_constraints={"imag"},
    )
    s_optdmd.fit(X, t)
    assert_model_accurate(s_optdmd)
    assert_modes_sparse(s_optdmd)
    assert np.all(s_optdmd.eigs.real == 0.0)


def test_use_optdmd_eigs():
    """
    Test that using the Optimized DMD eigenvalues produces accurate models.
    """
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 0.001},
        varpro_opts_dict={"use_optdmd_eigs": True},
    )
    s_optdmd.fit(X, t)
    assert_model_accurate(s_optdmd)
    assert_modes_sparse(s_optdmd)


def test_use_proj_1():
    """
    Test that fitting WITHOUT projection produces accurate models.
    """
    # (1) Test for the default SR3 model.
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 0.001},
        use_proj=False,
    )
    s_optdmd.fit(X, t)
    assert_model_accurate(s_optdmd)
    assert_modes_sparse(s_optdmd)

    # (2) Test for the FISTA model.
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 1.0},
        SR3_step=0,
        use_proj=False,
    )
    s_optdmd.fit(X, t)
    assert_model_accurate(s_optdmd)
    assert_modes_sparse(s_optdmd)

    # (3) Test for the thresholding model.
    def mode_prox(Z):
        return hard_threshold(Z, gamma=0.001)

    s_optdmd = BOPDMD(svd_rank=2, mode_prox=mode_prox, use_proj=False)
    s_optdmd.fit(X, t)
    assert_model_accurate(s_optdmd)
    assert_modes_sparse(s_optdmd)


def test_use_proj_2():
    """
    Test that using data projection actually reduces fitting time.
    Test using the large toy data set -- use any parameters.
    Tests the default SR3 model.
    """
    t1 = time.time()
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 0.001},
    )
    s_optdmd.fit(X_big, t)
    t2 = time.time()
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 0.001},
        use_proj=False,
    )
    s_optdmd.fit(X_big, t)
    t3 = time.time()

    assert t2 - t1 < t3 - t2


def test_use_proj_3():
    """
    Test that models generated with data projection and models generated
    without approximately produce the same model for various parameters.
    Checks similarity of the modes and the eigenvalues.
    Tests the default SR3 model.
    """
    for _lambda in [1e-4, 1e-3, 1e-2]:
        # Fit model WITH data projection.
        s_optdmd_proj = SparseBOPDMD(
            svd_rank=2,
            mode_regularizer="l0",
            regularizer_params={"lambda": _lambda},
        )
        s_optdmd_proj.fit(X, t)

        # Fit model WITHOUT data projection.
        s_optdmd_noproj = SparseBOPDMD(
            svd_rank=2,
            mode_regularizer="l0",
            regularizer_params={"lambda": _lambda},
            use_proj=False,
        )
        s_optdmd_noproj.fit(X, t)

        # Compare modes and eigenvalues.
        assert_model_similar(s_optdmd_proj, s_optdmd_noproj)


def test_use_proj_4():
    """
    Test that models generated with data projection and models generated
    without approximately produce the same model for various parameters.
    Checks similarity of the modes and the eigenvalues.
    Tests the default FISTA model.
    """
    for _lambda in [0.1, 1.0, 10.0]:
        # Fit model WITH data projection.
        s_optdmd_proj = SparseBOPDMD(
            svd_rank=2,
            mode_regularizer="l0",
            regularizer_params={"lambda": _lambda},
            SR3_step=0,
        )
        s_optdmd_proj.fit(X, t)

        # Fit model WITHOUT data projection.
        s_optdmd_noproj = SparseBOPDMD(
            svd_rank=2,
            mode_regularizer="l0",
            regularizer_params={"lambda": _lambda},
            SR3_step=0,
            use_proj=False,
        )
        s_optdmd_noproj.fit(X, t)

        # Compare modes and eigenvalues.
        assert_model_similar(s_optdmd_proj, s_optdmd_noproj)


def test_feature_tol():
    """
    Test that adjusting the feature tolerance actually reduces fitting time.
    Test using the large toy data set -- results must be dense.
    """
    t1 = time.time()
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 1e-4},
        varpro_opts_dict={"feature_tol": 1e-3},
    )
    s_optdmd.fit(X_big, t)
    t2 = time.time()
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 1e-4},
        varpro_opts_dict={"feature_tol": 0.0},
    )
    s_optdmd.fit(X_big, t)
    t3 = time.time()
    assert t2 - t1 < t3 - t2

