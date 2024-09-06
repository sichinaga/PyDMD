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

    def custom_prox(Z):
        return hard_threshold(Z, gamma=5.0)

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


def test_fit_prox():
    """
    Test that basic sparse-mode DMD with prox-gradient can accurately compute
    eigenvalues and reconstruct the data (i.e. it produces accurate models).
    """
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 1.0},
    )
    s_optdmd.fit(X, t)
    np.testing.assert_allclose(sort_imag(s_optdmd.eigs), true_eigs, rtol=0.01)
    assert relative_error(s_optdmd.reconstructed_data, X_clean) < 0.1


def test_fit_thresh():
    """
    Test that basic sparse-mode DMD with thresholding can accurately compute
    eigenvalues and reconstruct the data (i.e. it produces accurate models).
    """

    def mode_prox(Z):
        return hard_threshold(Z, gamma=0.001)

    s_optdmd = BOPDMD(svd_rank=2, mode_prox=mode_prox)
    s_optdmd.fit(X, t)
    np.testing.assert_allclose(sort_imag(s_optdmd.eigs), true_eigs, rtol=0.01)
    assert relative_error(s_optdmd.reconstructed_data, X_clean) < 0.1


def test_sparsity_1():
    """
    Test that the modes produced by sparse-mode DMD are actually sparse
    compared to the modes produced by regular DMD. Here, we fit only to
    the square mode of the data, polluted by noise.
    """
    # (0) Fit a regular OptDMD model to the data.
    optdmd = BOPDMD(svd_rank=1)
    optdmd.fit(X2, t)

    # (1) Fit a prox-gradient model to the data.
    s_optdmd_1 = SparseBOPDMD(
        svd_rank=1,
        mode_regularizer="l0",
        regularizer_params={"lambda": 1.0},
    )
    s_optdmd_1.fit(X2, t)

    # (2) Fit a thresholding model to the data.
    def mode_prox(Z):
        return hard_threshold(Z, gamma=0.001)

    s_optdmd_2 = BOPDMD(svd_rank=1, mode_prox=mode_prox)
    s_optdmd_2.fit(X2, t)

    # Finally, compare number of nonzero entries.
    assert np.count_nonzero(optdmd.modes) > np.count_nonzero(s_optdmd_1.modes)
    assert np.count_nonzero(optdmd.modes) > np.count_nonzero(s_optdmd_2.modes)


def test_sparsity_2():
    """
    Test that increasing the sparsity parameter actually results in sparser
    and sparser modes. This test uses the prox-gradient model and fits to the
    Gaussian mode of the data, polluted by noise.
    """
    # Test for various parameters of the L0 norm.
    n = 50
    for _lambda in [0.1, 1.0, 10.0]:
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
    for _lambda in [10.0, 20.0, 30.0]:
        s_optdmd = SparseBOPDMD(
            svd_rank=1,
            mode_regularizer="l1",
            regularizer_params={"lambda": _lambda},
        )
        s_optdmd.fit(X1, t)
        assert n > np.count_nonzero(s_optdmd.modes)
        n = np.count_nonzero(s_optdmd.modes)


def test_sparsity_3():
    """
    Test that increasing the sparsity parameter actually results in sparser
    and sparser modes. This test uses the thresholding model and fits to the
    Gaussian mode of the data, polluted by noise.
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


def test_sparsity_4():
    """
    Test that increasing the sparsity parameter actually results in sparser
    and sparser modes. This test uses the sequential thresholding model and
    fits to the Gaussian mode of the data, polluted by noise.
    """
    n = 50
    for _gamma in [1e-4, 1e-3, 1e-2]:

        def mode_prox(Z):
            return hard_threshold(Z, gamma=_gamma)

        s_optdmd = BOPDMD(
            svd_rank=1,
            mode_prox=mode_prox,
            varpro_opts_dict={"stlsq_opts": {}},
        )
        s_optdmd.fit(X1, t)
        assert n > np.count_nonzero(s_optdmd.modes)
        n = np.count_nonzero(s_optdmd.modes)


def test_fit_stlsq():
    """
    Test that fitting with sequential thresholding produces accurate models.
    Test using various parameterizations of the stlsq method.
    """

    def mode_prox(Z):
        return hard_threshold(Z, gamma=0.001)

    # (1) Using default STLSQ parameters.
    s_optdmd = BOPDMD(
        svd_rank=2,
        mode_prox=mode_prox,
        varpro_opts_dict={"stlsq_opts": {}},
    )
    s_optdmd.fit(X, t)
    np.testing.assert_allclose(sort_imag(s_optdmd.eigs), true_eigs, rtol=0.01)
    assert relative_error(s_optdmd.reconstructed_data, X_clean) < 0.1

    # (2) Using altered tolerance value.
    s_optdmd = BOPDMD(
        svd_rank=2,
        mode_prox=mode_prox,
        varpro_opts_dict={"stlsq_opts": {"tol": 1e-12}},
    )
    s_optdmd.fit(X, t)
    np.testing.assert_allclose(sort_imag(s_optdmd.eigs), true_eigs, rtol=0.01)
    assert relative_error(s_optdmd.reconstructed_data, X_clean) < 0.1

    # (3) Using fixed number of iterations.
    s_optdmd = BOPDMD(
        svd_rank=2,
        mode_prox=mode_prox,
        varpro_opts_dict={"stlsq_opts": {"fixed_iter": 1, "tol": None}},
    )
    s_optdmd.fit(X, t)
    np.testing.assert_allclose(sort_imag(s_optdmd.eigs), true_eigs, rtol=0.01)
    assert relative_error(s_optdmd.reconstructed_data, X_clean) < 0.1


def test_use_proj_1():
    """
    Test that fitting without projection produces accurate models.
    """
    # (1) Test for the prox-gradient model.
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 1.0},
        use_proj=False,
    )
    s_optdmd.fit(X, t)
    np.testing.assert_allclose(sort_imag(s_optdmd.eigs), true_eigs, rtol=0.01)
    assert relative_error(s_optdmd.reconstructed_data, X_clean) < 0.1

    # (2) Test for the thresholding model.
    def mode_prox(Z):
        return hard_threshold(Z, gamma=0.001)

    s_optdmd = BOPDMD(svd_rank=2, mode_prox=mode_prox, use_proj=False)
    s_optdmd.fit(X, t)
    np.testing.assert_allclose(sort_imag(s_optdmd.eigs), true_eigs, rtol=0.01)
    assert relative_error(s_optdmd.reconstructed_data, X_clean) < 0.1


def test_use_proj_2():
    """
    Test that using data projection actually reduces fitting time.
    Test using the large toy data set.
    """
    # (1) Test for the prox-gradient model.
    t1 = time.time()
    s_optdmd = SparseBOPDMD(svd_rank=2, mode_regularizer="l0")
    s_optdmd.fit(X_big, t)
    t2 = time.time()
    s_optdmd = SparseBOPDMD(svd_rank=2, mode_regularizer="l0", use_proj=False)
    s_optdmd.fit(X_big, t)
    t3 = time.time()
    assert t2 - t1 < t3 - t2

    # (2) Test for the thresholding model.
    def mode_prox(Z):
        return hard_threshold(Z, gamma=0.001)

    t1 = time.time()
    s_optdmd = BOPDMD(svd_rank=2, mode_prox=mode_prox)
    s_optdmd.fit(X_big, t)
    t2 = time.time()
    s_optdmd = BOPDMD(svd_rank=2, mode_prox=mode_prox, use_proj=False)
    s_optdmd.fit(X_big, t)
    t3 = time.time()
    assert t2 - t1 < t3 - t2


def test_use_proj_3():
    """
    Test that models generated with data projection and models generated
    without approximately produce the same model for various parameters.
    Checks similarity of the modes and the eigenvalues. Uses prox.
    """
    for _lambda in [0.1, 0.2, 0.5, 1.0, 5.0]:
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
        assert relative_error(s_optdmd_proj.modes, s_optdmd_noproj.modes) < 0.01
        np.testing.assert_allclose(
            sort_imag(s_optdmd_proj.eigs),
            sort_imag(s_optdmd_noproj.eigs),
            rtol=0.01,
        )


def test_use_proj_4():
    """
    Test that models generated with data projection and models generated
    without approximately produce the same model for various parameters.
    Checks similarity of the modes and the eigenvalues. Uses thresh.
    """
    for _gamma in [1e-4, 2e-4, 5e-4, 1e-3, 5e-3]:

        def mode_prox(Z):
            return hard_threshold(Z, gamma=_gamma)

        # Fit model WITH data projection.
        s_optdmd_proj = BOPDMD(svd_rank=2, mode_prox=mode_prox)
        s_optdmd_proj.fit(X, t)

        # Fit model WITHOUT data projection.
        s_optdmd_noproj = BOPDMD(
            svd_rank=2, mode_prox=mode_prox, use_proj=False
        )
        s_optdmd_noproj.fit(X, t)

        # Compare modes and eigenvalues.
        assert relative_error(s_optdmd_proj.modes, s_optdmd_noproj.modes) < 0.01
        np.testing.assert_allclose(
            sort_imag(s_optdmd_proj.eigs),
            sort_imag(s_optdmd_noproj.eigs),
            rtol=0.01,
        )


def test_use_mask_1():
    """
    Test that the pixel mask expectedly masks the correct pixels.
    Test using the step function mode.
    """
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 1.0},
    )
    s_optdmd.fit(X2, t)
    assert np.sum(s_optdmd.mask) == 45.0
    np.testing.assert_array_equal(s_optdmd.mask[15:20], np.zeros(5))


def test_use_mask_2():
    """
    Test that fitting without pixel masking produces accurate models.
    """
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 1.0},
        use_mask=False,
    )
    s_optdmd.fit(X, t)
    np.testing.assert_allclose(sort_imag(s_optdmd.eigs), true_eigs, rtol=0.01)
    assert relative_error(s_optdmd.reconstructed_data, X_clean) < 0.1


def test_use_mask_3():
    """
    Test that using pixel masking actually reduces fitting time.
    Test using the large toy data set.
    """
    t1 = time.time()
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 2.0},
    )
    s_optdmd.fit(X_big, t)
    t2 = time.time()
    s_optdmd = SparseBOPDMD(
        svd_rank=2,
        mode_regularizer="l0",
        regularizer_params={"lambda": 2.0},
        use_mask=False,
    )
    s_optdmd.fit(X_big, t)
    t3 = time.time()
    assert t2 - t1 < t3 - t2


def test_use_mask_4():
    """
    Test that models generated with pixel masking and models generated
    without produce the same model for sufficiently high sparsity.
    Checks similarity of the modes and the eigenvalues.
    """
    for _lambda in [1.0, 5.0, 10.0, 20.0]:
        # Fit model WITH pixel masking.
        s_optdmd_mask = SparseBOPDMD(
            svd_rank=2,
            mode_regularizer="l0",
            regularizer_params={"lambda": _lambda},
        )
        s_optdmd_mask.fit(X, t)

        # Fit model WITHOUT pixel masking.
        s_optdmd_nomask = SparseBOPDMD(
            svd_rank=2,
            mode_regularizer="l0",
            regularizer_params={"lambda": _lambda},
            use_mask=False,
        )
        s_optdmd_nomask.fit(X, t)

        # Compare modes and eigenvalues.
        assert relative_error(s_optdmd_mask.modes, s_optdmd_nomask.modes) < 1e-6
        np.testing.assert_allclose(
            sort_imag(s_optdmd_mask.eigs),
            sort_imag(s_optdmd_nomask.eigs),
            rtol=1e-6,
        )
