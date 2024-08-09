import numpy as np
from pydmd.sbopdmd import SparseBOPDMD

# Build the test data set:

# Dummy array for testing thresholding functions:
A = np.array([[-100, -1, 0, 1, 100], [-100j, -1j, 0, 1j, 100j]])


def test_l0():
    """
    See that "l0" preset uses the expected regularizer and proximal operator.
    """
    s_optdmd = SparseBOPDMD(
        mode_regularizer="l0",
        regularizer_params={"lambda": 5.0},
    )
    # Test the output of the mode regularizer function.
    assert s_optdmd.mode_regularizer(A) == 5.0 * np.count_nonzero(A)

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
    assert s_optdmd.mode_regularizer(A) == 5.0 * np.sum(np.abs(A))

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
    a = 100 * (1 - (5.0 / np.linalg.norm([100, 100], 2)))
    np.testing.assert_array_equal(
        s_optdmd.mode_prox(A, 1.0),
        a * np.array([[-1, 0, 0, 0, 1], [-1j, 0, 0, 0, 1j]]),
    )


def test_l02():
    """
    See that "l02" preset uses the expected regularizer and proximal operator.
    """
    raise NotImplementedError()


def test_l12():
    """
    See that "l12" preset uses the expected regularizer and proximal operator.
    """
    raise NotImplementedError()
