import numpy as np

from src.calculator import Calculator


def test_add():
    calc = Calculator()
    assert calc.add(1, 2) == 3
    assert calc.add(-1, 1) == 0
    assert calc.add(0, 0) == 0
    assert calc.add(1.5, 2.5) == 4.0
    assert calc.add(-1.5, -2.5) == -4.0
    assert calc.add(1000000, 2000000) == 3000000


def test_add_vectors():
    calc = Calculator()
    assert np.array_equal(calc.add([1, 2], [3, 4]), np.array([4, 6]))
    assert np.array_equal(calc.add([-1, -2], [1, 2]), np.array([0, 0]))
    assert np.array_equal(calc.add([0, 0], [0, 0]), np.array([0, 0]))
    assert np.array_equal(calc.add([1.5, 2.5], [3.5, 4.5]), np.array([5.0, 7.0]))
    assert np.array_equal(calc.add([-1.5, -2.5], [-3.5, -4.5]), np.array([-5.0, -7.0]))


def test_add_numpy_matrices():
    calc = Calculator()
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    expected_result = np.array([[6, 8], [10, 12]])
    assert np.array_equal(calc.add(a, b), expected_result)

    a = np.array([[-1, -2], [-3, -4]])
    b = np.array([[1, 2], [3, 4]])
    expected_result = np.array([[0, 0], [0, 0]])
    assert np.array_equal(calc.add(a, b), expected_result)
