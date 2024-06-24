import pytest
import numpy as np
import argparse

from src.rcan.utils import (
    normalize,
    tuple_of_ints,
    percentile,
    reshape_to_bcwh,
    normalize_between_zero_and_one,
)

TEST_ARRAY = np.arange(101)
rng = np.random.default_rng(seed=24062024)
TEST_ARRAY = rng.permutation(TEST_ARRAY)


def test_normalize():
    array_normalized = normalize(TEST_ARRAY, p_min=20, p_max=85)
    assert np.linalg.norm(array_normalized - ((TEST_ARRAY - 20) / 65)) < 1e-3


def test_tuple_of_ints_valid():
    assert tuple_of_ints("3, -5, 9") == (3, -5, 9)


def test_tuple_of_ints_single():
    assert tuple_of_ints("37") == (37,)


def test_tuple_of_ints_invalid():
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        tuple_of_ints("3.14, 4, -8")
    assert str(exc_info.value) == "3.14, 4, -8 not a tuple of integers"


def test_percentile_valid():
    assert percentile("  73.4    ") == 73.4


def test_percentile_invalid():
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        percentile("101.1")
    assert str(exc_info.value) == "101.1 not in range [0.0, 100.0]"


IMAGE_2D = np.arange(30 * 30).reshape((30, 30))
IMAGE_INVALID = np.arange(5**5).reshape((5, 5, 5, 5, 5))


def test_reshape_to_bcwh_valid():
    image_bcwh = reshape_to_bcwh(IMAGE_2D)
    assert image_bcwh.shape == (1, 1, 30, 30)
    assert (image_bcwh[0, 0] == IMAGE_2D).all()


def test_reshape_to_bcwh_invalid():
    with pytest.raises(ValueError) as exc_info:
        reshape_to_bcwh(IMAGE_INVALID)
    assert str(exc_info.value) == "data must be an array with 2, 3, or"
    " 4 dimensions."


def test_normalize_between_zero_and_one_valid():
    array_normalized = normalize_between_zero_and_one(TEST_ARRAY)
    assert np.linalg.norm(array_normalized - TEST_ARRAY / 100) < 1e-3


def test_normalize_between_zero_and_one_constant():
    array_normalized = normalize_between_zero_and_one(np.ones((30, 40)))
    assert (array_normalized == 0).all()
