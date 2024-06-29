import torch
import pytest

from src.rcan.model import _conv, _global_average_pooling, RCAN


# Test convolution layers
def test_conv_valid():
    ConvLayer = _conv(
        ndim=2,
        in_filters=3,
        out_filters=5,
        kernel_size=3,
    )
    input_tensor = torch.arange(
        3 * 10 * 10,
        dtype=torch.float32,
    ).reshape((3, 10, 10))
    assert ConvLayer(input_tensor).shape == (5, 10, 10)


def test_conv_invalid():
    with pytest.raises(NotImplementedError) as exc_info:
        _conv(
            ndim=4,
            in_filters=3,
            out_filters=5,
            kernel_size=3,
        )
    assert str(exc_info.value) == "4D convolution is not supported"


# Test global average pooling layers
def test_pooling_valid():
    PoolLayer = _global_average_pooling(ndim=2)
    input_tensor = torch.arange(18, dtype=torch.float32).reshape((2, 3, 3))
    assert PoolLayer(input_tensor)[0] == 4 and PoolLayer(input_tensor)[1] == 13


def test_pooling_invalid():
    with pytest.raises(NotImplementedError) as exc_info:
        _global_average_pooling(ndim=4)
    assert str(exc_info.value) == "4D global average pooling is not supported"


# Test RCAN function produces correct output shape
def test_RCAN():
    RCANModule = RCAN(
        input_shape=(32, 32),
        num_input_channels=4,
        num_hidden_channels=16,
        num_residual_blocks=3,
        num_residual_groups=3,
        channel_reduction=8,
        residual_scaling=2.0,
        num_output_channels=5,
    )
    input_tensor = torch.arange(4 * 32 * 32, dtype=torch.float32).reshape(
        (4, 32, 32)
    )
    output_tensor = RCANModule(input_tensor)
    assert output_tensor.shape == (5, 32, 32)
