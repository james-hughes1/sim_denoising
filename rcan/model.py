# Copyright 2021 SVision Technologies LLC.
# Copyright 2021-2022 Leica Microsystems, Inc.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import torch


def _get_spatial_ndim(x):
    return x.ndim - 2


def _get_num_channels(x):
    return x.shape[-1]


def _conv(
    ndim, in_filters, out_filters, kernel_size, padding="same", **kwargs
):
    if ndim == 1:
        conv_func = torch.nn.Conv1d
    elif ndim == 2:
        conv_func = torch.nn.Conv2d
    elif ndim == 3:
        conv_func = torch.nn.Conv3d
    else:
        raise NotImplementedError(f"{ndim}D convolution is not supported")

    return conv_func(
        in_filters, out_filters, kernel_size, padding=padding, **kwargs
    )


def _global_average_pooling(ndim):
    if ndim == 2:
        return torch.nn.AdaptiveAvgPool2d(1)
    elif ndim == 3:
        return torch.nn.AdaptiveAvgPool3d(1)
    else:
        raise NotImplementedError(
            f"{ndim}D global average pooling is not supported"
        )


class _channel_attention_block(torch.nn.Module):
    """
    Channel attention block.

    References
    ----------
    - Squeeze-and-Excitation Networks
      https://arxiv.org/abs/1709.01507
    - Image Super-Resolution Using Very Deep Residual Channel Attention
      Networks
      https://arxiv.org/abs/1807.02758
    - Fast, multicolour optical sectioning over extended fields of view by
      combining interferometric SIM with machine learning
      https://doi.org/10.1364/BOE.510912
      Implements the CALayer from the paper's source code:
      https://github.com/edward-n-ward/ML-OS-SIM/blob/master/RCAN/Training%20code/models.py
    """

    def __init__(self, ndim, num_channels, reduction=16):
        super(_channel_attention_block, self).__init__()
        self.global_average_pooling = _global_average_pooling(ndim)
        self.conv_1 = torch.nn.Sequential(
            _conv(ndim, num_channels, num_channels // reduction, 1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_2 = torch.nn.Sequential(
            _conv(ndim, num_channels // reduction, num_channels, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.global_average_pooling(x)
        y = self.conv_1(y)
        y = self.conv_2(y)

        return x * y


class _residual_channel_attention_blocks(torch.nn.Module):
    def __init__(
        self,
        ndim,
        num_channels,
        repeat=1,
        channel_reduction=8,
        residual_scaling=1.0,
    ):
        super(_residual_channel_attention_blocks, self).__init__()
        self.repeat = repeat
        self.residual_scaling = residual_scaling
        self.conv = torch.nn.Sequential(
            _conv(ndim, num_channels, num_channels, 3),
            torch.nn.ReLU(inplace=True),
            _conv(ndim, num_channels, num_channels, 3),
        )
        self.channel_attention_block = _channel_attention_block(
            ndim, num_channels, channel_reduction
        )

    def forward(self, x):
        for _ in range(self.repeat):
            skip = x

            x = self.conv(x)
            x = self.channel_attention_block(x)

            if self.residual_scaling != 1.0:
                x *= self.residual_scaling

        return x + skip


def _standardize(x):
    """
    Standardize the signal so that the range becomes [-1, 1] (assuming the
    original range is [0, 1]).
    """
    return 2 * x - 1


def _destandardize(x):
    """Undo standardization"""
    return 0.5 * x + 0.5


class RCAN(torch.nn.Module):
    """
    Builds a residual channel attention network. Note that the upscale module
    at the end of the network is omitted so that the input and output of the
    model have the same size.

    Parameters
    ----------
    input_shape: tuple of int
        Input shape of the model.
    num_channels: int
        Number of feature channels.
    num_residual_blocks: int
        Number of residual channel attention blocks in each residual group.
    num_residual_groups: int
        Number of residual groups.
    channel_reduction: int
        Channel reduction ratio for channel attention.
    residual_scaling: float
        Scaling factor applied to the residual component in the residual
        channel attention block.
    num_output_channels: int
        Number of channels in the output image. if negative, it is set to the
        same number as the input.

    Returns
    -------
    torch.nn.Module
        PyTorch model instance.

    References
    ----------
    Image Super-Resolution Using Very Deep Residual Channel Attention Networks
    https://arxiv.org/abs/1807.02758
    """

    def __init__(
        self,
        input_shape=(16, 256, 256, 1),
        *,
        num_channels=32,
        num_residual_blocks=3,
        num_residual_groups=5,
        channel_reduction=8,
        residual_scaling=1.0,
        num_output_channels=-1,
    ):
        super(RCAN, self).__init__()
        ndim = len(input_shape) - 1
        if num_output_channels < 0:
            num_output_channels = input_shape[-1]

        self.num_residual_groups = num_residual_groups
        self.rcab = _residual_channel_attention_blocks(
            ndim,
            num_channels,
            num_residual_blocks,
            channel_reduction,
            residual_scaling,
        )

        # Reshape from B,(Z),X,Y,C -> B,C,(Z),X,Y for the input, and revert
        # to the original for the output.
        if ndim == 2:
            self.reshape_1 = lambda x: torch.permute(x, (0, 3, 1, 2))
            self.reshape_2 = lambda x: torch.permute(x, (0, 2, 3, 1))
        else:
            self.reshape_1 = lambda x: torch.permute(x, (0, 4, 1, 2, 3))
            self.reshape_2 = lambda x: torch.permute(x, (0, 2, 3, 4, 1))
        self.conv_input = _conv(ndim, 1, num_channels, 3)
        self.conv = _conv(ndim, num_channels, num_channels, 3)
        self.conv_output = _conv(ndim, num_channels, num_output_channels, 3)

    def forward(self, x):
        x = _standardize(x)
        x = self.reshape_1(x)
        x = self.conv_input(x)

        long_skip = x

        for _ in range(self.num_residual_groups):
            short_skip = x

            x = self.rcab(x)

            if self.num_residual_groups == 1:
                break

            x = self.conv(x)
            x += short_skip

        x = self.conv(x)
        x += long_skip

        x = self.conv_output(x)
        x = self.reshape_2(x)
        x = _destandardize(x)

        return x
