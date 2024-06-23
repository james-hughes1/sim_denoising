"""!
@file model.py
@brief Module defining the RCAN model architecture.

@details Module that defines a number of classes inheriting from nn.Module,
implementing different levels of the RCAN architecture. This includes the
channel attention layer, residual channel attention block, and RCAN itself.

Migrated from https://github.com/AiviaCommunity/3D-RCAN/blob/TF2/rcan/model.py

Copyright 2021 SVision Technologies LLC.
Copyright 2021-2022 Leica Microsystems, Inc.
Creative Commons Attribution-NonCommercial 4.0 International Public License
(CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/
"""

import torch


def _conv(
    ndim, in_filters, out_filters, kernel_size, padding="same", **kwargs
):
    """!
    @brief Returns the appropriate torch.nn convolution layer based on
    parameters.

    @param ndim (int) - Specifies a 1, 2, or 3 dimensional convolution kernel
    @param in_filters (int) - Number of hidden input channels
    @param out_filters (int) - Number of hidden output channels
    @param kernel_size (int or tuple) Size of convolution kernel
    @param padding (str, optional) - Border padding strategy. Default: "same"

    @returns torch.nn.Module object of the specified type
    """
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
    """!
    @brief Returns the appropriate torch.nn pooling layer based on
    parameters.

    @param ndim (int) - Specifies a 2 or 3 dimensional convolution kernel

    @returns torch.nn.Module object of the specified type
    """
    if ndim == 2:
        return torch.nn.AdaptiveAvgPool2d(1)
    elif ndim == 3:
        return torch.nn.AdaptiveAvgPool3d(1)
    else:
        raise NotImplementedError(
            f"{ndim}D global average pooling is not supported"
        )


class _channel_attention_block(torch.nn.Module):
    """!
    @brief Implements channel attention block/layer.

    @details Instantiates a simple attention mechanism which pools all spatial
    information in each channel, and computes channel attention weights
    through a series of linear transformations and activation layers.
    Builds part of the architecture originally presented in [1].
    Software implementation based on [2].

    References
    ----------
    [1] Image Super-Resolution Using Very Deep Residual Channel Attention
        Networks
        https://arxiv.org/abs/1807.02758
    [2] Fast, multicolour optical sectioning over extended fields of view by
        combining interferometric SIM with machine learning
        https://doi.org/10.1364/BOE.510912
        (Implementation based on CALayer from the paper's source code:
        https://github.com/edward-n-ward/ML-OS-SIM/blob/master/RCAN/Training%20code/models.py)
    """

    def __init__(self, ndim, num_channels, reduction=16):
        """!
        @brief Initialises class.

        @param ndim (int) - Feature dimensionality
        @param num_channels (int) - Number of hidden channels
        @param reduction (int, optional) - Factor to reduce the number of
        channels by during the attention weight computation. Default: 16.
        """
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
        """!
        @brief Forward method for class.

        @param x (torch.Tensor) Input
        @returns torch.Tensor representing x multiplied by attention weights
        across channels.
        """
        y = self.global_average_pooling(x)
        y = self.conv_1(y)
        y = self.conv_2(y)

        return x * y


class _residual_channel_attention_blocks(torch.nn.Module):
    """!
    @brief Implements residual group based on [1].

    References
    ----------
    [1] Fast, multicolour optical sectioning over extended fields of view by
        combining interferometric SIM with machine learning
        https://doi.org/10.1364/BOE.510912
        (Implementation based on ResidualGroup from the paper's source code:
        https://github.com/edward-n-ward/ML-OS-SIM/blob/master/RCAN/Training%20code/models.py)
    """
    def __init__(
        self,
        ndim,
        num_channels,
        repeat=1,
        channel_reduction=8,
        residual_scaling=1.0,
    ):
        """!
        @brief Initialises object.

        @param ndim (int) - Spatial dimension of input features
        @param num_channels (int) - Number of hidden channels
        @param repeat (int) - Number of residual blocks in group
        @param channel_reduction (int) - Channel reduction during attention
        mechanism
        @param residual_scaling (float) - output multiplier before residual
        connection
        """
        super(_residual_channel_attention_blocks, self).__init__()
        self.repeat = repeat
        self.residual_scaling = residual_scaling
        self.conv_list = torch.nn.ModuleList([torch.nn.Sequential(
            _conv(ndim, num_channels, num_channels, 3),
            torch.nn.ReLU(inplace=True),
            _conv(ndim, num_channels, num_channels, 3),
        ) for i in range(self.repeat)])
        self.channel_attention_block_list = torch.nn.ModuleList(
            [_channel_attention_block(
                ndim, num_channels, channel_reduction
            ) for i in range(self.repeat)]
        )

    def forward(self, x):
        """!
        @brief Forward method for class.

        @param x (torch.Tensor) - Input values
        @returns torch.Tensor representing output values
        """
        for i in range(self.repeat):
            skip = x

            x = self.conv_list[i](x)
            x = self.channel_attention_block_list[i](x)

            if self.residual_scaling != 1.0:
                x *= self.residual_scaling

        return x + skip


def _standardize(x):
    """!
    @brief Standardises input data.

    @details Standardize the signal so that the range becomes [-1, 1]
    (assuming the original range is [0, 1]).
    @param x (torch.Tensor) Input
    @returns torch.Tensor representing standardised output
    """
    return 2 * x - 1


def _destandardize(x):
    """!
    @brief Inverse of _standardize

    @param x (torch.Tensor) Input
    @returns torch.Tensor representing destandardised output.
    """
    return 0.5 * x + 0.5


class RCAN(torch.nn.Module):
    """!
    @brief Builds a residual channel attention network. Note that the upscale
    module at the end of the network is omitted so that the input and output
    of the model have the same size.

    References
    ----------
    [1] Image Super-Resolution Using Very Deep Residual Channel Attention
        Networks
        https://arxiv.org/abs/1807.02758
    [2] Fast, multicolour optical sectioning over extended fields of view by
        combining interferometric SIM with machine learning
        https://doi.org/10.1364/BOE.510912
        (Implementation based on RCAN from the paper's source code:
        https://github.com/edward-n-ward/ML-OS-SIM/blob/master/RCAN/Training%20code/models.py)
    """

    def __init__(
        self,
        input_shape=(16, 256, 256),
        *,
        num_input_channels=9,
        num_hidden_channels=32,
        num_residual_blocks=3,
        num_residual_groups=5,
        channel_reduction=8,
        residual_scaling=1.0,
        num_output_channels=-1,
    ):
        """!
        @brief Initialises object.

        @details Builds a residual channel attention network. Note that the
        upscale module at the end of the network is omitted so that the input
        and output of the model have the same size.

        @param input_shape (tuple[int]) - Input shape of the model.
        @param num_channels (int) - Number of feature channels.
        @param num_residual_blocks (int) - Number of residual channel
        attention blocks in each residual group.
        @param num_residual_groups (int) - Number of residual groups.
        @param channel_reduction (int) - Channel reduction ratio for channel
        attention.
        @param residual_scaling (float) - Scaling factor applied to the
        residual component in the residual channel attention block.
        @param num_output_channels (int) - Number of channels in the output
        image. if negative, it is set to the same number as the input.

        @returns torch.nn.Module PyTorch model instance.
        """
        super(RCAN, self).__init__()
        ndim = len(input_shape)
        if num_output_channels < 0:
            num_output_channels = num_input_channels

        self.num_residual_groups = num_residual_groups
        self.rcab_list = torch.nn.ModuleList([
            _residual_channel_attention_blocks(
                ndim,
                num_hidden_channels,
                num_residual_blocks,
                channel_reduction,
                residual_scaling,
            ) for i in range(self.num_residual_groups)])
        self.conv_input = _conv(
            ndim, num_input_channels, num_hidden_channels, 3
        )
        self.conv_list = torch.nn.ModuleList([
            _conv(
                ndim, num_hidden_channels, num_hidden_channels, 3
            ) for i in range(self.num_residual_groups + 1)
        ])
        self.conv_output = _conv(
            ndim, num_hidden_channels, num_output_channels, 3
        )

    def forward(self, x):
        """!
        @brief Forward method for class.

        @param x (torch.Tensor) - Input
        @returns torch.Tensor Output
        """
        # The format here of x should be B,C,(Z),X,Y
        x = _standardize(x)
        x = self.conv_input(x)

        long_skip = x

        for i in range(self.num_residual_groups):
            short_skip = x

            x = self.rcab_list[i](x)

            if self.num_residual_groups == 1:
                break

            x = self.conv_list[i](x)
            x += short_skip

        x = self.conv_list[-1](x)
        x += long_skip

        x = self.conv_output(x)
        x = _destandardize(x)

        return x
