"""!
@file utils.py

@brief Contains utility functions for the training loop and inference.

Migrated from https://github.com/AiviaCommunity/3D-RCAN/blob/TF2/rcan/utils.py

Copyright 2021 SVision Technologies LLC.
Creative Commons Attribution-NonCommercial 4.0 International Public License
(CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/
"""

import numpy as np
import fractions
import itertools
import tqdm
import torch
import argparse

from .model import RCAN


def normalize(image, p_min=2, p_max=99.9, dtype="float32"):
    """!
    @brief Normalizes the image intensity so that the `p_min`-th and the
    `p_max`-th percentiles are converted to 0 and 1 respectively.

    @param image (np.ndarray) - Image to apply the normalization to
    @param p_min (float, optional) - Percentile that is mapped to zero.
    Default: 2
    @param p_max (float, optional) - Percentile that is mapped to one. Default:
    99.9
    @param dtype (str) - Datatype to use for the output

    @returns np.ndarray Image with transformed pixel values

    References
    ----------
    Content-Aware Image Restoration: Pushing the Limits of Fluorescence
    Microscopy
    https://doi.org/10.1038/s41592-018-0216-7
    """
    low, high = np.percentile(image, (p_min, p_max))
    return ((image - low) / (high - low + 1e-6)).astype(dtype)


def apply(
    model,
    data,
    model_input_image_shape,
    model_output_image_shape,
    num_input_channels,
    num_output_channels,
    batch_size,
    device,
    overlap_shape=None,
    verbose=False,
):
    """!
    @brief Applies a model to an input image.

    @details The input image stack is split into sub-blocks with model's input
    size, then the model is applied block by block.

    @param model (torch.nn.module) - PyTorch model
    @param data (array_like or list of array_like) - Input data. Either an
    image or a list of images
    @param batch_size (int) - Controls the batch size used to process image
    data
    @param device (torch.device) - PyTorch device object to specify processor
    to use
    @param overlap_shape (tuple of int or None) - Overlap size between
    sub-blocks in each dimension. If not specified, a default size ((32, 32)
    for 2D and (2, 32, 32) for 3D) is used. Results at overlapped areas are
    blended together linearly

    @returns np.ndarray Result image
    """
    model.eval()

    if len(model_input_image_shape) != len(model_output_image_shape):
        raise NotImplementedError

    image_dim = len(model_input_image_shape)

    scale_factor = tuple(
        fractions.Fraction(o, i)
        for i, o in zip(model_input_image_shape, model_output_image_shape)
    )

    def _scale_tuple(t):
        t = [v * f for v, f in zip(t, scale_factor)]

        if not all([v.denominator == 1 for v in t]):
            raise NotImplementedError

        return tuple(v.numerator for v in t)

    def _scale_roi(roi):
        roi = [
            slice(r.start * f, r.stop * f) for r, f in zip(roi, scale_factor)
        ]

        if not all(
            [r.start.denominator == 1 and r.stop.denominator == 1 for r in roi]
        ):
            raise NotImplementedError

        return tuple(slice(r.start.numerator, r.stop.numerator) for r in roi)

    # Set default overlap shape and check
    if overlap_shape is None:
        if image_dim == 2:
            overlap_shape = (32, 32)
        elif image_dim == 3:
            overlap_shape = (2, 32, 32)
        else:
            raise NotImplementedError
    elif len(overlap_shape) != image_dim:
        raise ValueError(
            f"Overlap shape must be {image_dim}D; "
            f"Received shape: {overlap_shape}"
        )

    step_shape = tuple(
        m - o for m, o in zip(model_input_image_shape, overlap_shape)
    )

    block_weight = np.ones(
        [
            m - 2 * o
            for m, o in zip(
                model_output_image_shape, _scale_tuple(overlap_shape)
            )
        ],
        dtype=np.float32,
    )

    block_weight = np.pad(
        block_weight,
        [(o + 1, o + 1) for o in _scale_tuple(overlap_shape)],
        "linear_ramp",
    )[(slice(1, -1),) * image_dim]

    if isinstance(data, (list, tuple)):
        input_is_list = True
    else:
        data = [data]
        input_is_list = False

    result = []

    for image in data:
        # Add the channel dimension if necessary
        if image.ndim == image_dim:
            image = image[np.newaxis, ...]

        if image.ndim != image_dim + 1 or image.shape[0] != num_input_channels:
            raise ValueError(
                f"Input image must be {image_dim}D with "
                f"{num_input_channels} channels; "
                f"Received image shape: {image.shape}"
            )

        input_image_shape = image.shape[1:]
        output_image_shape = _scale_tuple(input_image_shape)

        applied = np.zeros(
            (num_output_channels, *output_image_shape), dtype=np.float32
        )
        sum_weight = np.zeros(output_image_shape, dtype=np.float32)

        num_steps = tuple(
            (i + s - 1) // s for i, s in zip(input_image_shape, step_shape)
        )

        # top-left corner of each block
        blocks = list(
            itertools.product(
                *[np.arange(n) * s for n, s in zip(num_steps, step_shape)]
            )
        )

        for chunk_index in tqdm.trange(
            0,
            len(blocks),
            batch_size,
            disable=not verbose,
            dynamic_ncols=True,
            ascii=tqdm.utils.IS_WIN,
        ):
            rois = []
            batch = np.zeros(
                (batch_size, num_input_channels, *model_input_image_shape),
                dtype=np.float32,
            )
            for batch_index, tl in enumerate(
                blocks[chunk_index : chunk_index + batch_size]
            ):
                br = [
                    min(t + m, i)
                    for t, m, i in zip(
                        tl, model_input_image_shape, input_image_shape
                    )
                ]
                r1, r2 = zip(
                    *[(slice(s, e), slice(0, e - s)) for s, e in zip(tl, br)]
                )

                m = image[:, *r1]
                if model_input_image_shape != m.shape[1:]:
                    pad_width = [(0, 0)]
                    pad_width += [
                        (0, b - s)
                        for b, s in zip(model_input_image_shape, m.shape[1:])
                    ]
                    m = np.pad(m, pad_width, "reflect")

                batch[batch_index] = m
                rois.append((r1, r2))

            batch = torch.from_numpy(batch).to(device)
            model = model.to(device)
            p = model(batch)
            p = p.detach().cpu().numpy()

            for batch_index in range(len(rois)):
                for channel in range(num_output_channels):
                    p[batch_index, channel, ...] *= block_weight

                r1, r2 = [_scale_roi(roi) for roi in rois[batch_index]]
                applied[:, *r1] += p[batch_index][:, *r2]
                sum_weight[r1] += block_weight[r2]

        for channel in range(num_output_channels):
            applied[channel, ...] /= sum_weight

        if applied.shape[0] == 1:
            applied = applied[0, ...]

        result.append(applied)

    return result if input_is_list else result[0]


def load_rcan_checkpoint(ckpt_path, device):
    """!
    @brief Enables loading of RCAN checkpointed model.

    @details Uses the ``hyperparameters`` key saved in checkpoint file in
    order to avoid the need to know the architecture specifications in advance.

    @param ckpt_path (str) - filepath for checkpoint, should end in .pth
    @param device (torch.device) - handles processing unit for torch

    @return tuple of checkpoint, and model with weights loaded
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    RCAN_hyperparameters = ckpt["hyperparameters"]
    model = RCAN(
        RCAN_hyperparameters["input_shape"],
        num_input_channels=RCAN_hyperparameters["num_input_channels"],
        num_hidden_channels=RCAN_hyperparameters["num_hidden_channels"],
        num_residual_blocks=RCAN_hyperparameters["num_residual_blocks"],
        num_residual_groups=RCAN_hyperparameters["num_residual_groups"],
        channel_reduction=RCAN_hyperparameters["channel_reduction"],
        residual_scaling=RCAN_hyperparameters["residual_scaling"],
        num_output_channels=RCAN_hyperparameters["num_output_channels"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return ckpt, model


def tuple_of_ints(string):
    """!
    @brief Defines behaviour of parsing tuples of ints (argparse).
    """
    return tuple(int(s) for s in string.split(","))


def percentile(x):
    """!
    @brief Defines behaviour of parsing percentiles (argparse).
    """
    x = float(x)
    if 0.0 <= x <= 100.0:
        return x
    else:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 100.0]")


def reshape_to_bcwh(data):
    """!
    @brief Reshapes 2D or 3D array to have batch x channel x width x height
    format, by prepending extra dimensions.

    @param data (np.ndarray) - array to be reshaped
    @returns np.ndarray transformed data
    """
    if len(data.shape) == 2:
        return data.reshape((1, 1, *data.shape))
    elif len(data.shape) == 3:
        return data.reshape((1, *data.shape))
    else:
        return data


def normalize_between_zero_and_one(data):
    """!
    @brief Coerce pixel values to [0, 1] range.

    @param data (np.ndarray or torch.Tensor) - image array to transform
    @returns np.ndarray or torch.Tensor transformed image array
    """
    max_val, min_val = data.max(), data.min()
    diff = max_val - min_val
    return (data - min_val) / diff if diff > 0 else np.zeros_like(data)
