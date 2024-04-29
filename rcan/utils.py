# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import numpy as np
import fractions
import itertools
import tqdm
import torch
import tifffile
import argparse

from rcan.model import RCAN


def normalize(image, p_min=2, p_max=99.9, dtype="float32"):
    """
    Normalizes the image intensity so that the `p_min`-th and the `p_max`-th
    percentiles are converted to 0 and 1 respectively.

    References
    ----------
    Content-Aware Image Restoration: Pushing the Limits of Fluorescence
    Microscopy
    https://doi.org/10.1038/s41592-018-0216-7
    """
    low, high = np.percentile(image, (p_min, p_max))
    return ((image - low) / (high - low + 1e-6)).astype(dtype)


def rescale(restored, gt):
    """Affine rescaling to minimize the MSE to the GT"""
    cov = np.cov(restored.flatten(), gt.flatten())
    a = cov[0, 1] / cov[0, 0]
    b = gt.mean() - a * restored.mean()
    return a * restored + b


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
    """
    Applies a model to an input image. The input image stack is split into
    sub-blocks with model's input size, then the model is applied block by
    block.

    Parameters
    ----------
    model: torch.nn.module
        PyTorch model.
    data: array_like or list of array_like
        Input data. Either an image or a list of images.
    batch_size: int
        Controls the batch size used to process image data.
    device: torch.device
        PyTorch device object to specify processor to use.
    overlap_shape: tuple of int or None
        Overlap size between sub-blocks in each dimension. If not specified,
        a default size ((32, 32) for 2D and (2, 32, 32) for 3D) is used.
        Results at overlapped areas are blended together linearly.

    Returns
    -------
    ndarray
        Result image.
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


def save_imagej_hyperstack(filename, image):
    assert image.ndim in [3, 4]
    if image.ndim == 4:
        image = np.transpose(image, (1, 0, 2, 3))

    tifffile.imwrite(str(filename), image, imagej=True)


def save_ome_tiff(filename, image):
    assert image.ndim in [3, 4]
    image = np.expand_dims(image, (1, 2) if image.ndim == 3 else 1)
    c, t, z, y, x = image.shape

    pixel_type = {
        np.dtype("uint8"): "Uint8",
        np.dtype("uint16"): "Uint16",
        np.dtype("float32"): "Float",
    }[image.dtype]

    channel_names = ["Raw", "Restored", "Ground Truth"]
    lsid_base = "urn:lsid:ome.xsd:"

    channel_info = ""
    for i, name in enumerate(channel_names[:c]):
        channel_info += f"""\
    <ChannelInfo Name="{name}" ID="{lsid_base}ChannelInfo:{i + 3}">
      <ChannelComponent Index="{i}" Pixels="{lsid_base}Pixels:2"/>
    </ChannelInfo>
"""
    description = f"""\
<OME xmlns="http://www.openmicroscopy.org/XMLschemas/OME/FC/ome.xsd">
  <Image Name="Unnamed [{pixel_type} {x}x{y}x{z}x{t} Channels]"
         ID="{lsid_base}Image:1">
{channel_info}\
    <Pixels DimensionOrder="XYZTC" PixelType="{pixel_type}"
            SizeX="{x}" SizeY="{y}" SizeZ="{z}" SizeT="{t}" SizeC="{c}"
            BigEndian="false" ID="{lsid_base}Pixels:2">
      <TiffData IFD="0" NumPlanes="{z * c * t}"/>
    </Pixels>
  </Image>
</OME>
"""

    tifffile.imwrite(
        filename, data=image, description=description, metadata=None
    )


def save_tiff(filename, image, format):
    {"imagej": save_imagej_hyperstack, "ome": save_ome_tiff}[format](
        filename, image
    )


def load_rcan_checkpoint(ckpt_path, device):
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
    return tuple(int(s) for s in string.split(","))


def percentile(x):
    x = float(x)
    if 0.0 <= x <= 100.0:
        return x
    else:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 100.0]")
