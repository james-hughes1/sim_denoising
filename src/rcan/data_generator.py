# Copyright 2021 SVision Technologies LLC.
# Copyright 2021-2022 Leica Microsystems, Inc.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import numpy as np
import warnings
import tifffile

from .utils import normalize

import torch
from torch.utils.data import Dataset, DataLoader


class SIM_Dataset(Dataset):
    def __init__(
        self,
        images,
        shape,
        transform_function="rotate_and_flip",
        intensity_threshold=0.0,
        area_ratio_threshold=0.0,
        scale_factor=1,
        steps_per_epoch=1,
        p_min=2.0,
        p_max=99.9,
    ):
        # Note that image dimensions are (Z),X,Y,(C);
        # Patch shape is 'shape', which is (Z),X,Y
        # 'scale_factor' enables the target image, y, to be scaled-down
        # steps_per_epoch controls how many times each image is seen.

        # Default data augmentation
        def rotate_and_flip(x, y, dim):
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            k = np.random.randint(0, 4)
            x = None if x is None else torch.rot90(x, k=k, dims=[-2, -1])
            y = None if y is None else torch.rot90(y, k=k, dims=[-2, -1])
            if np.random.random() < 0.5:
                x = None if x is None else torch.flip(x, dims=[-1])
                y = None if y is None else torch.flip(y, dims=[-1])
            if np.random.random() < 0.5:
                x = None if x is None else torch.flip(x, dims=[-2])
                y = None if y is None else torch.flip(y, dims=[-2])
            if np.random.random() < 0.5 and dim == 3:
                x = None if x is None else torch.flip(x, dims=[0])
                y = None if y is None else torch.flip(y, dims=[0])
            return x, y

        # Set up dataset attributes with checks.
        self._shape = tuple(shape)
        dim = len(self._shape)
        self.steps_per_epoch = steps_per_epoch
        self.p_min = p_min
        self.p_max = p_max

        if transform_function == "rotate_and_flip":
            if shape[-2] != shape[-1]:
                raise ValueError(
                    "Patch shape must be square when using `rotate_and_flip`; "
                    f"Received shape: {shape}"
                )
            self._transform_function = lambda x, y: rotate_and_flip(x, y, dim)
        elif callable(transform_function):
            self._transform_function = transform_function
        elif transform_function is None:
            self._transform_function = lambda x, y: (x, y)
        else:
            raise ValueError("Invalid transform function")

        self._intensity_threshold = intensity_threshold

        if not 0 <= area_ratio_threshold <= 1:
            raise ValueError('"area_ratio_threshold" must be between 0 and 1')
        self._area_threshold = area_ratio_threshold * np.prod(shape)
        if isinstance(scale_factor, int):
            self._scale_factor = (scale_factor,) * (dim + 1)
        else:
            self._scale_factor = tuple(scale_factor)
        if any(not isinstance(f, int) or f == 0 for f in self._scale_factor):
            raise ValueError('"scale_factor" must be nonzero integer')

        # Check image files.
        x = [p["raw"] for p in images]
        y = [p["gt"] for p in images]

        for (
            s,
            f,
        ) in zip(shape, self._scale_factor):
            if f < 0 and s % -f != 0:
                raise ValueError(
                    "When downsampling, all elements in `shape` must be "
                    "divisible by the scale factor; "
                    f"Received shape: {shape}, "
                    f"scale factor: {self._scale_factor}"
                )

        # Store image file paths
        self._x, self._y = [
            list(m) if isinstance(m, (list, tuple)) else [m] for m in [x, y]
        ]

        if self._y is not None and len(self._x) != len(self._y):
            raise ValueError(
                "Different number of images are given: "
                f"{len(self._x)} vs. {len(self._y)}"
            )

        x_image_0 = tifffile.imread(self._x[0])
        y_image_0 = tifffile.imread(self._y[0])

        # Note image format C, Z, X, Y
        if len(x_image_0.shape) == len(shape):
            x_image_0 = x_image_0[np.newaxis, ...]

        if y_image_0 is not None:
            if len(y_image_0.shape) == len(shape):
                y_image_0 = y_image_0[np.newaxis, ...]

        for j in range(len(self._x)):
            x_image_j = tifffile.imread(self._x[j])
            y_image_j = tifffile.imread(self._y[j])

            if x_image_j.dtype != x_image_0.dtype:
                raise ValueError("All source images must be the same type")

            if self._y is not None and y_image_j.dtype != y_image_0.dtype:
                raise ValueError("All target images must be the same type")

            if len(x_image_j.shape) == len(shape):
                x_image_j = x_image_j[np.newaxis, ...]

            if len(x_image_j.shape) != len(shape) + 1:
                raise ValueError(f"Source image must be {len(shape)}D")

            if np.any(x_image_j.shape[1:] < tuple(shape)):
                raise ValueError("Source image must be larger than patch")

            if y_image_j is not None:
                if len(y_image_j.shape) == len(shape):
                    y_image_j = y_image_j[np.newaxis, ...]

                if len(y_image_j.shape) != len(shape) + 1:
                    raise ValueError(f"Target image must be {len(shape)}D")

                expected_y_image_size = self._scale(x_image_j.shape[1:])
                if y_image_j.shape[1:] != expected_y_image_size:
                    raise ValueError(
                        "Invalid target image size: "
                        f"expected {expected_y_image_size}, "
                        f"but received {y_image_j.shape[1:]}"
                    )

            if x_image_j.shape[0] != x_image_0.shape[0]:
                raise ValueError(
                    "All source images must have same number of channels"
                )

            if self._y is not None:
                if y_image_j.shape[0] != y_image_0.shape[0]:
                    raise ValueError(
                        "All target images must have same number of channels"
                    )

        # Define output signature, typically
        # ((input_shape, input_channels), (out_shape, out_channels))
        output_shape_x = (x_image_0.shape[0], *shape)

        if self._y is None:
            self.output_shape = (output_shape_x,)
        else:
            self.output_signature = (
                output_shape_x,
                (y_image_0.shape[0], *self._scale(shape)),
            )

    def _scale(self, shape):
        return tuple(
            s * f if f > 0 else s // -f
            for s, f in zip(shape, self._scale_factor)
        )

    def __getitem__(self, j):
        for _ in range(512):
            # Normalize pixel values between (approximately [0,1])
            x_image_j = normalize(
                tifffile.imread(self._x[j % len(self._x)]),
                p_min=self.p_min,
                p_max=self.p_max,
            )
            y_image_j = normalize(
                tifffile.imread(self._y[j % len(self._x)]),
                p_min=self.p_min,
                p_max=self.p_max,
            )

            if len(x_image_j.shape) == len(self._shape):
                x_image_j = x_image_j[np.newaxis, ...]

            if len(y_image_j.shape) == len(self._shape):
                y_image_j = y_image_j[np.newaxis, ...]

            # Specify random patch location
            tl = [
                np.random.randint(0, a - b + 1)
                for a, b in zip(x_image_j.shape, self.output_signature[0])
            ]

            patch_x_roi = tuple(
                slice(a, a + b) for a, b in zip(tl, self.output_signature[0])
            )
            patch_x = np.copy(x_image_j[patch_x_roi])

            if y_image_j is not None:
                patch_y_roi = tuple(
                    slice(a, a + b)
                    for a, b in zip(self._scale(tl), self.output_signature[1])
                )
                patch_y = np.copy(y_image_j[patch_y_roi])

            # Check that patch has sufficient intensity threshold.
            if self._intensity_threshold > 0:
                foreground_area = np.count_nonzero(
                    (patch_x if self._y is None else patch_y)
                    > self._intensity_threshold
                )
                if foreground_area < self._area_threshold:
                    continue

            break

        else:
            warnings.warn(
                "Failed to sample a valid patch",
                RuntimeWarning,
                stacklevel=3,
            )

        if self._y is None:
            return self._transform_function(patch_x, None)[0]
        else:
            return self._transform_function(patch_x, patch_y)

    def __len__(self):
        return len(self._x) * self.steps_per_epoch


def load_SIM_dataset(
    images,
    shape,
    batch_size,
    transform_function,
    intensity_threshold,
    area_threshold,
    scale_factor,
    steps_per_epoch,
    p_min,
    p_max,
):
    """
    Generates batches of images with real-time data augmentation.

    Parameters
    ----------
    shape: tuple of int
        Shape of batch images (excluding the channel dimension).
    batch_size: int
        Batch size.
    transform_function: str or callable or None
        Function used for data augmentation. Typically you will set
        ``transform_function='rotate_and_flip'`` to apply combination of
        randomly selected image rotation and flipping.  Alternatively, you can
        specify an arbitrary transformation function which takes two input
        images (source and target) and returns transformed images. If
        ``transform_function=None``, no augmentation will be performed.
    intensity_threshold: float
        If ``intensity_threshold > 0``, pixels whose intensities are greater
        than this threshold will be considered as foreground.
    area_ratio_threshold: float between 0 and 1
        If ``intensity_threshold > 0``, the generator calculates the ratio of
        foreground pixels in a target patch, and rejects the patch if the ratio
        is smaller than this threshold.
    scale_factor: int != 0
        Scale factor for the target patch size. Positive and negative values
        mean up- and down-scaling respectively.
    """
    dataset = SIM_Dataset(
        images,
        shape,
        transform_function,
        intensity_threshold,
        area_threshold,
        scale_factor,
        steps_per_epoch,
        p_min=p_min,
        p_max=p_max,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)