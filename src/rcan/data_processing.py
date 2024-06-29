"""!
@file data_processing.py

@brief Contains tools used to pre-process image data.
"""

import numpy as np


def crop_volume(volume, num_steps, start, step, label):
    """!
    @brief Takes an image volume and divides part of it into smaller volumes
    by cropping lateral sections (the full z dimension is used).

    @param volume (np.ndarray) - image volume to crop
    @param num_steps (tuple[int]) - number of images in each lateral dimension
    (total number of subvolumes is the product)
    @param start (tuple[int]) - start coordinates for crop region
    @param step (tuple[int]) - lateral size of subvolume images
    @param label (str) - prefix for output file names

    @returns generator that yields image subvolumes
    """
    for dim in range(2):
        crop_boundaries = [
            0,
            start[dim],
            start[dim] + num_steps[dim] * step[dim],
            volume.shape[1 + dim],
        ]
        if crop_boundaries != sorted(crop_boundaries):
            raise ValueError(
                f"Cropping layout out of bounds in lateral dimension {dim}."
            )
    for i in range(num_steps[0]):
        for j in range(num_steps[1]):
            subvolume = volume[
                :,
                start[0] + step[0] * i : start[0] + step[0] * (i + 1),
                start[1] + step[1] * j : start[1] + step[1] * (j + 1),
            ]
            filename = f"{label}_{i*num_steps[1] + j:03}.tif"
            yield subvolume, filename


def conv_omx_to_czxy(original, n_phases, n_angles):
    """!
    @brief Converts image array from OMX (PZA format) to CZXY format.

    @param original (np.ndarray) - Image array in original format
    @param n_phases (int) - Number of phases
    @param n_angles (int) - Number of angles

    @returns np.ndarray Converted image array
    """
    if original.shape[0] % (n_phases * n_angles) != 0:
        raise ValueError(
            "number of acquisitions should divide channel dimension length."
        )
    if len(original.shape) != 3:
        raise ValueError("omx image should have shape length 3.")
    converted = np.zeros(
        (
            n_phases * n_angles,
            original.shape[0] // (n_phases * n_angles),
            *original.shape[1:],
        )
    )
    for z in range(original.shape[0] // (n_phases * n_angles)):
        for a in range(n_angles):
            converted[a * n_phases : (a + 1) * n_phases, z, ...] = original[
                (a * original.shape[0] // n_angles)
                + z * n_phases : (a * original.shape[0] // n_angles)
                + (z + 1) * n_phases,
                ...,
            ]
    return converted


def conv_czxy_to_omx(original, n_phases, n_angles):
    """!
    @brief Converts image array from CZXY to OMX format.

    @param original (np.ndarray) - Image array in original format
    @param n_phases (int) - Number of phases
    @param n_angles (int) - Number of angles

    @returns np.ndarray Converted image array
    """
    if original.shape[0] != n_phases * n_angles:
        raise ValueError("channel dimension should equal n_phases x n_angles.")
    if len(original.shape) != 4:
        raise ValueError("czxy image should have shape length 4.")
    converted = np.zeros(
        (
            original.shape[0] * original.shape[1],
            *original.shape[2:],
        )
    )
    for z in range(original.shape[1]):
        for a in range(n_angles):
            converted[
                (a * converted.shape[0] // n_angles)
                + z * n_phases : (a * converted.shape[0] // n_angles)
                + (z + 1) * n_phases,
                ...,
            ] = original[a * n_phases : (a + 1) * n_phases, z, ...]
    return converted


def conv_omx_to_paz(original, n_phases, n_angles):
    """!
    @brief Converts image array from OMX (PZA format) to PAZ format.

    @param original (np.ndarray) - Image array in original format
    @param n_phases (int) - Number of phases
    @param n_angles (int) - Number of angles

    @returns np.ndarray Converted image array
    """
    if original.shape[0] % (n_phases * n_angles) != 0:
        raise ValueError(
            "number of acquisitions should divide channel dimension length."
        )
    if len(original.shape) != 3:
        raise ValueError("omx image should have shape length 3.")
    converted = np.zeros_like(original)
    for z in range(original.shape[0] // (n_phases * n_angles)):
        for a in range(n_angles):
            converted[
                z * n_phases * n_angles
                + a * n_phases : z * n_phases * n_angles
                + (a + 1) * n_phases,
                ...,
            ] = original[
                (a * original.shape[0] // n_angles)
                + z * n_phases : (a * original.shape[0] // n_angles)
                + (z + 1) * n_phases,
                ...,
            ]
    return converted


def conv_paz_to_omx(original, n_phases, n_angles):
    """!
    @brief Converts image array from PAZ to OMX(PZA) format.

    @param original (np.ndarray) - Image array in original format
    @param n_phases (int) - Number of phases
    @param n_angles (int) - Number of angles

    @returns np.ndarray Converted image array
    """
    if original.shape[0] % (n_phases * n_angles) != 0:
        raise ValueError(
            "number of acquisitions should divide channel dimension length."
        )
    if len(original.shape) != 3:
        raise ValueError("omx image should have shape length 3.")
    converted = np.zeros_like(original)
    for z in range(original.shape[0] // (n_phases * n_angles)):
        for a in range(n_angles):
            converted[
                (a * original.shape[0] // n_angles)
                + z * n_phases : (a * original.shape[0] // n_angles)
                + (z + 1) * n_phases,
                ...,
            ] = original[
                z * n_phases * n_angles
                + a * n_phases : z * n_phases * n_angles
                + (a + 1) * n_phases,
                ...,
            ]
    return converted


class ImageStack:
    """!
    @brief Handles creation and loading of image hyperstacks in order to make
    reconstructions using ImageJ easier.
    """

    def __init__(self, dim, stack_number, stack_idx, sample, files, n_acq):
        """!
        @brief Initialises class.

        @param dim (int) - Dimension of images
        @param stack_number (int) - Number of images in the stack
        @param stack_idx (int) - The index of the stack within the set of
        stacks for the files list
        @param sample (np.ndarray) - Image from the directory which enables
        correct image stack shape/dtype and error catching
        @param files (list) - List of all files in directory
        @param n_acq (int) - Number of SIM acquisitions in the images
        """
        self.dim = dim
        self.n_acq = n_acq
        self.sample = sample
        # Accounts for remainder of files at the end that cannot make a
        # full stack
        stack_number_adjusted = len(
            files[stack_idx * stack_number : (stack_idx + 1) * stack_number]
        )
        if dim == 2:
            stack = np.zeros(
                (
                    stack_number_adjusted,
                    *sample.shape,
                )
            ).astype(sample.dtype)
            self.stack = stack

        elif dim == 3:
            self.n_z = sample.shape[0] // n_acq
            stack = np.zeros(
                (
                    stack_number_adjusted * self.n_z,
                    n_acq,
                    *sample.shape[1:],
                )
            ).astype(sample.dtype)
            self.stack = stack

        else:
            raise ValueError("dim must be 2 or 3.")

    def add_image(self, img_data, i):
        """!
        @brief Adds an image to the initialised stack

        @param img_data (np.ndarray) - Image to be added
        @param i (int) - Index of the image in the stack
        """
        if img_data.shape != self.sample.shape:
            raise ValueError("all images must be the same shape.")
        elif img_data.dtype != self.sample.dtype:
            raise ValueError("all images must have the same data type.")
        if self.dim == 2:
            self.stack[i, ...] = img_data
        else:
            for z in range(self.n_z):
                self.stack[i * self.n_z + z, ...] = img_data[
                    z * self.n_acq : (z + 1) * self.n_acq, ...
                ]

    def export_stack(self):
        """!
        @brief Returns the stack.

        @returns np.ndarray
        """
        return self.stack
