"""!
@file data_processing.py

@brief Contains tools used to pre-process image data.
"""

import numpy as np


def crop_volume(volume, num_steps, start, step, label):
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
    assert original.shape[0] == n_phases * n_angles
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
    assert len(original.shape) == 3
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
    assert len(original.shape) == 3
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
