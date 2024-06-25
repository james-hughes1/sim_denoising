import pytest

from src.rcan.data_generator import SIM_Dataset

IMAGES = [
    {
        "raw": "test/testimages/Raw/488_561_microtubule_000_1_noisy.tif",
        "gt": "test/testimages/Raw/488_561_microtubule_000_1_noisy.tif",
    },
    {
        "raw": "test/testimages/Raw/488_561_microtubule_002_1_noisy.tif",
        "gt": "test/testimages/Raw/488_561_microtubule_002_1_noisy.tif",
    },
    {
        "raw": "test/testimages/Raw/488_561_microtubule_006_1_noisy.tif",
        "gt": "test/testimages/Raw/488_561_microtubule_006_1_noisy.tif",
    },
]


def test_sim_dataset():
    dataset = SIM_Dataset(
        IMAGES,
        shape=(8, 8),
        transform_function="rotate_and_flip",
        steps_per_epoch=10,
    )
    assert len(dataset) == 30
    # Patch from 2nd image (7 mod 3 == 1)
    patch_raw, patch_gt = dataset.__getitem__(7)
    assert patch_raw.shape == (9, 8, 8) and patch_gt.shape == (9, 8, 8)


def test_sim_dataset_non_square_patch():
    with pytest.raises(ValueError) as exc_info:
        SIM_Dataset(
            IMAGES,
            shape=(4, 8),
            transform_function="rotate_and_flip",
            steps_per_epoch=10,
        )
    assert (
        str(exc_info.value) == "Patch shape must be square when using"
        " `rotate_and_flip`; Received shape: (4, 8)"
    )


def test_sim_dataset_large_patch():
    with pytest.raises(ValueError) as exc_info:
        SIM_Dataset(
            IMAGES,
            shape=(1024, 1024),
            transform_function="rotate_and_flip",
            steps_per_epoch=10,
        )
    assert str(exc_info.value) == "Source image must be larger than patch"
