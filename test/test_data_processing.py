import numpy as np
import pytest

from src.rcan.data_processing import (
    crop_volume,
    conv_omx_to_czxy,
    conv_czxy_to_omx,
    conv_omx_to_paz,
    conv_paz_to_omx,
    ImageStack,
)

VOLUME = np.arange(3 * 100 * 80).reshape((3, 100, 80))


# Test cropped volumes have the correct shape and values
def test_crop_volume_valid():
    crop_volume_generator = crop_volume(
        VOLUME, num_steps=(3, 3), start=(70, 10), step=(5, 10), label="volume"
    )
    for i, (array, filename) in enumerate(crop_volume_generator):
        assert array.shape == (3, 5, 10)
        if i == 8:
            assert (VOLUME[:, 80:85, 30:40] == array).all()
            assert filename == "volume_008.tif"


# Cropping regime incompatible with image volume size
def test_crop_volume_invalid():
    with pytest.raises(ValueError) as exc_info:
        crop_volume_generator = crop_volume(
            VOLUME,
            num_steps=(3, 3),
            start=(70, 10),
            step=(5, 500),
            label="volume",
        )
        for array, filename in crop_volume_generator:
            pass
    assert (
        str(exc_info.value)
        == "Cropping layout out of bounds in lateral dimension 1."
    )


IMAGE_CZXY = np.arange(15 * 3 * 2 * 2).reshape((15, 3, 2, 2))
IMAGE_OMX = np.arange(45 * 2 * 2).reshape((45, 2, 2))
IMAGE_PAZ = np.arange(45 * 2 * 2).reshape((45, 2, 2))


# Test that in both images, 4th phase, 1st angle, 2nd z-plane image match up
def test_conv_omx_to_czxy():
    image_czxy = conv_omx_to_czxy(IMAGE_OMX, 5, 3)
    assert (image_czxy[3, 1] == IMAGE_OMX[8]).all()


def test_conv_czxy_to_omx():
    image_omx = conv_czxy_to_omx(IMAGE_CZXY, 5, 3)
    assert (image_omx[8] == IMAGE_CZXY[3, 1]).all()


def test_conv_omx_to_paz():
    image_paz = conv_omx_to_paz(IMAGE_OMX, 5, 3)
    assert (image_paz[18] == IMAGE_OMX[8]).all()


def test_conv_paz_to_omx():
    image_omx = conv_paz_to_omx(IMAGE_PAZ, 5, 3)
    assert (image_omx[8] == IMAGE_PAZ[18]).all()


# Simulate 2 3D SIM stacks, 3x2x2 size.
IMAGES = [np.arange(15 * 3 * 2 * 2).reshape((45, 2, 2)) for _ in range(2)]


# Check that stack has the correct shape and that image data is added in the
# expected order
def test_image_stack_valid():
    # Test the remainder image stack size is computed correctly.
    image_stack = ImageStack(3, 3, 2, IMAGES[0], range(8), 15)
    image_stack.add_image(IMAGES[1], 1)
    stack = image_stack.export_stack()
    assert stack.shape == (6, 15, 2, 2)
    assert (stack[:3] == 0).all()
    assert (stack[4, ...] == IMAGES[1][15:30, ...]).all()


# Test adding an image of incorrect size to the stack
def test_image_stack_invalid_shape():
    image_stack = ImageStack(3, 3, 2, IMAGES[0], range(8), 15)
    with pytest.raises(ValueError) as exc_info:
        image_stack.add_image(np.ones((1, 1)), 1)
    assert str(exc_info.value) == "all images must be the same shape."


# Test adding an image of incorrect dtype to the stack
def test_image_stack_invalid_type():
    image_stack = ImageStack(3, 3, 2, IMAGES[0], range(8), 15)
    with pytest.raises(ValueError) as exc_info:
        image_stack.add_image(np.ones((45, 2, 2), dtype="uint8"), 1)
    assert str(exc_info.value) == "all images must have the same data type."
