import numpy as np

from src.rcan.data_processing import (
    crop_volume,
    conv_omx_to_czxy,
    conv_czxy_to_omx,
    conv_omx_to_paz,
    conv_paz_to_omx,
)

VOLUME = np.arange(3 * 100 * 80).reshape((3, 100, 80))


def test_crop_volume():
    crop_volume_generator = crop_volume(
        VOLUME, num_steps=(3, 3), start=(70, 10), step=(5, 10), label="volume"
    )
    for i, (array, filename) in enumerate(crop_volume_generator):
        assert array.shape == (3, 5, 10)
        if i == 8:
            assert (VOLUME[:, 80:85, 30:40] == array).all()
            assert filename == "volume_008.tif"


IMAGE_CZXY = np.arange(15 * 3 * 2 * 2).reshape((15, 3, 2, 2))
IMAGE_OMX = np.arange(45 * 2 * 2).reshape((45, 2, 2))
IMAGE_PAZ = np.arange(45 * 2 * 2).reshape((45, 2, 2))


def test_conv_omx_to_czxy():
    image_czxy = conv_omx_to_czxy(IMAGE_OMX, 5, 3)
    # 4th phase, 1st angle, 2nd z-plane
    assert (image_czxy[3, 1] == IMAGE_OMX[8]).all()


def test_conv_czxy_to_omx():
    image_omx = conv_czxy_to_omx(IMAGE_CZXY, 5, 3)
    # 4th phase, 1st angle, 2nd z-plane
    assert (image_omx[8] == IMAGE_CZXY[3, 1]).all()


def test_conv_omx_to_paz():
    image_paz = conv_omx_to_paz(IMAGE_OMX, 5, 3)
    # 4th phase, 1st angle, 2nd z-plane
    assert (image_paz[18] == IMAGE_OMX[8]).all()


def test_conv_paz_to_omx():
    image_omx = conv_paz_to_omx(IMAGE_PAZ, 5, 3)
    # 4th phase, 1st angle, 2nd z-plane
    assert (image_omx[8] == IMAGE_PAZ[18]).all()
