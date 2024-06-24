"""!@file convert_slices_to_volumes.py
@brief Script enabling construction of 3D image volumes from large RGB 2D image
slices.

@details Takes a directory of 2D image slices as input, and converts to 3D
volumes. The 2D images are assumed to be ordered z-axially; the number of
images is the number of voxels in the z-direction of the 3D volumes. The
lateral cross-sections of the 3D images are determined by script arguments.
Saves in uint16 depth.

Arguments:
- i: directory path for 2D images
- o: directory path for 3D image volumes
- s: start pixel coordinates (x, y)
- j: crop size for image volume (crop_x, crop_y)
- n: number of crops to take in each direction (steps_x, steps_y)
- l: filename prefix, default "volume"
"""

import argparse
import pathlib
import tifffile
import numpy as np
from rcan.utils import tuple_of_ints
from rcan.data_processing import crop_volume

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-s", "--start", type=tuple_of_ints, required=True)
parser.add_argument("-j", "--step", type=tuple_of_ints, required=True)
parser.add_argument("-n", "--num_steps", type=tuple_of_ints, required=True)
parser.add_argument("-l", "--label", type=str, default="volume")
args = parser.parse_args()


input_dir = pathlib.Path(args.input)
output_dir = pathlib.Path(args.output)

input_files = sorted(input_dir.glob("*.tif"))
output_dir.mkdir(parents=True, exist_ok=True)

volume = np.zeros((len(input_files), 3061, 4096), dtype=np.uint8)
for i, file in enumerate(input_files):
    print(i)
    input_slice = tifffile.imread(file)
    volume[i] = np.uint8(np.mean(input_slice, axis=-1))

for subvolume, filename in crop_volume(
    volume, args.num_steps, args.start, args.step, args.label
):
    output_file = output_dir / filename
    print("Saving output image to", output_file)
    tifffile.imwrite(str(output_file), subvolume, imagej=True)
