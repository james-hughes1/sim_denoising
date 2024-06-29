"""!@file convert_omx_to_paz.py
@brief Script enabling .tif file conversion between OMX and PAZ.

@details This script takes directories of image volumes as input, and converts,
in place, between the OMX and PAZ formats (in either direction). In the OMX
format, the first dimension is of size n_phases x n_z x n_angles; moving along
this dimension, the phase changes first, then the z-value, then the angle. The
PAZ format is the same except the order is changed so that z-values and angels
are swapped.

Arguments:
- i: image directory
- p: number of phases
- a: number of angles
- b: specifies conversion - if not used it will be OMX to PAZ, the b flag
reverses this direction.
"""

import argparse
import pathlib
import tifffile

from rcan.data_processing import conv_omx_to_paz, conv_paz_to_omx

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-p", "--num_phases", type=int, required=True)
parser.add_argument("-a", "--num_angles", type=int, required=True)
parser.add_argument("-b", "--backwards", action="store_true")
args = parser.parse_args()


input_dir = pathlib.Path(args.input)
input_files = sorted(input_dir.rglob("*.tif"))


if not args.backwards:
    for input_file in input_files:
        original = tifffile.imread(input_file)
        converted = conv_omx_to_paz(original, args.num_phases, args.num_angles)
        print("Saving output image to", input_file)
        tifffile.imwrite(str(input_file), converted.astype("uint16"))

else:
    for input_file in input_files:
        original = tifffile.imread(input_file)
        converted = conv_paz_to_omx(original, args.num_phases, args.num_angles)
        print("Saving output image to", input_file)
        tifffile.imwrite(
            str(input_file), converted.astype("uint16"), imagej=True
        )
