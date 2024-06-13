"""!@file recon_postprocess.py
@brief Script handling the postprocessing of SIM reconstructions.

@details Takes a directory of images as input, clips zero values,
and scales to the full 16-bit depth range. Operates in-place.

Arguments:
- i: directory path of input images
"""

import tifffile
import numpy as np
import argparse
import pathlib

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, required=True)
args = parser.parse_args()

files = sorted(list(pathlib.Path(args.input_dir).rglob("*.tif")))

for input_file in files:
    print("\nProcessing", input_file.name)
    img_data = tifffile.imread(input_file)

    # Clip zeros and scale to full uint16 range.
    img_data = img_data.astype("float64")
    img_data /= np.max(img_data)
    img_data = np.clip(img_data, 0.0, 1.0)
    img_data = (img_data * 65535).astype("uint16")
    tifffile.imwrite(str(input_file), img_data)
