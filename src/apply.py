"""!@file apply.py
@brief Script producing restored images resulting from an RCAN denoiser being
applied to low SNR images.

@details This script takes directories of raw images, and a model checkpoint
file, and applies the model to the image in a patched fashion. The details of
this patching, and the output datatype, can be configured.

Arguments:
- m: model checkpoint filepath
- i: low SNR image directory path
- o: output directory path
- b: specifies pixel bit depth to save for output (8 or 16)
- O: block overlap shape (by default input_shape / 8)
- p_min: input normalization parameter, percentile maps to zero
- p_max: input normalization parameter, percentile maps to one
- normalize_output_range_between_zero_and_one: scaling for output

Adapted from https://github.com/AiviaCommunity/3D-RCAN/blob/TF2/apply.py

Copyright 2021 SVision Technologies LLC.
Copyright 2021-2022 Leica Microsystems, Inc.
Creative Commons Attribution-NonCommercial 4.0 International Public License
(CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/
"""

import argparse
import itertools
import numpy as np
import pathlib
import torch
import tifffile

from rcan.utils import (
    apply,
    normalize,
    load_rcan_checkpoint,
    tuple_of_ints,
    percentile,
    normalize_between_zero_and_one,
)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-b", "--bpp", type=int, choices=[8, 16], default=16)
parser.add_argument("-O", "--block_overlap_shape", type=tuple_of_ints)
parser.add_argument("--p_min", type=percentile, default=2.0)
parser.add_argument("--p_max", type=percentile, default=99.9)
parser.add_argument(
    "--normalize_output_range_between_zero_and_one", action="store_true"
)
args = parser.parse_args()

input_path = pathlib.Path(args.input)
output_path = pathlib.Path(args.output)

if input_path.is_dir() and not output_path.exists():
    print("Creating output directory", output_path)
    output_path.mkdir(parents=True)

if input_path.is_dir() != output_path.is_dir():
    raise ValueError("Mismatch between input and output path types")

if input_path.is_dir():
    raw_files = sorted(input_path.glob("*.tif"))
    data = itertools.zip_longest(raw_files, [])
else:
    data = [(input_path, None)]

# Initialise GPU(/CPU)
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print("Initialised processing device:", device)

# Load model
ckpt, model = load_rcan_checkpoint(pathlib.Path(args.model), device)
RCAN_hyperparameters = ckpt["hyperparameters"]

if args.block_overlap_shape is None:
    overlap_shape = [
        max(1, x // 8) if x > 2 else 0
        for x in RCAN_hyperparameters["input_shape"]
    ]
else:
    overlap_shape = args.block_overlap_shape

for raw_file, gt_file in data:
    print("Loading raw image from", raw_file)
    raw = normalize(tifffile.imread(raw_file), args.p_min, args.p_max)

    print("Applying model")
    restored = apply(
        model,
        raw,
        RCAN_hyperparameters["input_shape"],
        RCAN_hyperparameters["input_shape"],
        RCAN_hyperparameters["num_input_channels"],
        RCAN_hyperparameters["num_output_channels"],
        batch_size=1,
        device=device,
        overlap_shape=overlap_shape,
        verbose=True,
    )

    if args.normalize_output_range_between_zero_and_one:
        restored = normalize_between_zero_and_one(restored)

    if args.bpp == 8:
        restored = np.clip(255 * restored, 0, 255).astype("uint8")
    elif args.bpp == 16:
        restored = np.clip(65535 * restored, 0, 65535).astype("uint16")

    if output_path.is_dir():
        output_file = output_path / ("pred_" + raw_file.name)
    else:
        output_file = output_path

    print("Saving output image to", output_file)
    tifffile.imwrite(str(output_file), restored, imagej=True)
