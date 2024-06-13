"""!@file generate_sim.py
@brief Script simulating the acquisition of 3D SIM image volumes.

@details Takes a directory of 3D image volumes as input, and produces
synthetic 3-beam SIM volumes of size (15, 32, 256, 256).

Arguments:
- i: directory path of input volumes
- o: directory path of output volumes
- s: start index of input files to process
- e: end index of input files to process
- z: z_offset, used to specify the region of the input volume to use.
"""

import argparse
import numpy as np
import pathlib
import tifffile

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument(
    "-d", "--dimension", type=int, choices=[2, 3], required=True
)
# Here image data format is (Channels x Num_acq.), Z, X, Y
parser.add_argument("-c", "--channels", type=int, required=True)
parser.add_argument("-s", "--scale_factor", type=float, default=10.0)
# Test size (relative to full dataset)
parser.add_argument("-tf", "--test_fraction", type=float, default=0.20)
# Validation size (relative to size of training dataset)
parser.add_argument("-vf", "--val_fraction", type=float, default=0.20)
args = parser.parse_args()

input_path = pathlib.Path(args.input)
output_path = pathlib.Path(args.output)

if args.scale_factor <= 1.0:
    raise ValueError("Scale factor must exceed 1.0")

if args.test_fraction < 0.0 or args.test_fraction > 1.0:
    raise ValueError("Test fraction must be within interval [0,1].")

if args.val_fraction < 0.0 or args.val_fraction > 1.0:
    raise ValueError("Validation fraction must be within interval [0,1].")

if not output_path.exists():
    print("Creating output directory", output_path)
    output_path.mkdir(parents=True)

if not output_path.is_dir():
    raise ValueError("Output path should be a directory")


# Create output directory structure
output_train_gt_path = output_path.joinpath("Training", "GT")
output_train_raw_path = output_path.joinpath("Training", "Raw")
output_val_gt_path = output_path.joinpath("Validation", "GT")
output_val_raw_path = output_path.joinpath("Validation", "Raw")
output_test_gt_path = output_path.joinpath("Testing", "GT")
output_test_raw_path = output_path.joinpath("Testing", "Raw")
for path in [
    output_train_gt_path,
    output_train_raw_path,
    output_val_gt_path,
    output_val_raw_path,
    output_test_gt_path,
    output_test_raw_path,
]:
    if not path.exists():
        print("Creating directory", path)
        path.mkdir(parents=True)

if input_path.is_dir():
    data = sorted(input_path.glob("*.tif"))
else:
    data = [input_path]


def save_image_pair(gt_img, split, name, channel_idx):
    noised_img = np.uint16(rng.poisson(gt_img / args.scale_factor))
    output_gt_path, output_raw_path = {
        "train": (output_train_gt_path, output_train_raw_path),
        "val": (output_val_gt_path, output_val_raw_path),
        "test": (output_test_gt_path, output_test_raw_path),
    }[split]
    tifffile.imwrite(
        f"{output_gt_path}/{name}_{channel_idx}_gt.tif",
        gt_img,
        imagej=True,
    )
    tifffile.imwrite(
        f"{output_raw_path}/{name}_{channel_idx}_noisy.tif",
        noised_img,
        imagej=True,
    )


n_acquisitions = tifffile.imread(data[0]).shape[0] // args.channels
n_img = len(data)
train_size = int((1 - args.test_fraction) * n_img)
val_size = int(args.val_fraction * train_size)
rng = np.random.default_rng(seed=25042024)

for channel_idx in range(args.channels):
    # Set indices of train, test, validation image pairs.
    img_idx_all = list(range(n_img))
    rng.shuffle(img_idx_all)
    img_idx_test = img_idx_all[train_size:]
    img_idx_train = img_idx_all[: train_size - val_size]
    img_idx_val = img_idx_all[train_size - val_size : train_size]

    for img_idx, img_file in enumerate(data):
        gt = tifffile.imread(img_file)
        if len(gt.shape) != args.dimension + 1:
            raise ValueError(
                "Mismatch between specified dimensions and true image"
                " dimensions"
            )
        if img_idx in img_idx_train:
            split = "train"
        elif img_idx in img_idx_val:
            split = "val"
        else:
            split = "test"
        save_image_pair(
            gt[
                n_acquisitions
                * channel_idx : n_acquisitions
                * (channel_idx + 1),
                ...,
            ],
            split,
            img_file.with_suffix("").name,
            channel_idx,
        )
