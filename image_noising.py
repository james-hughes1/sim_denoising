import argparse
import numpy as np
import pathlib
import tifffile
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument(
    "-d", "--dimension", type=int, choices=[2, 3], required=True
)
parser.add_argument("-s", "--scale_factor", type=float, default=10.0)
parser.add_argument("-t", "--train_fraction", type=float, default=0.75)
parser.add_argument("-c", "--channels", type=int, default=1)
args = parser.parse_args()

input_path = pathlib.Path(args.input)
output_path = pathlib.Path(args.output)

if args.scale_factor <= 1.0:
    raise ValueError("Scale factor must exceed 1.0")

if args.train_fraction < 0.0 or args.train_fraction > 1.0:
    raise ValueError("Train fraction must be within interval [0,1].")

if not output_path.exists():
    print("Creating output directory", output_path)
    output_path.mkdir(parents=True)


def save_image_pair(gt_img, train, name, img_idx):
    noised_img = np.uint16(rng.poisson(gt_img / args.scale_factor))
    if train:
        tifffile.imwrite(
            f"{output_train_gt_path}/{name}_{img_idx}_gt.tif",
            gt_img,
            imagej=True,
        )
        tifffile.imwrite(
            f"{output_train_raw_path}/{name}_{img_idx}_noisy.tif",
            noised_img,
            imagej=True,
        )
    else:
        tifffile.imwrite(
            f"{output_val_gt_path}/{name}_{img_idx}_gt.tif",
            gt_img,
            imagej=True,
        )
        tifffile.imwrite(
            f"{output_val_raw_path}/{name}_{img_idx}_noisy.tif",
            noised_img,
            imagej=True,
        )


for channel_idx in range(args.channels):
    output_train_gt_path = output_path.joinpath(
        f"Channel_{channel_idx}", "Training", "GT"
    )
    output_train_raw_path = output_path.joinpath(
        f"Channel_{channel_idx}", "Training", "Raw"
    )
    output_val_gt_path = output_path.joinpath(
        f"Channel_{channel_idx}", "Validation", "GT"
    )
    output_val_raw_path = output_path.joinpath(
        f"Channel_{channel_idx}", "Validation", "Raw"
    )
    for path in [
        output_train_gt_path,
        output_train_raw_path,
        output_val_gt_path,
        output_val_raw_path,
    ]:
        if not path.exists():
            print("Creating GT directory", path)
            path.mkdir(parents=True)

    if not output_path.is_dir():
        raise ValueError("Output path should be a directory")

    if input_path.is_dir():
        data = sorted(input_path.glob("*.tif"))
    else:
        data = [input_path]

    rng = np.random.default_rng(seed=25042024)

    n_img = len(data)
    n_acquisitions = tifffile.imread(data[0]).shape[0] // args.channels

    img_idx_all = list(product(range(n_img), range(n_acquisitions)))
    rng.shuffle(img_idx_all)
    train_size = int(args.train_fraction * len(img_idx_all))
    img_idx_train = img_idx_all[:train_size]
    img_idx_test = img_idx_all[train_size:]

    for img_idx, img_file in enumerate(data):
        gt = tifffile.imread(img_file)
        if len(gt.shape) != args.dimension + 1:
            raise ValueError(
                "Mismatch between specified dimensions and true image"
                " dimensions"
            )
        for acq_idx in range(n_acquisitions):
            save_image_pair(
                gt[n_acquisitions * channel_idx + acq_idx, ...],
                ((img_idx, acq_idx) in img_idx_train),
                img_file.with_suffix("").name,
                acq_idx,
            )
