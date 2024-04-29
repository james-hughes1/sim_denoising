import numpy as np
import torch
from ignite.metrics import PSNR
import argparse
import pathlib
import tifffile
import itertools

from rcan.utils import normalize, apply, rescale
from rcan.plotting import plot_learning_curve, plot_predictions
from rcan.utils import load_rcan_checkpoint, tuple_of_ints, percentile


# Parse configuration file

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-g", "--ground_truth", type=str, required=True)
parser.add_argument("-b", "--bpp", type=int, choices=[8, 16, 32], default=32)
parser.add_argument("-B", "--block_shape", type=tuple_of_ints)
parser.add_argument("-O", "--block_overlap_shape", type=tuple_of_ints)
parser.add_argument("--p_min", type=percentile, default=2.0)
parser.add_argument("--p_max", type=percentile, default=99.9)
parser.add_argument("--rescale", action="store_true")
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

if args.ground_truth is None:
    gt_path = None
else:
    gt_path = pathlib.Path(args.ground_truth)
    if input_path.is_dir() != gt_path.is_dir():
        raise ValueError("Mismatch between input and ground truth path types")

if input_path.is_dir():
    raw_files = sorted(input_path.glob("*.tif"))

    if gt_path is None:
        data = itertools.zip_longest(raw_files, [])
    else:
        gt_files = sorted(gt_path.glob("*.tif"))

        if len(raw_files) != len(gt_files):
            raise ValueError(
                "Mismatch between raw and ground truth file counts "
                f"({len(raw_files)} vs. {len(gt_files)})"
            )

        data = zip(raw_files, gt_files)
else:
    data = [(input_path, gt_path)]

# Initialise GPU(/CPU)
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print("Initialised processing device:", device)

# Load model
ckpt, model = load_rcan_checkpoint(pathlib.Path(args.model), device)
RCAN_hyperparameters = ckpt["hyperparameters"]

output_dir = pathlib.Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

# Plot learning curve
plot_learning_curve(
    ckpt["losses_train"],
    ckpt["losses_val"],
    ckpt["psnr_train"],
    ckpt["psnr_val"],
    (7, 7),
    str(output_dir / "learning_curve.png"),
)

# Predictions
if args.block_overlap_shape is None:
    overlap_shape = [
        max(1, x // 8) if x > 2 else 0
        for x in RCAN_hyperparameters["input_shape"]
    ]
else:
    overlap_shape = args.block_overlap_shape

raw_imgs = []
restored_imgs = []
gt_imgs = []

raw_psnr = []
restored_psnr = []

psnr = PSNR(data_range=1.0, device=device)

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

    if gt_file is not None:
        print("Loading ground truth image from", gt_file)
        gt = tifffile.imread(str(gt_file))
        if raw.shape == gt.shape:
            gt = normalize(gt, args.p_min, args.p_max)
            if args.rescale:
                restored = rescale(restored, gt)
        else:
            print("Ground truth image discarded due to image shape mismatch")

    if args.normalize_output_range_between_zero_and_one:

        def normalize_between_zero_and_one(m):
            max_val, min_val = m.max(), m.min()
            diff = max_val - min_val
            return (m - min_val) / diff if diff > 0 else np.zeros_like(m)

        restored = normalize_between_zero_and_one(restored)

    if args.bpp == 8:
        restored = np.clip(255 * restored, 0, 255).astype("uint8")
    elif args.bpp == 16:
        restored = np.clip(65535 * restored, 0, 65535).astype("uint16")

    psnr.reset()
    psnr.update((torch.from_numpy(raw), torch.from_numpy(gt)))
    raw_psnr.append(psnr.compute())
    psnr.reset()
    psnr.update((torch.from_numpy(restored), torch.from_numpy(gt)))
    restored_psnr.append(psnr.compute())

    raw_imgs.append(raw)
    restored_imgs.append(restored)
    gt_imgs.append(gt)

for i in range(len(raw_psnr)):
    print(
        f"Image {i}: raw psnr={raw_psnr[i]:0.6f}"
        f" restored psnr={restored_psnr[i]:0.6f}"
    )

plot_predictions(
    6,
    raw_imgs,
    gt_imgs,
    restored_imgs,
    device,
    str(output_dir / "slice_predictions.png"),
)
