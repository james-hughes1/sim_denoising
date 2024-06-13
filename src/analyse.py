"""!@file analyse.py
@brief Script producing plots and small datasets that summarise the
performance of models.

@details This script reads directories of reconstructed images, and compares
raw versus model reconstructions versus ground truth. The script then produces
summary statistics, saves relevant metrics to a .csv file, and produces
samples of cropped image regions for comparison.

Arguments:
- g: directory path for ground-truth images
- r: directory path for raw images
- a: directory path for model-1-restored images
- b: directory path for model-2-restored images
- o: output directory for analysis plots, default "figures/"
- x: filepath for model 1 checkpoint (plots learning curve)
- y: filepath for model 2 checkpoint (plots learning curve)
- s: globbing string, to analyse a subset of images
- n: number of sample crops to display, default 0.
"""

import numpy as np
import torch
from ignite.metrics import PSNR, SSIM
import argparse
import pathlib
import tifffile
import pandas as pd

from rcan.plotting import plot_learning_curve, plot_reconstructions
from rcan.utils import load_rcan_checkpoint


def reshape_to_bcwh(data):
    if len(data.shape) == 2:
        return data.reshape((1, 1, *data.shape))
    elif len(data.shape) == 3:
        return data.reshape((1, *data.shape))
    else:
        return data


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gt_dir", type=str, required=True)
parser.add_argument("-r", "--raw_dir", type=str, required=True)
parser.add_argument("-a", "--model_1_dir", type=str, required=True)
parser.add_argument("-b", "--model_2_dir", type=str, default=None)
parser.add_argument("-o", "--output_dir", type=str, default="figures")
parser.add_argument("-x", "--model_1_ckpt", type=str, default=None)
parser.add_argument("-y", "--model_2_ckpt", type=str, default=None)
parser.add_argument("-s", "--glob_str", type=str, default="*.tif")
parser.add_argument("-n", "--num_samples", type=int, default=0)
args = parser.parse_args()

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Initialise GPU(/CPU)
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print("Initialised processing device:", device)

# Load models and plot learning curves
for i, model_ckpt_path in enumerate([args.model_1_ckpt, args.model_2_ckpt]):
    if model_ckpt_path:
        ckpt, model = load_rcan_checkpoint(
            pathlib.Path(model_ckpt_path), device
        )
        RCAN_hyperparameters = ckpt["hyperparameters"]

        # Plot learning curve
        plot_learning_curve(
            ckpt["losses_train"],
            ckpt["losses_val"],
            ckpt["psnr_train"],
            ckpt["psnr_val"],
            (7, 7),
            str(output_dir / f"model_{i+1}_learning_curve.png"),
        )

gt_dir = pathlib.Path(args.gt_dir)
raw_dir = pathlib.Path(args.raw_dir)
model_1_dir = pathlib.Path(args.model_1_dir)

gt_files = sorted(list(gt_dir.glob(args.glob_str)))
raw_files = sorted(list(raw_dir.glob(args.glob_str)))
model_1_files = sorted(list(model_1_dir.glob(args.glob_str)))

if args.model_2_dir:
    model_2_dir = pathlib.Path(args.model_2_dir)
    model_2_files = sorted(list(model_2_dir.glob(args.glob_str)))
else:
    model_2_files = []

assert len(gt_files) == len(raw_files)
assert len(model_1_files) == len(model_2_files) or model_2_files == []
assert len(model_1_files) == len(raw_files)

psnr = PSNR(data_range=65536, device=device)

ssim = SSIM(
    data_range=65536,
    kernel_size=(11, 11, 11),
    sigma=(1.5, 1.5, 1.5),
    k1=0.01,
    k2=0.03,
    gaussian=True,
    device=device,
)

df = pd.DataFrame(
    columns=[
        "file",
        "psnr_raw",
        "psnr_model_1",
        "psnr_model_2",
        "ssim_raw",
        "ssim_model_1",
        "ssim_model_2",
    ]
)

df["file"] = gt_files

for i in range(len(gt_files)):
    gt = reshape_to_bcwh(tifffile.imread(gt_files[i]))
    raw = reshape_to_bcwh(tifffile.imread(raw_files[i]))
    model_1 = reshape_to_bcwh(tifffile.imread(model_1_files[i]))
    if model_2_files:
        model_2 = reshape_to_bcwh(tifffile.imread(model_2_files[i]))

    # Raw metrics
    psnr.reset()
    psnr.update((torch.from_numpy(raw), torch.from_numpy(gt)))
    df.loc[i, "psnr_raw"] = psnr.compute()
    ssim.reset()
    ssim.update((torch.from_numpy(raw), torch.from_numpy(gt)))
    df.loc[i, "ssim_raw"] = ssim.compute()

    # Model 1 metrics
    psnr.reset()
    psnr.update((torch.from_numpy(model_1), torch.from_numpy(gt)))
    df.loc[i, "psnr_model_1"] = psnr.compute()
    ssim.reset()
    ssim.update((torch.from_numpy(model_1), torch.from_numpy(gt)))
    df.loc[i, "ssim_model_1"] = ssim.compute()

    # Model 2 metrics
    if model_2_files:
        psnr.reset()
        psnr.update((torch.from_numpy(model_2), torch.from_numpy(gt)))
        df.loc[i, "psnr_model_2"] = psnr.compute()
        ssim.reset()
        ssim.update((torch.from_numpy(model_2), torch.from_numpy(gt)))
        df.loc[i, "ssim_model_2"] = ssim.compute()

print(
    f"Mean PSNR raw = {np.mean(df['psnr_raw']):.6f}",
    f" +/- {np.std(df['psnr_raw']):.3f}",
)
print(
    f"Mean PSNR model_1 = {np.mean(df['psnr_model_1']):.6f}",
    f"+/- {np.std(df['psnr_model_1']):.3f}",
)
print(
    f"Mean PSNR model_2 = {np.mean(df['psnr_model_2']):.6f}",
    f"+/- {np.std(df['psnr_model_2']):.3f}",
)

print(
    f"Mean SSIM raw = {np.mean(df['ssim_raw']):.6f}",
    f"+/- {np.std(df['ssim_raw']):.3f}",
)
print(
    f"Mean SSIM model_1 = {np.mean(df['ssim_model_1']):.6f}",
    f"+/- {np.std(df['ssim_model_1']):.3f}",
)
print(
    f"Mean SSIM model_2 = {np.mean(df['ssim_model_2']):.6f}",
    f"+/- {np.std(df['ssim_model_2']):.3f}",
)

df.to_csv(output_dir / "reconstruction_data.csv")

if args.num_samples > 0:
    rng = np.random.default_rng(seed=31052024)
    img_idx = list(range(len(gt_files)))
    rng.shuffle(img_idx)
    img_idx = img_idx[: args.num_samples]
    gt_samples = [np.squeeze(tifffile.imread(gt_files[i])) for i in img_idx]
    raw_samples = [np.squeeze(tifffile.imread(raw_files[i])) for i in img_idx]
    model_1_samples = [
        np.squeeze(tifffile.imread(model_1_files[i])) for i in img_idx
    ]
    if model_2_files:
        model_2_samples = [
            np.squeeze(tifffile.imread(model_2_files[i])) for i in img_idx
        ]
    else:
        model_2_samples = None

    plot_reconstructions(
        device,
        output_dir / "reconstruction_samples.png",
        len(gt_samples[0].shape),
        gt_samples,
        raw_samples,
        model_1_samples,
        model_2_samples,
        cmap="inferno",
    )
