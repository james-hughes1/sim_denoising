# Copyright 2021 SVision Technologies LLC.
# Copyright 2021-2022 Leica Microsystems, Inc.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import argparse
import json
import jsonschema
import numpy as np
import pathlib
import torch
from ignite.metrics import PSNR, SSIM
import tifffile
from tqdm import tqdm

from rcan.data_generator import load_SIM_dataset
from rcan.model import RCAN

# Parse configuration file

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-o", "--output_dir", type=str, required=True)
args = parser.parse_args()

schema = {
    "type": "object",
    "properties": {
        "training_image_pairs": {"$ref": "#/definitions/image_pairs"},
        "validation_image_pairs": {"$ref": "#/definitions/image_pairs"},
        "training_data_dir": {"$ref": "#/definitions/raw_gt_pair"},
        "validation_data_dir": {"$ref": "#/definitions/raw_gt_pair"},
        "input_shape": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "minItems": 2,
            "maxItems": 3,
        },
        "num_input_channels": {"type": "integer", "minimum": 1},
        "num_hidden_channels": {"type": "integer", "minimum": 1},
        "num_output_channels": {"type": "integer", "minimum": 1},
        "num_residual_blocks": {"type": "integer", "minimum": 1},
        "num_residual_groups": {"type": "integer", "minimum": 1},
        "channel_reduction": {"type": "integer", "minimum": 1},
        "epochs": {"type": "integer", "minimum": 1},
        "steps_per_epoch": {"type": "integer", "minimum": 1},
        "batch_size": {"type": "integer", "minimum": 1},
        "num_accumulations": {"type": "integer", "minimum": 1},
        "save_interval": {"type": "integer", "minimum": 1},
        "data_augmentation": {"type": "boolean"},
        "intensity_threshold": {"type": "number"},
        "area_ratio_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "initial_learning_rate": {"type": "number", "minimum": 1e-6},
        "loss": {"type": "string", "enum": ["mae", "mse"]},
        "metrics": {
            "type": "array",
            "items": {"type": "string", "enum": ["psnr", "ssim"]},
        },
    },
    "additionalProperties": False,
    "anyOf": [
        {"required": ["training_image_pairs"]},
        {"required": ["training_data_dir"]},
    ],
    "definitions": {
        "raw_gt_pair": {
            "type": "object",
            "properties": {
                "raw": {"type": "string"},
                "gt": {"type": "string"},
            },
        },
        "image_pairs": {
            "type": "array",
            "items": {"$ref": "#/definitions/raw_gt_pair"},
            "minItems": 1,
        },
    },
}

with open(args.config) as f:
    config = json.load(f)

jsonschema.validate(config, schema)
config.setdefault("epochs", 300)
config.setdefault("steps_per_epoch", 256)
config.setdefault("batch_size", 1)
config.setdefault("num_accumulations", 1)
config.setdefault("save_interval", 10)
config.setdefault("num_input_channels", 9)
config.setdefault("num_hidden_channels", 32)
config.setdefault("num_output_channels", 9)
config.setdefault("num_residual_blocks", 3)
config.setdefault("num_residual_groups", 5)
config.setdefault("channel_reduction", 8)
config.setdefault("data_augmentation", True)
config.setdefault("intensity_threshold", 0.25)
config.setdefault("area_ratio_threshold", 0.5)
config.setdefault("initial_learning_rate", 1e-4)
config.setdefault("loss", "mae")
config.setdefault("metrics", ["psnr"])


def load_data_paths(config, data_type):
    # Create list of pairs of files for either training, or validation dataset.
    image_pair_list = config.get(data_type + "_image_pairs", [])
    input_shape_list = []

    if data_type + "_data_dir" in config:
        raw_dir, gt_dir = [
            pathlib.Path(config[data_type + "_data_dir"][t])
            for t in ["raw", "gt"]
        ]

        raw_files, gt_files = [
            sorted(d.glob("*.tif")) for d in [raw_dir, gt_dir]
        ]

        if not raw_files:
            raise RuntimeError(f"No TIFF file found in {raw_dir}")

        if len(raw_files) != len(gt_files):
            raise RuntimeError(
                f'"{raw_dir}" and "{gt_dir}" must contain the same number of '
                "TIFF files"
            )

        print(f"Collating and verifying {data_type} data")
        for raw_file, gt_file in zip(raw_files, gt_files):
            print("  - raw:", raw_file)
            print("    gt:", gt_file)

            # Check pair has same shape; save the shape to a list.
            raw_img, gt_img = (
                tifffile.imread(raw_file),
                tifffile.imread(gt_file),
            )
            input_shape_list.append(raw_img.shape[1:])

            if raw_img.shape != gt_img.shape:
                raise ValueError(
                    "Raw and GT images must be the same size: "
                    f"{raw_file} {raw_img.shape} vs. {gt_file} {gt_img.shape}"
                )

            if raw_img.ndim - 1 != len(input_shape_list[0]):
                raise ValueError(
                    "All images must have the same number of dimensions"
                )

            # Save image pair paths.
            image_pair_list.append({"raw": str(raw_file), "gt": str(gt_file)})

    min_input_shape = input_shape_list[0]
    for input_shape in input_shape_list:
        min_input_shape = np.minimum(min_input_shape, input_shape)

    return image_pair_list, min_input_shape


# Load datasets.
training_data, min_input_shape_training = load_data_paths(config, "training")
validation_data, min_input_shape_validation = load_data_paths(
    config, "validation"
)

# Check consistent dimensionality of training and validation data,
# also check that patch size is smaller than images.
# Note we assume that the .tiff files are formatted C, Z, X, Y
ndim = tifffile.imread(training_data[0]["raw"]).ndim - 1

if "input_shape" in config:
    input_shape = config["input_shape"]
    if len(input_shape) != ndim:
        raise ValueError(
            f"`input_shape` must be a {ndim}D array; received: {input_shape}"
        )
else:
    input_shape = (16, 256, 256) if ndim == 3 else (256, 256)

if np.any(input_shape > min_input_shape_training):
    raise ValueError(
        f"`input_shape` must be smaller than images; set as: {input_shape}"
    )

if validation_data:
    if tifffile.imread(validation_data[0]["raw"]).ndim - 1 != ndim:
        raise ValueError("All images must have the same number of dimensions")
    if np.any(input_shape > min_input_shape_validation):
        raise ValueError(
            f"`input_shape` must be smaller than images; set as: {input_shape}"
        )

# Create RCAN model and load to processor.
print("Building RCAN model")
print("  - input_shape =", input_shape)
for s in [
    "num_hidden_channels",
    "num_residual_blocks",
    "num_residual_groups",
    "channel_reduction",
]:
    print(f"  - {s} =", config[s])

model = RCAN(
    input_shape,
    num_input_channels=config["num_input_channels"],
    num_hidden_channels=config["num_hidden_channels"],
    num_residual_blocks=config["num_residual_blocks"],
    num_residual_groups=config["num_residual_groups"],
    channel_reduction=config["channel_reduction"],
    residual_scaling=1.0,
    num_output_channels=config["num_output_channels"],
)

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print("Processor found:", device)
model.to(device)

# Save model hyperparameters
RCAN_hyperparameters = {
    "input_shape": input_shape,
    "num_input_channels": config["num_input_channels"],
    "num_hidden_channels": config["num_hidden_channels"],
    "num_residual_blocks": config["num_residual_blocks"],
    "num_residual_groups": config["num_residual_groups"],
    "channel_reduction": config["channel_reduction"],
    "residual_scaling": 1.0,
    "num_output_channels": config["num_output_channels"],
}


def train(
    train_loader,
    val_loader,
    net,
    batchsize,
    n_accumulations,
    saveinterval,
    nepoch,
):

    loss_function = {
        "mae": torch.nn.L1Loss(),
        "mse": torch.nn.MSELoss(),
    }[config["loss"]]

    optimizer = torch.optim.Adam(
        net.parameters(), lr=config["initial_learning_rate"]
    )

    loss_function.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["epochs"] // 4, gamma=0.5
    )

    losses_train_epoch = []
    losses_val_epoch = []

    psnr_train_epoch = []
    psnr_val_epoch = []

    ssim_train_epoch = []
    ssim_val_epoch = []

    psnr = PSNR(data_range=1.0, device=device)
    ssim = SSIM(
        data_range=1.0,
        kernel_size=(11, 11),
        sigma=(1.5, 1.5),
        k1=0.01,
        k2=0.03,
        gaussian=True,
        device=device,
    )

    for epoch in range(nepoch):
        losses_train_batch = []
        losses_val_batch = []
        description = "Epoch: %d/%d" % (epoch + 1, nepoch)

        psnr.reset()
        ssim.reset()
        net.train()
        for i, bat in enumerate(tqdm(train_loader, desc=description)):
            raw, gt = bat[0], bat[1]
            raw = raw.to(device)
            gt = gt.to(device)

            pred = net(raw)

            # Use Gradient Accumulation
            loss = loss_function(pred, gt)
            loss = loss / n_accumulations
            loss.backward()
            if (i + 1) % n_accumulations == 0:
                optimizer.step()
                optimizer.zero_grad()

            losses_train_batch.append(loss.data.item())
            psnr.update((pred, gt))
            ssim.update(
                (torch.movedim(pred, -1, 1), torch.movedim(gt, -1, 1))
            )  # Shape B,X,Y,C -> B,C,X,Y

        psnr_train_epoch.append(psnr.compute())
        ssim_train_epoch.append(ssim.compute())

        psnr.reset()
        ssim.reset()
        net.eval()
        for raw, gt in val_loader:
            raw = raw.to(device)
            gt = gt.to(device)

            pred = net(raw)
            val_loss = loss_function(pred, gt)
            losses_val_batch.append(val_loss.data.item())
            psnr.update((pred, gt))
            ssim.update((torch.movedim(pred, -1, 1), torch.movedim(gt, -1, 1)))

        psnr_val_epoch.append(psnr.compute())
        ssim_val_epoch.append(ssim.compute())

        # Display epoch results and save.
        losses_train_epoch.append(np.average(losses_train_batch))
        losses_val_epoch.append(np.average(losses_val_batch))
        print(
            "Epoch %d done, loss=%0.6f, val_loss=%0.6f"
            % (epoch, losses_train_epoch[-1], losses_val_epoch[-1])
        )
        print(
            "PSNR: train=%0.6f, val=%0.6f"
            % (psnr_train_epoch[-1], psnr_val_epoch[-1])
        )
        print(
            "SSIM: train=%0.6f, val=%0.6f"
            % (ssim_train_epoch[-1], ssim_val_epoch[-1])
        )

        if (epoch + 1) % saveinterval == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "hyperparameters": RCAN_hyperparameters,
                "losses_train": losses_train_epoch,
                "losses_val": losses_val_epoch,
                "psnr_train": psnr_train_epoch,
                "psnr_val": psnr_val_epoch,
                "ssim_train": ssim_train_epoch,
                "ssim_val": ssim_val_epoch,
            }
            checkpoint_filepath = "weights_{0:03d}_{1:.8f}.pth".format(
                epoch + 1, losses_val_epoch[-1]
            )
            torch.save(checkpoint, str(output_dir / checkpoint_filepath))

    checkpoint = {
        "epoch": nepoch,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "hyperparameters": RCAN_hyperparameters,
        "losses_train": losses_train_epoch,
        "losses_val": losses_val_epoch,
        "psnr_train": psnr_train_epoch,
        "psnr_val": psnr_val_epoch,
        "ssim_train": ssim_train_epoch,
        "ssim_val": ssim_val_epoch,
    }
    checkpoint_filepath = "final_{0:03d}_{1:.8f}.pth".format(
        nepoch, losses_val_epoch[-1]
    )
    torch.save(checkpoint, str(output_dir / checkpoint_filepath))


train_loader = load_SIM_dataset(
    training_data,
    input_shape,
    batch_size=config["batch_size"],
    transform_function=(
        "rotate_and_flip" if config["data_augmentation"] else None
    ),
    intensity_threshold=config["intensity_threshold"],
    area_threshold=config["area_ratio_threshold"],
    scale_factor=1,
    steps_per_epoch=config["steps_per_epoch"],
)

if validation_data is not None:
    val_loader = load_SIM_dataset(
        validation_data,
        input_shape,
        batch_size=config["batch_size"],
        transform_function=(
            "rotate_and_flip" if config["data_augmentation"] else None
        ),
        intensity_threshold=config["intensity_threshold"],
        area_threshold=config["area_ratio_threshold"],
        scale_factor=1,
        steps_per_epoch=config["steps_per_epoch"],
    )

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("Training RCAN model")

train(
    train_loader,
    val_loader,
    model,
    config["batch_size"],
    n_accumulations=config["num_accumulations"],
    saveinterval=config["save_interval"],
    nepoch=config["epochs"],
)
