import numpy as np
import torch
from ignite.metrics import PSNR
import matplotlib.pyplot as plt


def plot_learning_curve(
    losses_train,
    losses_val,
    psnr_train,
    psnr_val,
    figsize,
    output_path,
):
    fig, ax = plt.subplots(2, 1, figsize=figsize)

    # Loss plot
    ax[0].plot(losses_train, label="Training Loss", color="red")
    ax[0].plot(losses_val, label="Validation Loss", color="blue")
    ax[0].set(ylabel="Loss")
    ax[0].legend()

    # PSNR plot
    ax[1].plot(psnr_train, label="Training PSNR", color="red")
    ax[1].plot(psnr_val, label="Validation PSNR", color="blue")
    ax[1].set(ylabel="psnr")
    plt.savefig(output_path)


def plot_predictions(
    num_imgs, raw_imgs, gt_imgs, restored_imgs, device, output_path
):
    psnr = PSNR(data_range=1.0, device=device)
    rng = np.random.default_rng(seed=29042024)
    img_idx = list(range(num_imgs))
    rng.shuffle(img_idx)
    fig, ax = plt.subplots(num_imgs, 3, figsize=(24, num_imgs * 8))
    for plot_idx, j in enumerate(img_idx[:num_imgs]):
        acq_idx = rng.integers(low=0, high=9)
        raw = raw_imgs[j][acq_idx]
        restored = restored_imgs[j][acq_idx]
        gt = gt_imgs[j][acq_idx]
        ax[plot_idx, 0].imshow(raw)
        ax[plot_idx, 1].imshow(restored)
        ax[plot_idx, 2].imshow(gt)

        # Record PSNR
        psnr.reset()
        psnr.update((torch.from_numpy(raw), torch.from_numpy(gt)))
        ax[plot_idx, 0].set(xlabel=f"psnr = {psnr.compute():.5g}")
        psnr.reset()
        psnr.update((torch.from_numpy(restored), torch.from_numpy(gt)))
        ax[plot_idx, 1].set(xlabel=f"psnr = {psnr.compute():.5g}")

        # Remove axis labels
        ax[plot_idx, 0].set_xticks([])
        ax[plot_idx, 1].set_xticks([])
        ax[plot_idx, 2].set_xticks([])
        ax[plot_idx, 0].set_yticks([])
        ax[plot_idx, 1].set_yticks([])
        ax[plot_idx, 2].set_yticks([])
    ax[0, 0].set(title="Raw")
    ax[0, 1].set(title="Restored")
    ax[0, 2].set(title="GT")
    plt.savefig(output_path)
