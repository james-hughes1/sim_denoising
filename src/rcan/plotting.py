"""!
@file plotting.py
@brief Module providing helper functions for matplotlib plots.

@details Provides tools to assist with analysis of trained networks, including
samples of restored reconstructions, metrics, and model progress during
training.
"""

import torch
from ignite.metrics import PSNR, SSIM
import matplotlib.pyplot as plt


def plot_learning_curve(
    losses_train,
    losses_val,
    psnr_train,
    psnr_val,
    ssim_train,
    ssim_val,
    figsize,
    output_path,
):
    """!
    @brief Plots the learning curve metrics from a model checkpoint according
    to loss, PSNR, and SSIM.

    @param losses_train (list[float]) - List of training losses
    @param losses_val (list[float]) - List of validation losses
    @param psnr_train (list[float]) - List of training psnrs
    @param psnr_val (list[float]) - List of validation psnrs
    @param ssim_train (list[float]) - List of training ssims
    @param ssim_val (list[float]) - List of validation ssims
    @param figsize (tuple[int]) - Specifies matplotlib layout size
    @param output_path (str) - Determines where plot is saved
    """
    fig, ax = plt.subplots(3, 1, figsize=figsize)

    # Loss plot
    ax[0].plot(losses_train, label="Training Loss", color="red")
    ax[0].plot(losses_val, label="Validation Loss", color="blue")
    ax[0].set(ylabel="Loss")
    ax[0].legend()

    # PSNR plot
    ax[1].plot(psnr_train, label="Training PSNR", color="red")
    ax[1].plot(psnr_val, label="Validation PSNR", color="blue")
    ax[1].set(ylabel="psnr")

    # SSIM plot
    ax[2].plot(ssim_train, label="Training SSIM", color="red")
    ax[2].plot(ssim_val, label="Validation SSIM", color="blue")
    ax[2].set(ylabel="ssim")
    plt.tight_layout()
    plt.savefig(output_path)


def plot_reconstructions(
    device,
    output_path,
    dim,
    gt_imgs,
    raw_imgs,
    model_1_imgs,
    model_2_imgs=None,
    cmap="inferno",
):
    """!
    @brief Plots a sample of reconstructions comparing GT vs Raw vs Restored.

    @param device (torch.device) - Handles the processing unit for torch
    @param output_path (str) - Determines where the plot is saved
    @param dim (int) - Dimensionality of the images
    @param gt_imgs (list[np.ndarray]) - List containing GT image arrays
    @param raw_imgs (list[np.ndarray]) - List containing Raw image arrays
    @param model_1_imgs (list[np.ndarray]) - List containing Step 1 image
    arrays
    @param model_2_imgs (list[np.ndarray], optional) - List containing Step 2
    image arrays. Default: None
    @param cmap (str) - Matplotlib colormap string
    """
    num_imgs = len(gt_imgs)
    assert dim in [2, 3]

    psnr = PSNR(data_range=65536, device=device)
    ssim = SSIM(
        data_range=65536,
        kernel_size=(11, 11),
        sigma=(1.5, 1.5),
        k1=0.01,
        k2=0.03,
        gaussian=True,
        device=device,
    )

    # Create matplotlib layout
    if model_2_imgs:
        if dim == 2:
            fig, ax = plt.subplots(num_imgs, 4, figsize=(16, num_imgs * 4))
        else:
            fig, ax = plt.subplots(
                num_imgs * 2,
                4,
                figsize=(18, num_imgs * 6),
                gridspec_kw={
                    "width_ratios": [4, 4, 4, 4],
                    "height_ratios": [4, 1] * num_imgs,
                },
            )
    else:
        if dim == 2:
            fig, ax = fig, ax = plt.subplots(
                num_imgs, 3, figsize=(12, num_imgs * 4)
            )
        else:
            fig, ax = fig, ax = plt.subplots(
                num_imgs * 2,
                3,
                figsize=(14, num_imgs * 6),
                gridspec_kw={
                    "width_ratios": [4, 4, 4],
                    "height_ratios": [4, 1] * num_imgs,
                },
            )

    # Plot sample of images
    for i in range(num_imgs):
        if dim == 2:
            gt = gt_imgs[i][128:256, 128:256]
            raw = raw_imgs[i][128:256, 128:256]
            model_1 = model_1_imgs[i][128:256, 128:256]
            if model_2_imgs:
                model_2 = model_2_imgs[i][128:256, 128:256]
            plot_idx = i
        else:
            z_slice = gt_imgs[i].shape[0] // 2
            gt = gt_imgs[i][z_slice, 128:256, 128:256]
            raw = raw_imgs[i][z_slice, 128:256, 128:256]
            model_1 = model_1_imgs[i][z_slice, 128:256, 128:256]
            if model_2_imgs:
                model_2 = model_2_imgs[i][z_slice, 128:256, 128:256]
            # If dimension is 3, plot on every other row to leave room for
            # axial views
            plot_idx = 2 * i

        ax[plot_idx, 0].imshow(gt, cmap=cmap)
        ax[plot_idx, 1].imshow(raw, cmap=cmap)
        ax[plot_idx, 2].imshow(model_1, cmap=cmap)
        if model_2_imgs:
            ax[plot_idx, 3].imshow(model_2, cmap=cmap)

        # Indicate that lateral view is plotted
        ax[plot_idx, 0].set(xlabel="x", ylabel="y")

        # Record metrics
        psnr.reset()
        psnr.update(
            (
                torch.from_numpy(raw)[None, None, ...],
                torch.from_numpy(gt)[None, None, ...],
            )
        )
        ssim.reset()
        ssim.update(
            (
                torch.from_numpy(raw)[None, None, ...],
                torch.from_numpy(gt)[None, None, ...],
            )
        )
        ax[plot_idx, 1].set(
            xlabel=f"psnr = {psnr.compute():.5g} / ssim = {ssim.compute():.5g}"
        )

        psnr.reset()
        psnr.update(
            (
                torch.from_numpy(model_1)[None, None, ...],
                torch.from_numpy(gt)[None, None, ...],
            )
        )
        ssim.reset()
        ssim.update(
            (
                torch.from_numpy(model_1)[None, None, ...],
                torch.from_numpy(gt)[None, None, ...],
            )
        )
        ax[plot_idx, 2].set(
            xlabel=f"psnr = {psnr.compute():.5g} / ssim = {ssim.compute():.5g}"
        )

        if model_2_imgs:
            psnr.reset()
            psnr.update(
                (
                    torch.from_numpy(model_2)[None, None, ...],
                    torch.from_numpy(gt)[None, None, ...],
                )
            )
            ssim.reset()
            ssim.update(
                (
                    torch.from_numpy(model_2)[None, None, ...],
                    torch.from_numpy(gt)[None, None, ...],
                )
            )
            ax[plot_idx, 3].set(
                xlabel=f"psnr = {psnr.compute():.5g}"
                + f" / ssim = {ssim.compute():.5g}"
            )

        # Remove axis labels
        ax[plot_idx, 0].set_xticks([])
        ax[plot_idx, 1].set_xticks([])
        ax[plot_idx, 2].set_xticks([])
        ax[plot_idx, 0].set_yticks([])
        ax[plot_idx, 1].set_yticks([])
        ax[plot_idx, 2].set_yticks([])
        if model_2_imgs:
            ax[plot_idx, 3].set_xticks([])
            ax[plot_idx, 3].set_yticks([])

        # Plot axial views
        if dim == 3:
            gt = gt_imgs[i][:, 192, 128:256]
            raw = raw_imgs[i][:, 192, 128:256]
            model_1 = model_1_imgs[i][:, 192, 128:256]
            if model_2_imgs:
                model_2 = model_2_imgs[i][:, 192, 128:256]

            ax[plot_idx + 1, 0].imshow(gt, cmap=cmap)
            ax[plot_idx + 1, 1].imshow(raw, cmap=cmap)
            ax[plot_idx + 1, 2].imshow(model_1, cmap=cmap)
            if model_2_imgs:
                ax[plot_idx + 1, 3].imshow(model_2, cmap=cmap)

            # Indicate that axial view is plotted
            ax[plot_idx + 1, 0].set(xlabel="x", ylabel="z")

            # Record metrics
            psnr.reset()
            psnr.update(
                (
                    torch.from_numpy(raw)[None, None, ...],
                    torch.from_numpy(gt)[None, None, ...],
                )
            )
            ssim.reset()
            ssim.update(
                (
                    torch.from_numpy(raw)[None, None, ...],
                    torch.from_numpy(gt)[None, None, ...],
                )
            )
            ax[plot_idx + 1, 1].set(
                xlabel=f"psnr = {psnr.compute():.5g}"
                + f" / ssim = {ssim.compute():.5g}"
            )

            psnr.reset()
            psnr.update(
                (
                    torch.from_numpy(model_1)[None, None, ...],
                    torch.from_numpy(gt)[None, None, ...],
                )
            )
            ssim.reset()
            ssim.update(
                (
                    torch.from_numpy(model_1)[None, None, ...],
                    torch.from_numpy(gt)[None, None, ...],
                )
            )
            ax[plot_idx + 1, 2].set(
                xlabel=f"psnr = {psnr.compute():.5g}"
                + f" / ssim = {ssim.compute():.5g}"
            )

            if model_2_imgs:
                psnr.reset()
                psnr.update(
                    (
                        torch.from_numpy(model_2)[None, None, ...],
                        torch.from_numpy(gt)[None, None, ...],
                    )
                )
                ssim.reset()
                ssim.update(
                    (
                        torch.from_numpy(model_2)[None, None, ...],
                        torch.from_numpy(gt)[None, None, ...],
                    )
                )
                ax[plot_idx + 1, 3].set(
                    xlabel=f"psnr = {psnr.compute():.5g}"
                    + f" / ssim = {ssim.compute():.5g}"
                )

            # Remove axis labels
            ax[plot_idx + 1, 0].set_xticks([])
            ax[plot_idx + 1, 1].set_xticks([])
            ax[plot_idx + 1, 2].set_xticks([])
            ax[plot_idx + 1, 0].set_yticks([])
            ax[plot_idx + 1, 1].set_yticks([])
            ax[plot_idx + 1, 2].set_yticks([])
            if model_2_imgs:
                ax[plot_idx + 1, 3].set_xticks([])
                ax[plot_idx + 1, 3].set_yticks([])

    # Set titles
    ax[0, 0].set(title="GT")
    ax[0, 1].set(title="Raw")
    ax[0, 2].set(title="Step 1")
    if model_2_imgs:
        ax[0, 3].set(title="Step 2")
    plt.tight_layout()
    plt.savefig(output_path)
