import numpy as np
import torch
from ignite.metrics import PSNR, SSIM
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


def plot_reconstructions(
    device,
    output_path,
    dim,
    z,
    pxl_width,
    gt_imgs,
    raw_imgs,
    model_1_imgs,
    model_2_imgs=None,
    cmap="inferno",
):
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
    if model_2_imgs:
        if dim == 2:
            fig, ax = plt.subplots(num_imgs, 4, figsize=(32, num_imgs * 15))
        else:
            fig, ax = plt.subplots(
                num_imgs * 2,
                4,
                figsize=(36, num_imgs * 12),
                gridspec_kw={
                    "width_ratios": [8, 8, 8, 8],
                    "height_ratios": [8, 2] * num_imgs,
                },
            )
    else:
        if dim == 2:
            fig, ax = fig, ax = plt.subplots(
                num_imgs, 3, figsize=(24, num_imgs * 15)
            )
        else:
            fig, ax = fig, ax = plt.subplots(
                num_imgs * 2,
                3,
                figsize=(27, num_imgs * 12),
                gridspec_kw={
                    "width_ratios": [8, 8, 8],
                    "height_ratios": [8, 2] * num_imgs,
                },
            )

    rng = np.random.default_rng(seed=31052024)

    for i in range(num_imgs):
        if dim == 2:
            gt = gt_imgs[i, :pxl_width, :pxl_width]
            raw = raw_imgs[i, :pxl_width, :pxl_width]
            model_1 = model_1_imgs[i, :pxl_width, :pxl_width]
            model_2 = model_2_imgs[i, :pxl_width, :pxl_width]
            plot_idx = i
        else:
            z_slice = rng.integers(0, z)
            gt = gt_imgs[i, z_slice, :pxl_width, :pxl_width]
            raw = raw_imgs[i, z_slice, :pxl_width, :pxl_width]
            model_1 = model_1_imgs[i, z_slice, :pxl_width, :pxl_width]
            model_2 = model_2_imgs[i, z_slice, :pxl_width, :pxl_width]
            plot_idx = 2 * i

        ax[plot_idx, 0].imshow(gt, cmap=cmap)
        ax[plot_idx, 1].imshow(raw, cmap=cmap)
        ax[plot_idx, 2].imshow(model_1, cmap=cmap)
        if model_2_imgs:
            ax[plot_idx, 3].imshow(model_2, cmap=cmap)

        # Record metrics
        psnr.reset()
        psnr.update((torch.from_numpy(raw), torch.from_numpy(gt)))
        ssim.reset()
        ssim.update((torch.from_numpy(raw), torch.from_numpy(gt)))
        ax[plot_idx, 1].set(
            xlabel=f"psnr = {psnr.compute():.5g} / ssim = {ssim.compute():.5g}"
        )

        psnr.reset()
        psnr.update((torch.from_numpy(model_1), torch.from_numpy(gt)))
        ssim.reset()
        ssim.update((torch.from_numpy(model_1), torch.from_numpy(gt)))
        ax[plot_idx, 2].set(
            xlabel=f"psnr = {psnr.compute():.5g} / ssim = {ssim.compute():.5g}"
        )

        if model_2_imgs:
            psnr.reset()
            psnr.update((torch.from_numpy(model_2), torch.from_numpy(gt)))
            ssim.reset()
            ssim.update((torch.from_numpy(model_2), torch.from_numpy(gt)))
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

        # Plot axial
        if dim == 3:
            z_slice = rng.integers(0, z)
            gt = gt_imgs[i, :, pxl_width // 2, :pxl_width]
            raw = raw_imgs[i, :, pxl_width // 2, :pxl_width]
            model_1 = model_1_imgs[i, :, pxl_width // 2, :pxl_width]
            model_2 = model_2_imgs[i, :, pxl_width // 2, :pxl_width]

            ax[plot_idx + 1, 0].imshow(gt, cmap=cmap)
            ax[plot_idx + 1, 1].imshow(raw, cmap=cmap)
            ax[plot_idx + 1, 2].imshow(model_1, cmap=cmap)
            if model_2_imgs:
                ax[plot_idx + 1, 3].imshow(model_2, cmap=cmap)

            # Record metrics
            psnr.reset()
            psnr.update((torch.from_numpy(raw), torch.from_numpy(gt)))
            ssim.reset()
            ssim.update((torch.from_numpy(raw), torch.from_numpy(gt)))
            ax[plot_idx + 1, 1].set(
                xlabel=f"psnr = {psnr.compute():.5g}"
                + f" / ssim = {ssim.compute():.5g}"
            )

            psnr.reset()
            psnr.update((torch.from_numpy(model_1), torch.from_numpy(gt)))
            ssim.reset()
            ssim.update((torch.from_numpy(model_1), torch.from_numpy(gt)))
            ax[plot_idx + 1, 2].set(
                xlabel=f"psnr = {psnr.compute():.5g}"
                + f" / ssim = {ssim.compute():.5g}"
            )

            if model_2_imgs:
                psnr.reset()
                psnr.update((torch.from_numpy(model_2), torch.from_numpy(gt)))
                ssim.reset()
                ssim.update((torch.from_numpy(model_2), torch.from_numpy(gt)))
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
    ax[0, 2].set(title="Model 1")
    if model_2_imgs:
        ax[0, 3].set(title="Model 2")
    plt.savefig(output_path)
