import pathlib
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import t as t_distr

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--dataset", type=str, required=True)
parser.add_argument("-m", "--mode", type=int, choices=[1, 2], required=True)
parser.add_argument("-o", "--output_dir", type=str, default="figures")
parser.add_argument("-s", "--filename_str", type=str, default="")
args = parser.parse_args()

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


def paired_t(gt_data, data):
    z = np.array(data) - np.array(gt_data)
    t = np.mean(z) / (np.std(z, ddof=1)/np.sqrt(len(z)))
    sig = 1 - t_distr.cdf(t, df=len(z)-1)
    pos = np.sum(np.array(data) > np.array(gt_data))
    return sig, pos


df = pd.read_csv(
    pathlib.Path(args.dataset),
    index_col=False
).drop(columns="Unnamed: 0")

df = df[df["file"].str.contains(args.filename_str)]

if args.mode == 1:
    fig, ax = plt.subplots(2, 2, figsize=(10, 15))
else:
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

# Histogram ranges
psnr_diff_1_max = np.max(
    np.array(df["psnr_model_1"]) - np.array(df["psnr_raw"])
)
psnr_diff_2_max = np.max(
    np.array(df["psnr_model_2"]) - np.array(df["psnr_raw"])
)
psnr_diff_1_min = np.min(
    np.array(df["psnr_model_1"]) - np.array(df["psnr_raw"])
)
psnr_diff_2_min = np.min(
    np.array(df["psnr_model_2"]) - np.array(df["psnr_raw"])
)
hist_range_psnr = (
    min(psnr_diff_1_min, psnr_diff_2_min),
    max(psnr_diff_1_max, psnr_diff_2_max)
)

ssim_diff_1_max = np.max(
    np.array(df["ssim_model_1"]) - np.array(df["ssim_raw"])
)
ssim_diff_2_max = np.max(
    np.array(df["ssim_model_2"]) - np.array(df["ssim_raw"])
)
ssim_diff_1_min = np.min(
    np.array(df["ssim_model_1"]) - np.array(df["ssim_raw"])
)
ssim_diff_2_min = np.min(
    np.array(df["ssim_model_2"]) - np.array(df["ssim_raw"])
)
hist_range_ssim = (
    min(ssim_diff_1_min, ssim_diff_2_min),
    max(ssim_diff_1_max, ssim_diff_2_max)
)

# 1st step improvements
ax[1, 0].set(xlabel="$\Delta$(PSNR)", title="Model 1 vs Raw")
ax[1, 1].set(xlabel="$\Delta$(SSIM)", title="Model 1 vs Raw")

ax[1, 0].hist(
    np.array(df["psnr_model_1"]) - np.array(df["psnr_raw"]),
    range=hist_range_psnr
)
ax[1, 0].vlines(
    np.mean(np.array(df["psnr_model_1"]) - np.array(df["psnr_raw"])),
    0,
    12,
    color="black"
)
print(
    "PSNR Model 1 vs Raw t-Test: alpha = "
    f"{paired_t(df['psnr_raw'], df['psnr_model_1'])[0]:.5e}"
)

mean_psnr_1 = np.mean(np.array(df['psnr_model_1']) - np.array(df['psnr_raw']))
se_psnr_1 = np.std(
    np.array(df['psnr_model_1']) - np.array(df['psnr_raw']), ddof=1
)/np.sqrt(len(df['psnr_model_1']))
print(f"PSNR Model 1 - Raw = {mean_psnr_1:.4f} +/- {se_psnr_1:.4f}")

ax[1, 1].hist(
    np.array(df["ssim_model_1"]) - np.array(df["ssim_raw"]),
    range=hist_range_ssim
)
ax[1, 1].vlines(
    np.mean(np.array(df["ssim_model_1"]) - np.array(df["ssim_raw"])),
    0,
    12,
    color="black"
)
print(
    "SSIM Model 1 vs Raw t-Test: alpha = "
    f"{paired_t(df['ssim_raw'], df['ssim_model_1'])[0]:.5e}"
)
mean_ssim_1 = np.mean(np.array(df['ssim_model_1']) - np.array(df['ssim_raw']))
se_ssim_1 = np.std(
    np.array(df['ssim_model_1']) - np.array(df['ssim_raw']), ddof=1
)/np.sqrt(len(df['ssim_model_1']))
print(f"SSIM Model 1 - Raw = {mean_ssim_1:.4f} +/- {se_ssim_1:.4f}")

if args.mode == 2:
    # 2nd step improvements
    ax[2, 0].set(xlabel="$\Delta$(PSNR)", title="Model 2 vs Raw")
    ax[2, 1].set(xlabel="$\Delta$(SSIM)", title="Model 2 vs Raw")

    ax[2, 0].hist(
        np.array(df["psnr_model_2"]) - np.array(df["psnr_raw"]),
        range=hist_range_psnr
    )
    ax[2, 0].vlines(
        np.mean(np.array(df["psnr_model_2"]) - np.array(df["psnr_raw"])),
        0,
        12,
        color="black"
    )
    print(
        "PSNR Model 2 vs Raw t-Test: alpha ="
        f" {paired_t(df['psnr_raw'], df['psnr_model_2'])[0]:.5e}"
    )
    mean_psnr_2 = np.mean(
        np.array(df['psnr_model_2']) - np.array(df['psnr_raw'])
    )
    se_psnr_2 = np.std(
        np.array(df['psnr_model_2']) - np.array(df['psnr_raw']), ddof=1
    )/np.sqrt(len(df['psnr_model_2']))
    print(f"PSNR Model 2 - Raw = {mean_psnr_2:.4f} +/- {se_psnr_2:.4f}")
    ax[2, 1].hist(
        np.array(df["ssim_model_2"]) - np.array(df["ssim_raw"]),
        range=hist_range_ssim
    )
    ax[2, 1].vlines(
        np.mean(np.array(df["ssim_model_2"]) - np.array(df["ssim_raw"])),
        0,
        12,
        color="black"
    )
    print(
        "SSIM Model 2 vs Raw t-Test: alpha ="
        f" {paired_t(df['ssim_raw'], df['ssim_model_2'])[0]:.5e}"
    )
    mean_ssim_2 = np.mean(
        np.array(df['ssim_model_2']) - np.array(df['ssim_raw'])
    )
    se_ssim_2 = np.std(
        np.array(df['ssim_model_2']) - np.array(df['ssim_raw']), ddof=1
    )/np.sqrt(len(df['ssim_model_2']))
    print(f"SSIM Model 2 - Raw = {mean_ssim_2:.4f} +/- {se_ssim_2:.4f}")

# Pivot to long-form dataframe
psnr_cols = df.columns[1:4] if args.mode == 2 else df.columns[1:3]
ssim_cols = df.columns[4:7] if args.mode == 2 else df.columns[3:5]

dflong = pd.melt(
    df,
    id_vars=["file"],
    value_vars=df.columns[1:4],
    var_name="type",
    value_name="psnr"
)
dflongssim = pd.melt(
    df,
    id_vars=["file"],
    value_vars=df.columns[4:7],
    var_name="type",
    value_name="ssim"
)
dflong["type"] = [x[5:] for x in dflong["type"]]
dflong["ssim"] = dflongssim["ssim"]

sns.pointplot(
    data=dflong,
    x="type",
    y="psnr",
    hue="file",
    dodge=False,
    legend=False,
    palette="tab20",
    alpha=0.7,
    ax=ax[0, 0],
    lw=2
)
sns.pointplot(
    data=dflong,
    x="type",
    y="ssim",
    hue="file",
    dodge=False,
    legend=False,
    palette="tab20",
    alpha=0.7,
    ax=ax[0, 1],
    lw=2
)

ax[0, 0].set(xlabel="")
ax[0, 1].set(xlabel="")

plt.tight_layout()
plt.savefig(output_dir / "pipeline_stats.png")
