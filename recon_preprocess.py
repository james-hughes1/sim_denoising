import tifffile
import numpy as np
import argparse
import pathlib
from rcan.utils import normalize, percentile


def normalize_acquisition_intensity(data, dim):
    if dim == 2:
        assert len(data.shape) == 3
        print(f"Found {data.shape[0]} acquisitions")
        mean_total_intensity = np.mean(np.sum(data, axis=(1, 2)))
        for acq_idx in range(data.shape[0]):
            normalization_factor = mean_total_intensity / np.sum(data[acq_idx])
            data[acq_idx] = data[acq_idx] * normalization_factor
        return data
    elif dim == 3:
        # Assume czxy format for 3D data.
        assert len(data.shape) == 4
        print(f"Found {data.shape[0]} acquisitions, shape {data.shape[1:]}")
        mean_total_intensity = np.mean(np.sum(data, axis=(1, 2, 3)))
        for acq_idx in range(data.shape[0]):
            normalization_factor = mean_total_intensity / np.sum(data[acq_idx])
            data[acq_idx] = data[acq_idx] * normalization_factor
        return data
    else:
        print("Data must either be 2D or 3D (in czxy format)")


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, required=True)
parser.add_argument("-o", "--output_dir", type=str, required=True)
parser.add_argument(
    "-d", "--dimension", type=int, choices=[2, 3], required=True
)
parser.add_argument("-l", "--bkg_quantile", type=percentile, default=10.0)
parser.add_argument("-u", "--bright_quantile", type=percentile, default=99.9)
parser.add_argument(
    "-n", "--normalize_acquisition_intensity", action="store_true"
)
args = parser.parse_args()

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(list(pathlib.Path(args.input_dir).glob("*.tif")))

for input_file in files:
    print("\nProcessing", input_file.name)
    img_data = tifffile.imread(input_file).astype("float32")

    if args.normalize_acquisition_intensity:
        # Normalize acquisitions
        img_data = normalize_acquisition_intensity(img_data, args.dimension)

    # Map bkg threshold to zero, bright threshold to one.
    img_data = normalize(
        tifffile.imread(input_file),
        p_min=args.bkg_quantile,
        p_max=args.bright_quantile,
    )

    # Clip background and bright values
    img_data = 65535 * np.clip(img_data, 0.0, 1.0)

    output_file = output_dir / input_file.name
    tifffile.imwrite(str(output_file), img_data.astype("uint16"))
