import tifffile
import numpy as np
import argparse
import pathlib


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, required=True)
parser.add_argument("-o", "--output_dir", type=str, required=True)
parser.add_argument("-n", "--output_name", type=str, required=True)
parser.add_argument(
    "-d", "--dimension", type=int, choices=[2, 3], required=True
)
parser.add_argument("-q", "--num_acquisions", type=int, default=15)
parser.add_argument("-g", "--glob_str", default="*.tif")
parser.add_argument("-u", "--unstack", action="store_true")
parser.add_argument("-s", "--start_index", type=int, default=0)
parser.add_argument("-e", "--end_index", type=int, default=-1)
args = parser.parse_args()

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(list(pathlib.Path(args.input_dir).glob(args.glob_str)))

if args.end_index != -1:
    files = files[args.start_index : args.end_index]
else:
    files = files[args.start_index :]

if not args.unstack:
    if args.dimension == 2:
        sample = tifffile.imread(files[0])
        stack = np.zeros((len(files), *sample.shape)).astype(sample.dtype)

        for i, input_file in enumerate(files):
            print("\nProcessing", input_file.name)
            img_data = tifffile.imread(input_file)
            stack[i, ...] = img_data

        output_file = output_dir / args.output_name
        tifffile.imwrite(str(output_file) + ".tif", stack)
    else:
        # Expect paz format
        n_acq = args.num_acquisions
        n_z = sample.shape[0] // n_acq
        sample = tifffile.imread(files[0])
        stack = np.zeros((len(files) * n_z, n_acq, *sample.shape[1:])).astype(
            sample.dtype
        )
        for i, input_file in enumerate(files):
            print("\nProcessing", input_file.name)
            img_data = tifffile.imread(input_file)
            for z in range(n_z):
                stack[i * n_z + z, ...] = img_data[
                    z * n_acq : (z + 1) * n_acq, ...
                ]

        output_file = output_dir / args.output_name
        tifffile.imwrite(str(output_file) + ".tif", stack)

else:
    for i, input_file in enumerate(files):
        print("\nProcessing", input_file.name)
        img_data = tifffile.imread(input_file)
        for j in range(img_data.shape[0]):
            output_file = output_dir / f"{j:05d}_{input_file.name}"
            output_data = img_data[j]
            # Adds a single channel at the start of the shape (reconstructions)
            output_data = output_data.reshape((1, *output_data.shape))
            tifffile.imwrite(str(output_file), output_data)
