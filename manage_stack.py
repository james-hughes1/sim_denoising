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
parser.add_argument("-q", "--num_acquisitions", type=int, default=15)
parser.add_argument("-g", "--glob_str", default="*.tif")
parser.add_argument("-u", "--unstack", action="store_true")
parser.add_argument("-s", "--start_index", type=int, default=0)
parser.add_argument("-e", "--end_index", type=int, default=-1)
parser.add_argument("-t", "--stack_number", type=int, default=-1)
args = parser.parse_args()

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(list(pathlib.Path(args.input_dir).glob(args.glob_str)))

if args.end_index != -1:
    files = files[args.start_index : args.end_index]
else:
    files = files[args.start_index :]

if not args.unstack:
    stack_number = len(files) if args.stack_number == -1 else args.stack_number
    number_of_stacks = len(files) // stack_number
    if len(files) % stack_number != 0:
        number_of_stacks += 1
    for stack_idx in range(number_of_stacks):
        if args.dimension == 2:
            sample = tifffile.imread(files[0])
            stack = np.zeros(
                (
                    len(
                        files[
                            stack_idx
                            * stack_number : (stack_idx + 1)
                            * stack_number
                        ]
                    ),
                    *sample.shape,
                )
            ).astype(sample.dtype)

            for i, input_file in enumerate(
                files[
                    stack_idx * stack_number : (stack_idx + 1) * stack_number
                ]
            ):
                print("\nProcessing", input_file.name)
                img_data = tifffile.imread(input_file)
                stack[i, ...] = img_data

            filename = (
                args.output_name
                + f"_stack{stack_idx*stack_number:04d}"
                + f"_{(stack_idx+1)*stack_number:04d}"
            )
            output_file = output_dir / filename
            tifffile.imwrite(str(output_file) + ".tif", stack)
        else:
            # Expect paz format
            n_acq = args.num_acquisitions
            sample = tifffile.imread(files[0])
            n_z = sample.shape[0] // n_acq
            stack = np.zeros(
                (
                    len(
                        files[
                            stack_idx
                            * stack_number : (stack_idx + 1)
                            * stack_number
                        ]
                    )
                    * n_z,
                    n_acq,
                    *sample.shape[1:],
                )
            ).astype(sample.dtype)
            for i, input_file in enumerate(
                files[
                    stack_idx * stack_number : (stack_idx + 1) * stack_number
                ]
            ):
                print("\nProcessing", input_file.name)
                img_data = tifffile.imread(input_file)
                for z in range(n_z):
                    stack[i * n_z + z, ...] = img_data[
                        z * n_acq : (z + 1) * n_acq, ...
                    ]

            filename = (
                args.output_name
                + f"_stack{stack_idx*stack_number:04d}"
                + f"_{(stack_idx+1)*stack_number:04d}"
            )
            output_file = output_dir / filename
            tifffile.imwrite(str(output_file) + ".tif", stack)

else:
    assert tifffile.imread(files[0]).shape[0] % args.num_acquisitions == 0
    for i, input_file in enumerate(files):
        print("\nProcessing", input_file.name)
        img_data = tifffile.imread(input_file)
        for j in range(img_data.shape[0] // args.num_acquisitions):
            output_file = output_dir / f"{j:05d}_{input_file.name}"
            output_data = img_data[
                j * args.num_acquisitions : (j + 1) * args.num_acquisitions
            ]
            # Adds a single channel at the start of the shape (reconstructions)
            output_data = output_data.reshape((1, *output_data.shape))
            tifffile.imwrite(str(output_file), output_data)
