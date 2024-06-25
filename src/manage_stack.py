"""!@file manage_stack.py
@brief Script handling the stacking and unstacking of groups of images, for
the purpose of batch reconstructions.

@details Takes a directory of images as input, and either stacks or unstacks
the images there according to the configuration. 3D Image Volumes are expected
to be in PAZ format. Note in unstack mode, images are saved with a first
dimension of length 1 - this is the correct format for training the second
step models (CZXY).

Arguments:
- i: directory path of input images
- o: directory path of output images
- n: output image name prefix - only applies in 'stack' mode
- d: dimension
- q: number of SIM acquisitions per image
- g: glob string used to choose images from input directory
- u: if used, sets mode to 'unstack'
- s: start index of sorted input files to process
- e: end index of sorted input files to process
- t: number of images to stack together - only applies in 'stack' mode.
Default: -1 (all images are stacked)
- z: number of z slices of images - only applies in 'unstack' mode
"""

import tifffile
import argparse
import pathlib

from rcan.utils.data_processing import ImageStack


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
parser.add_argument("-z", "--z_slices", type=int, default=1)
args = parser.parse_args()

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Sort directory files and select subset
files = sorted(list(pathlib.Path(args.input_dir).glob(args.glob_str)))

if args.end_index != -1:
    files = files[args.start_index : args.end_index]
else:
    files = files[args.start_index :]

if not args.unstack:
    # Stacking procedure
    stack_number = len(files) if args.stack_number == -1 else args.stack_number

    # Compute number of stacks; if number of files isn't a multiple of stack
    # number, this is no. files // stack_number + 1 for the remainder.
    number_of_stacks = len(files) // stack_number
    if len(files) % stack_number != 0:
        number_of_stacks += 1

    for stack_idx in range(number_of_stacks):
        sample = tifffile.imread(files[0])
        stack_handler = ImageStack(
            args.dimension,
            stack_number,
            stack_idx,
            sample,
            files,
            args.num_acquisitions,
        )
        for i, input_file in enumerate(
            files[stack_idx * stack_number : (stack_idx + 1) * stack_number]
        ):
            print("\nProcessing", input_file.name)
            img_data = tifffile.imread(input_file)
            stack_handler.add_image(img_data, i)
        filename = (
            args.output_name
            + f"_stack{stack_idx*stack_number:04d}"
            + f"_{(stack_idx+1)*stack_number:04d}"
        )
        output_file = output_dir / filename
        tifffile.imwrite(
            str(output_file) + ".tif", stack_handler.export_stack()
        )

else:
    # Unstacking procedure
    assert tifffile.imread(files[0]).shape[0] % args.z_slices == 0
    for i, input_file in enumerate(files):
        print("\nProcessing", input_file.name)
        img_data = tifffile.imread(input_file)
        for j in range(img_data.shape[0] // args.z_slices):
            output_file = output_dir / f"{j:05d}_{input_file.name}"
            output_data = img_data[j * args.z_slices : (j + 1) * args.z_slices]
            if args.z_slices != 1:
                # Prepends a single channel
                output_data = output_data.reshape((1, *output_data.shape))
            tifffile.imwrite(str(output_file), output_data)
