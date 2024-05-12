import argparse
import pathlib
import tifffile
import numpy as np
from rcan.utils import tuple_of_ints

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-s", "--start", type=tuple_of_ints, required=True)
parser.add_argument("-j", "--step", type=tuple_of_ints, required=True)
parser.add_argument("-n", "--num_steps", type=tuple_of_ints, required=True)
parser.add_argument("-l", "--label", type=str, default="volume")
args = parser.parse_args()


input_dir = pathlib.Path(args.input)
output_dir = pathlib.Path(args.output)

input_files = sorted(input_dir.glob("*.tif"))
output_dir.mkdir(parents=True, exist_ok=True)

volume = np.zeros((len(input_files), 3061, 4096), dtype=np.uint8)
for i, file in enumerate(input_files):
    print(i)
    input_slice = tifffile.imread(file)
    volume[i] = np.uint16(np.mean(input_slice, axis=-1))

for i in range(args.num_steps[0]):
    for j in range(args.num_steps[1]):
        subvolume = volume[
            :,
            args.start[0]
            + args.step[0] * i : args.start[0]
            + args.step[0] * (i + 1),
            args.start[1]
            + args.step[1] * j : args.start[1]
            + args.step[1] * (j + 1),
        ]
        output_file = (
            output_dir / f"{args.label}_{i*args.num_steps[1] + j:03}.tif"
        )
        print("Saving output image to", output_file)
        tifffile.imwrite(str(output_file), subvolume, imagej=True)
