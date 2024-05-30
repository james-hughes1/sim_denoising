import argparse
import pathlib
import tifffile
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-p", "--num_phases", type=int, required=True)
parser.add_argument("-a", "--num_angles", type=int, required=True)
parser.add_argument("-b", "--backwards", action="store_true")
args = parser.parse_args()


input_dir = pathlib.Path(args.input)
input_files = sorted(input_dir.rglob("*.tif"))


if not args.backwards:
    for input_file in input_files:
        original = tifffile.imread(input_file)
        assert len(original.shape) == 3
        n_phases = args.num_phases
        n_angles = args.num_angles
        converted = np.zeros_like(original)
        for z in range(original.shape[0] // (n_phases * n_angles)):
            for a in range(n_angles):
                converted[
                    z * n_phases * n_angles
                    + a * n_phases : z * n_phases * n_angles
                    + (a + 1) * n_phases,
                    ...,
                ] = original[
                    (a * original.shape[0] // n_angles)
                    + z * n_phases : (a * original.shape[0] // n_angles)
                    + (z + 1) * n_phases,
                    ...,
                ]
        print("Saving output image to", input_file)
        tifffile.imwrite(str(input_file), converted.astype("uint16"))

else:
    for input_file in input_files:
        original = tifffile.imread(input_file)
        assert len(original.shape) == 3
        n_phases = args.num_phases
        n_angles = args.num_angles
        converted = np.zeros_like(original)
        for z in range(original.shape[0] // (n_phases * n_angles)):
            for a in range(n_angles):
                converted[
                    (a * original.shape[0] // n_angles)
                    + z * n_phases : (a * original.shape[0] // n_angles)
                    + (z + 1) * n_phases,
                    ...,
                ] = original[
                    z * n_phases * n_angles
                    + a * n_phases : z * n_phases * n_angles
                    + (a + 1) * n_phases,
                    ...,
                ]
        print("Saving output image to", input_file)
        tifffile.imwrite(
            str(input_file), converted.astype("uint16"), imagej=True
        )
