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
        n_phases = args.num_phases
        n_angles = args.num_angles
        converted = np.zeros(
            (
                n_phases * n_angles,
                original.shape[0] // (n_phases * n_angles),
                *original.shape[1:],
            )
        )
        for z in range(original.shape[0] // (n_phases * n_angles)):
            for a in range(n_angles):
                converted[a * n_phases : (a + 1) * n_phases, z, ...] = (
                    original[
                        (a * original.shape[0] // n_angles)
                        + z * n_phases : (a * original.shape[0] // n_angles)
                        + (z + 1) * n_phases,
                        ...,
                    ]
                )
        print("Saving output image to", input_file)
        tifffile.imwrite(str(input_file), converted.astype("uint16"))

else:
    for input_file in input_files:
        original = tifffile.imread(input_file)
        n_phases = args.num_phases
        n_angles = args.num_angles
        assert original.shape[0] == n_phases * n_angles
        converted = np.zeros(
            (
                original.shape[0] * original.shape[1],
                *original.shape[2:],
            )
        )
        for z in range(original.shape[1]):
            for a in range(n_angles):
                converted[
                    (a * converted.shape[0] // n_angles)
                    + z * n_phases : (a * converted.shape[0] // n_angles)
                    + (z + 1) * n_phases,
                    ...,
                ] = original[a * n_phases : (a + 1) * n_phases, z, ...]
        print("Saving output image to", input_file)
        tifffile.imwrite(
            str(input_file), converted.astype("uint16"), imagej=True
        )
