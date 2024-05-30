import tifffile
import numpy as np
import argparse
import pathlib

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, required=True)
parser.add_argument("-o", "--output_dir", type=str, required=True)
args = parser.parse_args()

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(list(pathlib.Path(args.input_dir).glob("*.tif")))

for input_file in files:
    print("\nProcessing", input_file.name)
    img_data = tifffile.imread(input_file)

    # Clip in 0-65536
    img_data = np.clip(img_data, 0, 65535).astype("uint16")
    output_file = output_dir / input_file.name
    tifffile.imwrite(str(output_file), img_data)
