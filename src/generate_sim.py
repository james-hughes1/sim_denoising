"""!@file generate_sim.py
@brief Script simulating the acquisition of 3D SIM image volumes.

@details Takes a directory of 3D image volumes as input, and produces
synthetic 3-beam SIM volumes of size (15, 32, 256, 256).

Arguments:
- i: directory path of input volumes
- o: directory path of output volumes
- s: start index of sorted input files to process
- e: end index of sorted input files to process
- z: z_offset, used to specify the region of the input volume to use.
"""

import argparse

from synthetic_sim.simulation import SimulationRunner

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-s", "--start", type=int, default=0)
parser.add_argument("-e", "--end", type=int, default=1)
parser.add_argument("-z", "--z_offset", type=int, default=0)
args = parser.parse_args()

# Run simulation of SIM imaging
runner = SimulationRunner(
    args.input, args.output, range(args.start, args.end), args.z_offset
)
runner.run()
