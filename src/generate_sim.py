"""!@file generate_sim.py
@brief Script simulating the acquisition of 3D SIM image volumes.

@details Takes a directory of 3D image volumes as input, and produces
synthetic 3-beam SIM volumes of size (15, 32, 256, 256).

Arguments:
- i: directory path of input volumes
- o: directory path of output volumes
- s: start index of input files to process
- e: end index of input files to process
- z: z_offset, used to specify the region of the input volume to use.
"""

import numpy as np
from synthetic_sim.otf import calc_psf, PsfParameters
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import numexpr as ne
import os
import argparse
from skimage import io
import json
import tifffile
import pathlib


def arange_zero(n, spacing=1):
    "Returns an array A with A[n//2] = 0.0 and A[m] - A[m-1] = spacing."
    return spacing * (np.arange(n) - n // 2)


def threshold_norm(sample):
    """Applies a threshold and normalises the sample to improve contrast"""
    sample = sample.astype(float)
    sample -= np.min(sample)
    sample /= np.max(sample)
    hist, bins = np.histogram(sample, 16)
    ind = np.where(hist == np.amax(hist))
    mini = bins[ind[0][0]]
    maxi = bins[ind[0][0] + 1]
    sub = (maxi + mini) / 2
    sample = sample - sub
    sample[sample < 0] = 0
    sample /= np.max(sample)
    return sample**2


class Simulator:
    """The Simulator class encapsulates the state of a 3D microscope
    simulation. A single instance of this class corresponds to a specific set
    of microscope parameters. These parameters are randomly chosen upon object
    creation."""

    def __init__(self, **kwargs):
        "Initialises constant parameters"
        self.n_shifts = 5  # no. of phase shifts
        self.n_angles = 3  # no. of angles
        self.alpha: float = 0.0  # orientation offset
        self.n_x = 256
        self.n_z = 64
        self.n_rotations = 3  # number of beam rotations
        self.n_shifts = 5  # number of phase shifts
        self.res_axial = self.res_lateral = 50e-9

        self.delta_z_p = self.res_axial * np.linspace(
            -self.n_z // 2, self.n_z // 2 - 1, self.n_z
        )

        self._psf = None
        self._superres_psf = None
        self._illumination = None

        self.randomise()
        self.__dict__.update(kwargs)

    def randomise(self):
        "Initialises random parameters"
        rng = np.random.default_rng(seed=10052024)
        self.n_sample = rng.uniform(1.3, 1.33)  # sample RI
        self.n_i = rng.uniform(1.33, 1.5)  # immersion layer RI
        self.n_g = rng.uniform(1.45, 1.55)  # coverslip RI
        self.t_g: float = rng.uniform(
            150e-6, 200e-6
        )  # coverslip thickness (m)
        self.NA: float = rng.uniform(1.0, 1.2)  # numerical aperture

        z_max = (self.n_i / self.n_sample) * self.delta_z_p[0]
        # stage displacement relative to ideal working distance; must be < 0
        self.z = rng.uniform(z_max, 0.7 * z_max)
        self.z_p = -(self.n_sample / self.n_i) * self.z + self.delta_z_p
        self.angle_error = rng.uniform(-np.pi, np.pi)
        self.poisson_photons = rng.integers(40000, 80000)
        self.signal_to_noise = rng.uniform(50, 100)
        self.lambda0 = rng.uniform(400e-9, 600e-9)
        self.k0 = 2 * np.pi / self.lambda0
        self.lambda_exc = self.lambda0 - 30e-9
        self.k_exc = 2 * np.pi / self.lambda_exc
        # dimensionless radial position of beams entering objective:
        self.beam_position = rng.uniform(0.7, 0.8)

    def params_dict(self):
        return {
            s: getattr(self, s)
            for s in """z angle_error poisson_photons signal_to_noise
            lambda0 lambda_exc alpha NA n_x n_z beam_position
            res_axial res_lateral n_sample n_i n_g t_g
            """.split()
        }

    def psf_params(self):
        "Returns a PsfParameters object for generating an appropriate PSF"
        return PsfParameters(
            size=(128, 128),
            particle_z=self.z_p,
            z=[self.z],
            res_lateral=self.res_lateral,
            num_basis=100,
            num_samp=1000,
            oversampling=2.0,
            NA=self.NA,
            lambda0=self.lambda0,
            mag=100.0,
            ns=self.n_sample,
            ng0=self.n_g,
            ng=self.n_g,
            ni0=self.n_i,
            ni=self.n_i,
            ti0=150e-6,
            tg0=self.t_g,
            tg=self.t_g,
        )

    def wavevectors(self):
        """Calculates wavevectors inside the sample for the three beams,
        for a given number of rotations of those beams.

        Returns ndarray of shape (n_rotations, n_beams, 3), where n_beams = 3
        """
        theta_sample = np.arcsin(self.beam_position * self.NA / self.n_sample)
        unrotated = (
            self.k_exc
            * self.n_sample
            * np.array(
                [
                    [0, np.sin(theta_sample), np.cos(theta_sample)],
                    [0, 0, 1],
                    [0, -np.sin(theta_sample), np.cos(theta_sample)],
                ]
            )
        )
        res = np.zeros((self.n_rotations, 3, 3))
        angle_step = np.pi / self.n_rotations
        for i in range(self.n_rotations):
            rotation = Rotation.from_euler(
                "Z", self.alpha + self.angle_error + i * angle_step
            )
            res[i, :] = rotation.apply(unrotated)
        return res

    def illumination(self):
        """Calculates the illumination intensity in the sample; returns
        ndarray of shape (n_rotations, n_shifts, n_x, n_x, n_z)"""
        if self._illumination is not None:
            return self._illumination
        x = y = arange_zero(self.n_x, self.res_lateral)
        z = arange_zero(self.n_z, self.res_axial)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        R = np.stack([X, Y, Z], axis=-1)
        k = self.wavevectors()  # (n_rotations, n_beams, 3)
        delta_k = k - np.roll(k, 1, axis=1)  # pairwise wavevector differences
        phase = np.tensordot(
            delta_k, R, ([-1], [-1])
        )  # (n_rotations, n_beams, n_x, n_x, n_z)
        shift = 2 * np.pi / self.n_shifts * np.arange(self.n_shifts)
        shifts = np.array(
            [shift, np.zeros(self.n_shifts), -shift]
        )  # (n_beams, n_shifts)
        shifts = np.expand_dims(shifts, 0)  # extra dim for rotations
        shifts -= np.roll(shifts, 1, axis=1)
        ph = phase[:, :, np.newaxis, ...]
        sh = np.expand_dims(shifts, axis=(-1, -2, -3))
        self._illumination = 3 + 2 * ne.evaluate("sum(cos(ph + sh), axis=1)")
        return self._illumination

    def in_focus_plane(self, sample):
        """Returns the designated `ground truth' plane"""
        return sample[:, :, self.n_z // 2]

    def psf(self):
        """Calculates a PSF if it has not been done already"""
        if self._psf is not None:
            return self._psf
        self._psf = calc_psf(self.psf_params()).squeeze()
        return self._psf

    def simulate_sim(self, sample):
        """Calculates the 15 simulated SIM images for a given sample."""
        psf = self.psf()
        illumination = self.illumination()
        image = np.zeros((self.n_rotations, self.n_shifts, self.n_x, self.n_x))
        for j in range(self.n_rotations):
            for k in range(self.n_shifts):
                emission = illumination[j, k, ...] * sample
                for i in range(self.n_z):
                    image[j, k, ...] += fftconvolve(
                        emission[:, :, i], psf[:, :, i], mode="same"
                    )
        image = np.clip(image, 0, None)
        image /= np.max(image)
        return image

    def simulate_ideal_superres(self, sample):
        """Simulates the best-case scenario for a 3D SIM reconstruction, by
        convolving the in-focus plane with a small PSF."""
        ifp = self.in_focus_plane(sample)
        superres_psf = self.superres_psf()
        image = fftconvolve(ifp, superres_psf, mode="same")
        image = np.clip(image, 0, None)
        image /= np.max(image)
        return image

    def add_noise(self, image):
        """Adds a combination of Gaussian and Poissonian noise to the image."""
        rng = np.random.default_rng(seed=10052024)
        noise_std = np.std(image) / np.sqrt(self.signal_to_noise)
        white_noise = rng.normal(0, scale=noise_std, size=image.shape)
        shot_mean = self.poisson_photons * image
        shot_noise = rng.normal(shot_mean, scale=np.sqrt(shot_mean))
        return shot_noise / self.poisson_photons + white_noise


class SimulationRunner:
    """Class which performs a batch of simulations, either sequentially or in
    parallel."""

    def __init__(self, input_dir, output_dir, index_range, z_offset):
        self.input_dir = pathlib.Path(input_dir)
        self.input_files = sorted(self.input_dir.glob("*.tif"))
        self.output_dir = pathlib.Path(output_dir)
        self.range = index_range
        self.z_offset = z_offset

    def do_sim(self, i, sim, vol):
        """Creates a new random virtual microscope simulator, takes a new
        sample from the VHP dataset, runs the simulation on the sample, and
        saves the results, along with the ground truth, in a single TIFF file.
        The parameters are saved in an accompanying JSON file."""
        num_slices = sim.n_z // 2
        z_stack = np.zeros((5 * num_slices * 3, sim.n_x, sim.n_x))
        ground_truth = np.zeros((num_slices, sim.n_x, sim.n_x))
        for z_slice in tqdm(range(num_slices)):
            sample = np.zeros((num_slices * 2, sim.n_x, sim.n_x))
            # Take the corner of the image (if it is larger than (n_z,n_x,n_x))
            sample[z_slice:, :, :] = vol[
                self.z_offset : self.z_offset + num_slices * 2 - z_slice,
                : sim.n_x,
                : sim.n_x,
            ]
            sample = np.ascontiguousarray(np.moveaxis(sample, 0, -1))
            sample = sample.astype(float)
            sample -= np.min(sample)
            sample /= np.max(sample)
            image = sim.simulate_sim(sample)
            ground_truth[z_slice] = sim.in_focus_plane(sample)
            stack = image.reshape(
                [image.shape[0] * image.shape[1], *image.shape[2:]]
            )
            stack /= np.max(stack)
            for angle_number in range(sim.n_angles):
                z_stack[
                    angle_number * 5 * num_slices
                    + z_slice * 5 : angle_number * 5 * num_slices
                    + (z_slice + 1) * 5
                ] = stack[angle_number * 5 : (angle_number + 1) * 5]
        sim_path = os.path.join(self.output_dir, f"{i:06}.tif")
        io.imsave(sim_path, (z_stack * 65535).astype("uint16"))
        gt_path = os.path.join(self.output_dir, f"{i:06}_gt.tif")
        io.imsave(gt_path, (ground_truth * 65535).astype("uint16"))
        d = sim.params_dict()
        js_path = os.path.join(self.output_dir, f"{i:06}.json")
        with open(js_path, "w") as f:
            json.dump(d, f, indent=4, default=float)

    def run(self):
        """Runs a series of simulations sequentially"""
        for i in tqdm(self.range):
            sim = Simulator()
            vol = tifffile.imread(self.input_files[i])
            self.do_sim(i, sim, vol)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-s", "--start", type=int, default=0)
parser.add_argument("-e", "--end", type=int, default=1)
parser.add_argument("-z", "--z_offset", type=int, default=0)
args = parser.parse_args()

runner = SimulationRunner(
    args.input, args.output, range(args.start, args.end), args.z_offset
)
runner.run()
