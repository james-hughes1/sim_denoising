# Copyright 2021 SVision Technologies LLC.
# Copyright 2021-2022 Leica Microsystems, Inc.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import argparse
import json
import jsonschema
import numpy as np
import pathlib
import torch
import tifffile
from tqdm import tqdm
import time

from rcan.data_generator import load_SIM_dataset
from rcan.model import RCAN

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-o', '--output_dir', type=str, required=True)
args = parser.parse_args()

schema = {
    'type': 'object',
    'properties': {
        'training_image_pairs': {'$ref': '#/definitions/image_pairs'},
        'validation_image_pairs': {'$ref': '#/definitions/image_pairs'},
        'training_data_dir': {'$ref': '#/definitions/raw_gt_pair'},
        'validation_data_dir': {'$ref': '#/definitions/raw_gt_pair'},
        'input_shape': {
            'type': 'array',
            'items': {'type': 'integer', 'minimum': 1},
            'minItems': 2,
            'maxItems': 3,
        },
        'num_channels': {'type': 'integer', 'minimum': 1},
        'num_residual_blocks': {'type': 'integer', 'minimum': 1},
        'num_residual_groups': {'type': 'integer', 'minimum': 1},
        'channel_reduction': {'type': 'integer', 'minimum': 1},
        'epochs': {'type': 'integer', 'minimum': 1},
        'steps_per_epoch': {'type': 'integer', 'minimum': 1},
        'batch_size': {'type': 'integer', 'minimum': 1},
        'num_accumulations': {'type': 'integer', 'minimum': 1},
        'save_interval': {'type': 'integer', 'minimum': 1},
        'data_augmentation': {'type': 'boolean'},
        'intensity_threshold': {'type': 'number'},
        'area_ratio_threshold': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'initial_learning_rate': {'type': 'number', 'minimum': 1e-6},
        'loss': {'type': 'string', 'enum': ['mae', 'mse']},
        'metrics': {
            'type': 'array',
            'items': {'type': 'string', 'enum': ['psnr', 'ssim']},
        },
    },
    'additionalProperties': False,
    'anyOf': [
        {'required': ['training_image_pairs']},
        {'required': ['training_data_dir']},
    ],
    'definitions': {
        'raw_gt_pair': {
            'type': 'object',
            'properties': {
                'raw': {'type': 'string'},
                'gt': {'type': 'string'},
            },
        },
        'image_pairs': {
            'type': 'array',
            'items': {'$ref': '#/definitions/raw_gt_pair'},
            'minItems': 1,
        },
    },
}


def load_data_paths(config, data_type):
    image_pair_list = config.get(data_type + '_image_pairs', [])
    ndim_list = []
    input_shape_list = []

    if data_type + '_data_dir' in config:
        raw_dir, gt_dir = [
            pathlib.Path(config[data_type + '_data_dir'][t])
            for t in ['raw', 'gt']
        ]

        raw_files, gt_files = [
            sorted(d.glob('*.tif')) for d in [raw_dir, gt_dir]
        ]

        if not raw_files:
            raise RuntimeError(f'No TIFF file found in {raw_dir}')

        if len(raw_files) != len(gt_files):
            raise RuntimeError(
                f'"{raw_dir}" and "{gt_dir}" must contain the same number of '
                'TIFF files'
            )

        for raw_file, gt_file in zip(raw_files, gt_files):
            image_pair_list.append({'raw': str(raw_file), 'gt': str(gt_file)})

    if not image_pair_list:
        return None, None

    print(f'Verifying {data_type} data')
    for p in image_pair_list:
        raw_file, gt_file = [p[t] for t in ['raw', 'gt']]

        print('  - raw:', raw_file)
        print('    gt:', gt_file)

        raw, gt = [tifffile.imread(p[t]) for t in ['raw', 'gt']]
        ndim_list.append(raw.ndim)
        input_shape_list.append(raw.shape)

        if raw.shape != gt.shape:
            raise ValueError(
                'Raw and GT images must be the same size: '
                f'{p["raw"]} {raw.shape} vs. {p["gt"]} {gt.shape}'
            )
    for ndim in ndim_list:
        if ndim != ndim_list[0]:
            raise ValueError(
                'All images must have the same number of dimensions'
            )

    min_input_shape = input_shape_list[0]
    for input_shape in input_shape_list:
        min_input_shape = np.minimum(min_input_shape, input_shape)

    return image_pair_list, min_input_shape


with open(args.config) as f:
    config = json.load(f)

jsonschema.validate(config, schema)
config.setdefault('epochs', 300)
config.setdefault('steps_per_epoch', 256)
config.setdefault('batch_size', 1)
config.setdefault('num_accumulations', 1)
config.setdefault('save_interval', 10)
config.setdefault('num_channels', 32)
config.setdefault('num_residual_blocks', 3)
config.setdefault('num_residual_groups', 5)
config.setdefault('channel_reduction', 8)
config.setdefault('data_augmentation', True)
config.setdefault('intensity_threshold', 0.25)
config.setdefault('area_ratio_threshold', 0.5)
config.setdefault('initial_learning_rate', 1e-4)
config.setdefault('loss', 'mae')
config.setdefault('metrics', ['psnr'])

training_data, min_input_shape_training = load_data_paths(config, 'training')
validation_data, min_input_shape_validation = load_data_paths(
    config, 'validation'
)

ndim = tifffile.imread(training_data[0]['raw']).ndim

if validation_data:
    if tifffile.imread(validation_data[0]['raw']).ndim != ndim:
        raise ValueError('All images must have the same number of dimensions')

if 'input_shape' in config:
    input_shape = config['input_shape']
    if len(input_shape) != ndim:
        raise ValueError(
            f'`input_shape` must be a {ndim}D array; received: {input_shape}'
        )
else:
    input_shape = (16, 256, 256) if ndim == 3 else (256, 256)

input_shape = np.minimum(input_shape, min_input_shape_training)
if validation_data:
    input_shape = np.minimum(input_shape, min_input_shape_validation)

print('Building RCAN model')
print('  - input_shape =', input_shape)
for s in [
    'num_channels',
    'num_residual_blocks',
    'num_residual_groups',
    'channel_reduction',
]:
    print(f'  - {s} =', config[s])

model = RCAN(
    (*input_shape, 1),
    num_channels=config['num_channels'],
    num_residual_blocks=config['num_residual_blocks'],
    num_residual_groups=config['num_residual_groups'],
    channel_reduction=config['channel_reduction'],
)

model.cuda()

def train(dataloader, validloader, net, batchsize, n_accumulations, saveinterval, log=True, nepoch=10):

    loss_function = {
        'mae': torch.nn.L1Loss(),
        'mse': torch.nn.MSELoss(),
    }[config['loss']]

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config['initial_learning_rate']
    )

    loss_function.cuda()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['epochs'] // 4, gamma=0.5)
    count = 0
    t0 = time.perf_counter()

    for epoch in range(nepoch):
        mean_loss = 0
        description = 'Epoch: %d/%d' % (epoch+1,nepoch)

        net.train()
        for i, bat in enumerate(tqdm(dataloader, desc=description)):
            data, labels = bat[0], bat[1]
            data, labels = data.cuda(), labels.cuda()

            target = net(data)

            loss = loss_function(target, labels)  # Compute loss function
            loss = loss / n_accumulations  # Normalize our loss (if averaged)
            loss.backward()

            if (i+1) % n_accumulations == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()  # Reset gradients tensors

            # ------- Status and display ------------
            mean_loss += loss.data.item()

            count += 1
            if log and count*batchsize // 1000 > 0:
                t1 = time.perf_counter() - t0
                mem = torch.cuda.memory_allocated()
                print(epoch, count*batchsize, t1, mem,
                      mean_loss / count)
                count = 0

        net.eval()
        for data, labels in validloader:
            data, labels = data.cuda(), labels.cuda()

            target = net(data)
            loss = loss_function(target, labels)
            val_loss = loss.item() * data.size(0)

        # ---------------- Printing -----------------
        print('Epoch %d done, loss=%0.6f, val_loss=%0.6f' %
              (epoch, (mean_loss / len(dataloader)), val_loss))

        # TO DO: Implement Metrics (PSNR, SSIM).

        if (epoch + 1) % saveinterval == 0:
            checkpoint = {'epoch': epoch + 1,
                          'state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict()}
            checkpoint_filepath = 'weights_{0:03d}_{1:.8f}.pth'.format(epoch + 1, val_loss)
            torch.save(checkpoint, str(output_dir / checkpoint_filepath))

    checkpoint = {'epoch': nepoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
    checkpoint_filepath = 'final_{0:03d}_{1:.8f}.pth'.format(nepoch, val_loss)
    torch.save(checkpoint, str(output_dir / checkpoint_filepath))


dataloader = load_SIM_dataset(
    training_data,
    input_shape,
    batch_size=config['batch_size'],
    transform_function=(
        'rotate_and_flip' if config['data_augmentation'] else None
    ),
    intensity_threshold=config['intensity_threshold'],
    area_threshold=config['area_ratio_threshold'],
)

if validation_data is not None:
    validloader = load_SIM_dataset(
        validation_data,
        input_shape,
        batch_size=config['batch_size'],
        transform_function=(
            'rotate_and_flip' if config['data_augmentation'] else None
        ),
        intensity_threshold=config['intensity_threshold'],
        area_threshold=config['area_ratio_threshold'],
    )

steps_per_epoch = config['steps_per_epoch']
validation_steps = None if validation_data is None else steps_per_epoch

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print('Training RCAN model')

train(
    dataloader,
    validloader,
    model,
    config['batch_size'],
    n_accumulations=config['num_accumulations'],
    saveinterval=config['save_interval'],
    log=True,
    nepoch=config['epochs']
)
