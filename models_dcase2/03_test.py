#!/usr/bin/env python
'''DCASE model evaluation (part 2)'''

import argparse
import os
import sys
import pickle
import json

import pandas as pd

from jams.util import smkdirs
from milsed.models import MODELS
from milsed.eval import score_model
import milsed.utils

OUTPUT_PATH = os.path.expanduser('~/dev/milsed/models_dcase2/resources')

DCASE_CLASSES = ['Air horn, truck horn',
                 'Ambulance (siren)',
                 'Bicycle',
                 'Bus',
                 'Car',
                 'Car alarm',
                 'Car passing by',
                 'Civil defense siren',
                 'Fire engine, fire truck (siren)',
                 'Motorcycle',
                 'Police car (siren)',
                 'Reversing beeps',
                 'Screaming',
                 'Skateboard',
                 'Train',
                 'Train horn',
                 'Truck']


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--alpha', dest='alpha', type=float,
                        default=1.0, help='Alpha parameter for softmax')

    parser.add_argument('--max_samples', dest='max_samples', type=int,
                        default=128,
                        help='Maximum number of samples to draw per streamer')

    parser.add_argument('--patch-duration', dest='duration', type=float,
                        default=10.0,
                        help='Duration (in seconds) of training patches')

    parser.add_argument('--seed', dest='seed', type=int,
                        default='20170612',
                        help='Seed for the random number generator')

    parser.add_argument('--train-streamers', dest='train_streamers', type=int,
                        default=64,
                        help='Number of active training streamers')

    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=16,
                        help='Size of training batches')

    parser.add_argument('--rate', dest='rate', type=int,
                        default=4,
                        help='Rate of pescador stream deactivation')

    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=150,
                        help='Maximum number of epochs to train for')

    parser.add_argument('--epoch-size', dest='epoch_size', type=int,
                        default=2048,
                        help='Number of batches per epoch')

    parser.add_argument('--validation-size', dest='validation_size', type=int,
                        default=1024,
                        help='Number of batches per validation')

    parser.add_argument('--early-stopping', dest='early_stopping', type=int,
                        default=30,
                        help='# epochs without improvement to stop')

    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int,
                        default=10,
                        help='# epochs before reducing learning rate')

    parser.add_argument('--augment', dest='augment', action='store_const',
                        const=True, default=True,
                        help='Use augmented PITCHSHIFT data for training.')

    parser.add_argument('--augment_drc', dest='augment_drc',
                        action='store_const', const=True, default=False,
                        help='Use augmented DRC data for training.')

    parser.add_argument('--verbose', dest='verbose', action='store_const',
                        const=True, default=False,
                        help='Call keras fit with verbose mode (1)')

    parser.add_argument('--train_balanced', dest='train_balanced',
                        action='store_const', const=True, default=False,
                        help='Balance classes when training')

    parser.add_argument(dest='modelname', type=str,
                        help='Name of model to train')

    parser.add_argument(dest='modelid', type=str,
                        help='Model ID number, e.g. "model001"')

    parser.add_argument(dest='jobid', type=str,
                        help='HPC job ID')

    parser.add_argument(dest='working', type=str,
                        help='Path to working directory')

    parser.add_argument(dest='strong_labels_file', type=str,
                        help='Path to csv file containing strong labels for '
                             'the test set.')

    parser.add_argument('--train-strong', dest='train_strong',
                        action='store_const', const=True, default=False,
                        help='Train with strong labels')

    return parser.parse_args(args)


def test(modelname, modelid, working, strong_label_file, alpha, max_samples,
         duration, rate,
         batch_size, epochs, epoch_size, validation_size,
         early_stopping, reduce_lr, seed, train_streamers, augment,
         augment_drc, verbose, train_balanced, train_strong):
    '''
    Parameters
    ----------
    modelname : str
        name of the model to train

    modelid : str
        Model identifier in the form <modelXXX>, e.g. model001

    working : str
        directory that contains the experiment data (h5)

    strong_label_file : str
        path to CSV file containing strong labels for the test set

    alpha : float > 0
        Alpha parameter for softmax

    max_samples : int
        Maximum number of samples per streamer

    duration : float
        Duration of training patches

    batch_size : int
        Size of batches

    rate : int
        Poisson rate for pescador

    epochs : int
        Maximum number of epoch

    epoch_size : int
        Number of batches per epoch

    validation_size : int
        Number of validation batches

    early_stopping : int
        Number of epochs before early stopping

    reduce_lr : int
        Number of epochs before reducing learning rate

    seed : int
        Random seed

    train_streamers : int
        Number of streamers to keep alive simultaneously during training

    augment : bool
        Include augmented PITCHSHIFT data during training

    augment_drc : bool
        Include augmented DRC data during training

    verbose : bool
        Verbose output during keras training.

    train_strong: bool
        Use strong training labels
    '''

    # Load the pump
    with open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'rb') as fd:
        pump = pickle.load(fd)

    # Build the model
    print('Build model...')
    construct_model = MODELS[modelname]
    model, inputs, outputs = construct_model(pump, alpha)

    # Construct the weight path
    weight_path = os.path.join(OUTPUT_PATH, modelid, 'model.h5')

    # Evaluate model
    print('Evaluate model...')
    # Load best params
    model.load_weights(weight_path)
    test_idx = pd.read_json(os.path.join(OUTPUT_PATH, 'index_test.json'))['id']

    # Compute eval scores
    results = score_model(OUTPUT_PATH, pump, model, test_idx, working,
                          strong_label_file, duration, modelid,
                          use_orig_duration=True, use_tqdm=True)

    # Save results to disk
    results_file = os.path.join(OUTPUT_PATH, modelid, 'results.json')
    with open(results_file, 'w') as fp:
        json.dump(results, fp, indent=2)

    print('Done!')


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    smkdirs(OUTPUT_PATH)

    # Get current directory
    cwd = os.getcwd()
    # Get directory where git repo lives
    curfilePath = os.path.relpath(milsed.__file__)
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))
    parentDir = os.path.abspath(os.path.join(curDir, os.pardir))
    # Change to the repo directory
    os.chdir(parentDir)
    # Return to working dir
    os.chdir(cwd)

    # Add version to params
    d = vars(params)

    # Create folder for resutls
    # smkdirs(os.path.join(OUTPUT_PATH, version))
    smkdirs(os.path.join(OUTPUT_PATH, params.modelid))

    print('{}: testing'.format(__doc__))
    print('Model ID: {}'.format(params.modelid))
    print(params)

    test(params.modelname,
         params.modelid,
         params.working,
         params.strong_labels_file,
         params.alpha,
         params.max_samples,
         params.duration,
         params.rate,
         params.batch_size,
         params.epochs,
         params.epoch_size,
         params.validation_size,
         params.early_stopping,
         params.reduce_lr,
         params.seed,
         params.train_streamers,
         params.augment,
         params.augment_drc,
         params.verbose,
         params.train_balanced,
         params.train_strong)
