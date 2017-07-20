#!/usr/bin/env python
'''DCASE pre-processing'''

import argparse
import sys
import os
import pickle

from tqdm import tqdm
from joblib import Parallel, delayed

from jams.util import smkdirs

import pumpp

import milsed.utils

OUTPUT_PATH = 'resources'


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

    parser.add_argument('--sample-rate',
                        dest='sr', type=float, default=44100.,
                        help='Sampling rate for audio analysis')

    parser.add_argument('--hop-length', dest='hop_length', type=int,
                        default=1024,
                        help='Hop length for audio analysis')

    parser.add_argument('--nfft', dest='n_fft', type=int,
                        default=2048,
                        help='Number of FFT bins')

    parser.add_argument('--nmels', dest='n_mels', type=int,
                        default=128,
                        help='Number of Mel bins')

    parser.add_argument('--jobs', dest='n_jobs', type=int,
                        default=1,
                        help='Number of jobs to run in parallel')

    parser.add_argument('--augmentation-path', dest='augment_path', type=str,
                        default=None,
                        help='Path for augmented data (optional)')

    parser.add_argument('input_path', type=str,
                        help='Path for directory containing (audio, jams)')

    parser.add_argument('output_path', type=str,
                        help='Path to store pump output')

    return parser.parse_args(args)


def make_pump(sr, hop_length, n_fft, n_mels):

    p_feature = pumpp.feature.Mel(name='mel',
                                  sr=sr,
                                  hop_length=hop_length,
                                  n_fft=n_fft,
                                  n_mels=n_mels,
                                  log=True,
                                  conv='tf')

    p_tag = pumpp.task.StaticLabelTransformer(name='static',
                                              namespace='tag_open',
                                              labels=DCASE_CLASSES)

    pump = pumpp.Pump(p_feature, p_tag)

    # Save the pump
    with open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'wb') as fd:
        pickle.dump(pump, fd)

    return pump


def convert(aud, jam, pump, outdir):
    data = pump.transform(aud, jam)
    fname = os.path.extsep.join([os.path.join(outdir, milsed.utils.base(aud)),
                                'h5'])
    milsed.utils.save_h5(fname, **data)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    smkdirs(OUTPUT_PATH)
    smkdirs(params.output_path)

    print('{}: pre-processing'.format(__doc__))
    print(params)
    pump = make_pump(params.sr,
                     params.hop_length,
                     params.n_fft,
                     params.n_mels)

    stream = tqdm(milsed.utils.get_ann_audio(params.input_path),
                  desc='Converting training data')
    Parallel(n_jobs=params.n_jobs)(delayed(convert)(aud, ann,
                                                    pump,
                                                    params.output_path)
                                   for aud, ann in stream)

    if params.augment_path:
        stream = tqdm(milsed.utils.get_ann_audio(params.augment_path),
                      desc='Converting augmented data')
        Parallel(n_jobs=params.n_jobs)(delayed(convert)(aud, ann,
                                                        pump,
                                                        params.output_path)
                                       for aud, ann in stream)
