#!/usr/bin/env python
'''DCASE pre-processing (part 2)'''

import argparse
import sys
import os
import pickle

from tqdm import tqdm
from joblib import Parallel, delayed

from jams.util import smkdirs
from librosa.util import find_files

import milsed.utils

OUTPUT_PATH = os.path.expanduser('~/dev/milsed/models/resources')


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

    parser.add_argument('--jobs', dest='n_jobs', type=int,
                        default=1,
                        help='Number of jobs to run in parallel')

    parser.add_argument('--tqdm', dest='tqdm', action='store_const',
                        const=True, default=False)

    parser.add_argument('--overwrite', dest='overwrite', action='store_const',
                        const=True, default=False)

    parser.add_argument('input_path', type=str,
                        help='Path for directory containing (audio, jams)')

    parser.add_argument('output_path', type=str,
                        help='Path to store pump output')

    return parser.parse_args(args)


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

    pumpfile = os.path.join(OUTPUT_PATH, 'pump.pkl')
    with open(pumpfile, 'rb') as fp:
        pump = pickle.load(fp)

    # stream = milsed.utils.get_ann_audio(params.input_path)
    stream = find_files(params.input_path)

    if not params.overwrite:
        missing_stream = []
        for af in stream:
            basename = milsed.utils.base(af)
            pumpfile = os.path.join(params.output_path, basename + '.h5')
            if not os.path.isfile(pumpfile):
                missing_stream.append(af)
        stream = missing_stream

    if params.tqdm:
        stream = tqdm(stream, desc='Converting eval data')

    Parallel(n_jobs=params.n_jobs)(delayed(convert)(aud, None,
                                                    pump,
                                                    params.output_path)
                                   for aud in stream)

