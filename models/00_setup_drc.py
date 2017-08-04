#!/usr/bin/env python
'''DCASE data augmentation'''

import argparse
import sys
import os
import pickle

from tqdm import tqdm
from joblib import Parallel, delayed

from jams.util import smkdirs
import muda

import milsed.utils

OUTPUT_PATH = os.path.expanduser('~/dev/milsed/models/resources')


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--presets', dest='presets', nargs='+', type='str',
        default=['radio', 'film standard', 'music standard', 'speech'],
        help='DRC presets to apply')

    parser.add_argument('--audio-ext', dest='audio_ext', type=str,
                        default='ogg',
                        help='Output format for audio (ogg, flac, etc)')

    parser.add_argument('--jams-ext', dest='jams_ext', type=str,
                        default='jams',
                        help='Output format for annotations (jams, jamz)')

    parser.add_argument('--jobs', dest='n_jobs', type=int,
                        default=1,
                        help='Number of jobs to run in parallel')

    parser.add_argument('--start_index', dest='start_index', type=int,
                        default=0,
                        help='Start index for numbering output files')

    parser.add_argument('input_path', type=str,
                        help='Path for input (audio, jams) pairs')

    parser.add_argument('output_path', type=str,
                        help='Path to store augmented data')

    return parser.parse_args(args)


def augment(afile, jfile, deformer, outpath, audio_ext, jams_ext,
            start_index, sr=44100):
    '''Run the data through muda'''
    jam = muda.load_jam_audio(jfile, afile, sr=sr)

    base = milsed.utils.base(afile)

    outfile = os.path.join(outpath, base)
    for i, jam_out in enumerate(deformer.transform(jam)):
        muda.save('{}.{}.{}'.format(outfile, start_index+i, audio_ext),
                  '{}.{}.{}'.format(outfile, start_index+i, jams_ext),
                  jam_out, strict=False)


def make_muda(presets):
    '''Construct a MUDA pitch shifter'''

    drc = muda.deformers.DynamicRangeCompression(preset=presets)

    smkdirs(OUTPUT_PATH)
    with open(os.path.join(OUTPUT_PATH, 'muda_drc.pkl'), 'wb') as fd:
        pickle.dump(drc, fd)

    return drc


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    smkdirs(OUTPUT_PATH)
    smkdirs(params.output_path)

    print('{}: setup'.format(__doc__))
    print(params)

    # Build the deformer
    deformer = make_muda(params.presets)

    # Get the file list
    stream = tqdm(milsed.utils.get_ann_audio(params.input_path),
                  desc='Augmenting training data')

    # Launch the job
    Parallel(n_jobs=params.n_jobs)(delayed(augment)(aud, ann, deformer,
                                                    params.output_path,
                                                    params.audio_ext,
                                                    params.jams_ext,
                                                    params.start_index)
                                   for aud, ann in stream)
