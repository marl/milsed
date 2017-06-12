#!/usr/bin/env python
'''CREMA utilities'''

import os
import h5py
from librosa.util import find_files


def save_h5(filename, **kwargs):
    '''Save data to an hdf5 file.

    Parameters
    ----------
    filename : str
        Path to the file

    kwargs
        key-value pairs of data

    See Also
    --------
    load_h5
    '''
    with h5py.File(filename, 'w') as hf:
        hf.update(kwargs)


def load_h5(filename):
    '''Load data from an hdf5 file created by `save_h5`.

    Parameters
    ----------
    filename : str
        Path to the hdf5 file

    Returns
    -------
    data : dict
        The key-value data stored in `filename`

    See Also
    --------
    save_h5
    '''
    data = {}

    def collect(k, v):
        if isinstance(v, h5py.Dataset):
            data[k] = v.value

    with h5py.File(filename, mode='r') as hf:
        hf.visititems(collect)

    return data


def base(filename):
    '''Identify a file by its basename:

    /path/to/base.name.ext => base.name

    Parameters
    ----------
    filename : str
        Path to the file

    Returns
    -------
    base : str
        The base name of the file
    '''
    return os.path.splitext(os.path.basename(filename))[0]


def get_ann_audio(directory):
    '''Get a list of annotations and audio files from a directory.

    This also validates that the lengths match and are paired properly.

    Parameters
    ----------
    directory : str
        The directory to search

    Returns
    -------
    pairs : list of tuples (audio_file, annotation_file)
    '''

    audio = find_files(directory)
    annos = find_files(directory, ext=['jams', 'jamz'])

    paired = list(zip(audio, annos))

    if (len(audio) != len(annos) or
       any([base(aud) != base(ann) for aud, ann in paired])):
        raise RuntimeError('Unmatched audio/annotation '
                           'data in {}'.format(directory))

    return paired
