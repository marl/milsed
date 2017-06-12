from setuptools import setup, find_packages

import imp

version = imp.load_source('milsed.version', 'milsed/version.py')

setup(
    name='milsed',
    version=version.version,
    description="Multiple instance learning for sound event detection",
    author='',
    url='http://github.com/justinsalamon/milsed',
    download_url='http://github.com/justinsalamon/milsed/releases',
    packages=find_packages(),
    package_data={'': ['models/*/*.pkl',
                       'models/*/*.h5',
                       'models/*/*.json',
                       'models/*/*.txt']},
    long_description="Convolutional-recurrent estimators for music analysis",
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='audio music learning',
    license='ISC',
    install_requires=['six',
                      'librosa>=0.5.1',
                      'jams>=0.2.3',
                      'scikit-learn>=0.18',
                      'keras>=2.0',
                      'tensorflow>=1.0',
                      'h5py>=2.7'],
    extras_require={
        'docs': ['numpydoc'],
        'tests': ['pytest', 'pytest-cov'],
        'training': ['pescador>=1.0']
    }
)
