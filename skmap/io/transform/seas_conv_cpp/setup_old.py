import os
from setuptools import setup, Extension

module = Extension('old_sc', sources=['old_sc.cpp'], libraries=['fftw3_threads', 'fftw3', 'm'])

setup(
    name = 'old_sc',
    version = "0.1",
    author = "Davide Consoli",
    author_email = "davide.consoli@opengeohub.org",
    description = ("A funtion to perfrom fast seasonal and renormalized convolution for earth-observation time series"),
    license = "BSD",
    ext_modules = [module],
    )