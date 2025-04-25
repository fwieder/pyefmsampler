#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 10:46:04 2025

@author: frederik
"""

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pyefmsampler",
    version="1.0.0",
    author="Frederik Wieder",
    description=("pyEFMsampler"),
    url="https://github.com/fwieder/pyefmsampler",
    packages=["pyefmsampler"],
    long_description=read("Readme.md"),
    install_requires=[
        "numpy",
        "efmtool",
        "scipy",
        "cobra",
        "tqdm",
        "umap-learn"
        "matplotlib"
    ],
)