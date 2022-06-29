#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NISCAL Project (https://github.com/cgonzalodiaz/niscal)
# Copyright (c) 2022,Gonzalo Díaz
# License: MIT
#   Full Text: https://github.com/cgonzalodiaz/niscal/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute and install NISCAL
"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

from setuptools import setup

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = [
    "numpy",
    "PyAstonomy",
    "astropy>=4.2",
    "specutils",
    "matplotlib",
    "yaml",
    "munch",
    "os",
    "sys",
]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()

with open(PATH / "niscal" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


DESCRIPTION = "Telluric and Flux calibrate NIR spectra"


# =============================================================================
# FUNCTIONS
# =============================================================================


def do_setup():
    setup(
        name="niscal",
        version=VERSION,
        description=DESCRIPTION,
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author=["Gonzalo Díaz"],
        author_email="cgonzadiaz@gmail.com",
        url="https://github.com/cgonzalodiaz/niscal",
        py_modules=["ez_setup"],
        packages=[
            "niscal",
        ],
        license="MIT",
        keywords=["niscal", "telluric", "flux", "spectra", "NIR"],
        classifiers=[
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.9",
        ],
        install_requires=REQUIREMENTS,
    )


if __name__ == "__main__":
    do_setup()
