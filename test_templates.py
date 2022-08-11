#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Gonzalo Diaz
# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import print_function, division
import numpy as np

from astropy.table import Table
from astropy.io import fits
from astropy.io import ascii
import astropy.wcs as wcs

import astropy.units as u
from astropy.visualization import quantity_support
quantity_support()

    h_temp = fits.open(filename)

    #w1_temp = h_temp[0].header['CRVAL1']
    #disp_temp = 0.6 #h_temp[0].header['CDELT1']
