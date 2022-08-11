#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Gonzalo Diaz
# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import print_function, division
import numpy as np

from PyAstronomy import pyasl
import os 
import os.path
import sys
import yaml 
from munch import Munch
from astropy.table import Table
from astropy.io import fits
from astropy.io import ascii
import astropy.wcs as wcs

import astropy.units as u
from astropy.visualization import quantity_support
quantity_support()

from specutils.manipulation import extract_region
from specutils.io.registers import custom_writer
from specutils import Spectrum1D, SpectralRegion, SpectrumCollection
from specutils.manipulation import FluxConservingResampler
from specutils.manipulation import LinearInterpolatedResampler
from specutils.manipulation import SplineInterpolatedResampler
from specutils.fitting import fit_generic_continuum


def read_GSL_tmplt_reg(filename, wave_array, reg_extraction):
    """Loads a fits file into a Spuntrum1D object
 
    This tool does ....

    Parameters
    ----------
    filename : str
        The name of the fits to read

    Returns
    -------
    auxspec
        A Spectrum1d object of the fits file spectrum
    wcs_file
        Word Coord. System information in the fits
    """
    print('Read Template')
    h_temp = fits.open(filename)
    spec_temp = h_temp[0].data
    aux_lamb_temp = wave_array  
    meta_file = h_temp[0].header
    print('Create Spectrum')
    auxspec = Spectrum1D(
        spectral_axis= aux_lamb_temp * u.AA,
        flux = np.nan_to_num(spec_temp,nan=0) * u.Unit('erg cm-2 s-1 cm-1'),
        meta = meta_file)
    h_temp.close()
    print('Extract region of interest')
    sub_auxspec = extract_region(auxspec, reg_extraction)
    return sub_auxspec

def generic_fits(
        spectrum, file_name, **kwargs):
    """Writes a fits file from a Spectrum1D

    Parameters
    ----------
    spectrum : str
        The ....
    file_name : str
        A fla...
    """
    print('Writing Spectrum')
    flux = spectrum.flux.value
    #inverse_var = spectrum.uncertainty.array
    wavelength = spectrum.spectral_axis.value
    meta = spectrum.meta
    tab = Table([wavelength, flux]) #, names=("wavelength", "flux")) #, meta=meta)
    tab.write(file_name, format="ascii")#, overwrite='True')
    
# =====================================================================
# PROGRAM
# =====================================================================
def template_converter():
    # Reads wavelength file in Angstroms
    print( ' Reading wavelength axis ')
    h_wav = fits.open("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
    wav_array = h_wav[0].data
    region = SpectralRegion(19000*u.AA, 24500*u.AA)
    spec_temp = read_GSL_tmplt_reg(
        "lte10000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte10000-4.00-0.0.PHOENIX-Kband.txt")
    """
    spec_temp = read_GSL_tmplt_reg(
        "lte02500-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte02500-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte03000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte03000-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte04000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte04000-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte04500-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte04500-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte05000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte05000-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte05500-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte05500-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte06000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte06000-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte06500-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte06500-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte07000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte07000-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte08000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte08000-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte09000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte09000-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte10000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte10000-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte11000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte11000-4.00-0.0.PHOENIX-Kband.fits")

    spec_temp = read_GSL_tmplt_reg(
        "lte12000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
        wav_array, region)
    generic_fits(spec_temp, "lte12000-4.00-0.0.PHOENIX-Kband.fits")
    """

if __name__ == "__main__":
    template_converter()
