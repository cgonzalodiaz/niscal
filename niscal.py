#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Gonzalo Diaz

A first draft of NISCAL
#-----------------------------------------
#Telluric correction to the science spectrum              #
#---------------------------------------------
# Note that IF the telluric spectrum is not corrected to remove intrinsic stellar
# spectrum it will leave false emission features in the final science spectrum.
# This code will remove the intrinsic stellar spectrum using stellar templates.
# Briefly, the code will:
# 1 - Load a telluric spectra and a science spectra.
# 2 - Load the stellar template spectra to use for comparison and removal of 
#     intrisec stellar features.
# 3 - If radial velocity of the telluric star is not provided, 
#     get radial velocity of the telluric star from the templates
#     Select the wavelength range for the cross correlation
#     Model for the continuum can be changed too.
# 4 - Shift template in radial velocity
# 5 - Resample the template and Copy the region of interest, which can be edited. 
# 6 - Calculate the ratio and the difference of the telluric spectrum 
#     and the velocity-corrected templates.
#
# 7 - Evaluates the quality of the correction: 
#     How good is the fit? minimise residuals
#
# 8 - Applies a telluric correction to the science using the ratio telluric/template.
#
#----------------------------------------------
# Flux calibration of the science spectrum              #
#----------------------------------------------
# 9 - If the 2MASS magnitude is know, use photflux_cal and calibrate in flux
#     based on the 2MASS magnitude of the source
#
# 10 - If the 2MASS magnitude of the telluric star is known, calculte the 
#     flux calibration function to convert ADUs/s in flux using the telluric 
#      observation.
#      
"""
from __future__ import print_function, division
import numpy as np
from PyAstronomy import pyasl
import os 
import os.path
import sys
import yaml 
from munch import Munch

import matplotlib.pyplot as plt
import matplotlib.axes as ax
import matplotlib.gridspec as gridspec
from matplotlib import rc
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.interpolate import interp1d

from astropy.nddata import StdDevUncertainty
from astropy.table import Table
from astropy.io import fits
from astropy.io import ascii
import astropy.wcs as wcs
from astropy.utils.data import get_pkg_data_filename
import astropy.units as u
from astropy.visualization import quantity_support
quantity_support()
from astropy.modeling import models, fitting
from astropy.convolution import Box2DKernel
from astropy.convolution import Gaussian2DKernel
from astropy.modeling.models import Lorentz1D
from astropy.modeling.functional_models import Box2D
from astropy.convolution import convolve_fft
from astropy.modeling.polynomial import Chebyshev1D, Legendre1D
from astropy.modeling.fitting import LevMarLSQFitter
from specutils.manipulation import extract_region
from specutils.io.registers import custom_writer
from specutils import Spectrum1D,SpectralRegion,SpectrumCollection
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
from specutils.fitting import fit_generic_continuum


def generic_fits(spectrum, file_name, **kwargs):
    flux = spectrum.flux.value
    inverse_var = spectrum.uncertainty.array
    wavelength = spectrum.spectral_axis.value
    meta = spectrum.meta
    tab = Table([wavelength, flux, inverse_var], names=("wavelength", "flux", "invvar"), meta=meta)
    tab.write(file_name, format="fits",overwrite='True')
    

def plot_continuum_fit_Ks(ni,ds_telu,de_telu,telu_continuum,ds_temp,temp_continuum,av1 ,min_wav, max_wav,plot_dirRV):
    #av1_spectrum = extract_region(ds_telu, av1[0])
    #av2_spectrum = extract_region(ds_telu, av1[1])
    #av3_spectrum = extract_region(ds_telu, av1[2])
    #av1_temp = extract_region(ds_temp, av1[0])
    #av2_temp = extract_region(ds_temp, av1[1])
    #av3_temp = extract_region(ds_temp, av1[2])
    # PLOT SPECTRA
    ######################################
    fn = plt.gcf()
    fn.set_size_inches(12,8)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gsn = gridspec.GridSpec(2, 1)
    gsn.update(left=0.1, right=0.95,bottom=0.15,top=0.95,hspace=0.3,wspace=0.3)
    ax21 = plt.subplot(gsn[0:1,0:1])
    kk = fn.sca(ax21)      
    
    ax21.step(ds_telu.spectral_axis, telu_continuum,color='g', linestyle='--',label='Telluric continuum') 
    ax21.step(ds_telu.spectral_axis, ds_telu.flux,color='b', linestyle='-',label='Telluric spectrum') 
    plt.ylabel("Arbitrary scale")
    # techo y piso del error mas el valor:
    #roof0 = ds_telu.flux + de_telu
    #floor0 = ds_telu.flux - de_telu
    #plt.fill_between(ds_telu.spectral_axis.value, floor0, roof0, facecolor='0.75',edgecolor='0.5')
    for jj in np.arange(len(av1)):    
        av1_spectrum = extract_region(ds_telu, av1[jj])
        #av1_temp = extract_region(ds_temp, av1[jj])
        ax21.step(av1_spectrum.spectral_axis, av1_spectrum.flux,color='r', linestyle=':')#,label='Excluded region')
    plt.legend()
    #ax21.step(av2_spectrum.spectral_axis, av2_spectrum.flux,color='r', linestyle='-') 
    #ax21.step(av3_spectrum.spectral_axis, av3_spectrum.flux,color='r', linestyle='-') 
    #ax21.axis([min_wav, max_wav, 5000, 33000])         
    ax22 = plt.subplot(gsn[1:2,0:1])
    kk = fn.sca(ax22)  
    # techo y piso del error mas el valor:
    #roof1=(de_temp.flux+ds_temp.flux)/ds_temp.flux[1]
    #floor1=(ds_temp.flux-de_temp.flux)/ds_temp.flux[1]

    ax22.step(ds_temp.spectral_axis, ds_temp.flux/ds_temp.flux[1], color='b', linestyle='-',label='Template spectrum') 
    #plt.fill_between(de_temp.spectral_axis.value, floor1, roof1, facecolor='0.75',edgecolor='0.5')
    ax22.step(ds_temp.spectral_axis, temp_continuum/ds_temp.flux[1],color='g', linestyle='--',label='Template continuum') 
    for jj in np.arange(len(av1)):    
        #av1_spectrum = extract_region(ds_telu, av1[jj])
        av1_temp = extract_region(ds_temp, av1[jj])
        ax22.step(av1_temp.spectral_axis, av1_temp.flux/ds_temp.flux[1],color='r', linestyle=':') #,label='Excluded region') 
    plt.ylabel("Arbitrary scale")
    #ax22.step(av2_temp.spectral_axis, av2_temp.flux/ds_temp.flux[1],color='r', linestyle='-') 
    #ax22.step(av3_temp.spectral_axis, av3_temp.flux/ds_temp.flux[1],color='r', linestyle='-')    
    #ax22.axis([min_wav, max_wav, 0.4, 1.1])         
    #ax22.plot(h_x_Br,h_y_Br+ 0.90,'sb')
    plt.legend()
    fn.savefig(plot_dirRV+'rv_continuum'+str(ni+1)+'.png', dpi=100, bbox_inches='tight')

def plot_norm_spec_Ks(ni,telu_n,telu_e,temp_n,av1,min_wav, max_wav, plot_dirRV):
    av1_spectrum = extract_region(telu_n, av1[0])
    #av2_spectrum = extract_region(telu_n, av2)
    #av3_spectrum = extract_region(telu_n, av3)
    av1_temp = extract_region(temp_n, av1[0])
    #av1_temp_e = extract_region(temp_e, av1)
    #av2_temp = extract_region(temp_n, av2)
    #av3_temp = extract_region(temp_n, av3)
    # PLOT SPECTRA
    ######################################
    fn = plt.gcf()
    fn.set_size_inches(12,8)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gsn = gridspec.GridSpec(2, 1)
    gsn.update(left=0.1, right=0.95,bottom=0.15,top=0.95,hspace=0.3,wspace=0.3)
    ax21 = plt.subplot(gsn[0:1,0:1])
    kk = fn.sca(ax21)  
    ax21.step(telu_n.spectral_axis, telu_n.flux,color='b', linestyle='-',label='Telluric spectrum') 
    #roof1 = telu_n.flux + telu_e
    #floor1 = telu_n.flux - telu_e
    #plt.fill_between(telu_n.spectral_axis.value, floor1, roof1, facecolor='0.75',edgecolor='0.5')
    for jj in np.arange(len(av1)):    
        av1_spectrum = extract_region(telu_n, av1[jj])    
        ax21.step(av1_spectrum.spectral_axis, av1_spectrum.flux,color='r', linestyle='--')#,label='Excluded region')  
    #ax21.step(av2_spectrum.spectral_axis, av2_spectrum.flux,color='r', linestyle='--') 
    #ax21.step(av3_spectrum.spectral_axis, av3_spectrum.flux,color='r', linestyle='--') 
    ax21.axis([min_wav, max_wav, 0.6, 1.2])         
    plt.legend()

    ax22 = plt.subplot(gsn[1:2,0:1])
    kk = fn.sca(ax22)  
    # roof and floor of the error plus signal value:
    #roof1 = temp_n.flux + temp_e.flux
    #floor1 = temp_n.flux - temp_e.flux
    ax22.step(temp_n.spectral_axis, temp_n.flux, color='b', linestyle='-',label='Template spectrum') 
    #plt.fill_between(temp_e.spectral_axis.value, floor1, roof1, facecolor='0.75',edgecolor='0.5')
    ax22.step(telu_n.spectral_axis, telu_n.flux,color='g', linestyle=':',label='Telluric spectrum') 
    for jj in np.arange(len(av1)):    
        av1_temp = extract_region(temp_n, av1[jj])    
        ax22.step(av1_temp.spectral_axis, av1_temp.flux,color='r', linestyle='--') 
    #ax22.step(av2_temp.spectral_axis, av2_temp.flux,color='r', linestyle='--') 
    #ax22.step(av3_temp.spectral_axis, av3_temp.flux,color='r', linestyle='--') 
    ax22.axis([min_wav, max_wav, 0.6, 1.2])         
    #ax22.plot(h_x_Br,h_y_Br+ 0.90,'sb')
    plt.legend()
    fn.savefig(plot_dirRV+'rv_normalized'+str(ni+1)+'.png', dpi=100, bbox_inches='tight')
        
def read_spec(filename):
    h_telu = fits.open(filename)
    wcs_sci = wcs.WCS(h_telu[('sci',1)].header,h_telu)
    w1_telu = h_telu[1].header['CRVAL1']
    disp_telu = h_telu[1].header['CD1_1']
    telu_data = fits.getdata(filename, ext=1)
    npix_telu = telu_data.shape
    spec_telu = h_telu[1].data
    meta_telu = h_telu[0].header
    #  ES VARIANZA
    var_spec_telu =  h_telu[2].data
    stdev_spec_telu = np.sqrt(h_telu[2].data)
    h_telu.close()
    lamb_telu = w1_telu * u.AA  + np.arange(npix_telu[0]) * disp_telu * u.AA 
    auxspec = Spectrum1D(spectral_axis=lamb_telu,
                         flux= spec_telu  * u.Unit('erg cm-2 s-1 AA-1') , 
                         uncertainty = StdDevUncertainty(stdev_spec_telu), 
                         meta = meta_telu)#, wcs=wcs_sci) 
    return auxspec, wcs_sci

def read_template(filename):
    #template_file = get_pkg_data_filename(filename)
    #fits.info(template_file)
    h_temp = fits.open(filename)
    #w1_temp = h_temp[0].header['CRVAL1']
    #disp_temp = 0.6 #h_temp[0].header['CDELT1']
    #temp_data = fits.getdata(filename, ext=0)
    #npix_temp = temp_data.shape
    spec_temp = h_temp[1].data['FLUX']
    aux_lamb_temp = h_temp[1].data['WAVE']* 10.0 
    error_temp = h_temp[1].data['ERR']
    #lamb_temp = h_temp[1].data['WAVE'][0] * 10.0 * u.AA + np.arange(len(aux_lamb_temp))  * disp_temp * u.AA 
    h_temp.close()
    #w1_temp * u.AA  + np.arange(npix_temp[0]) * disp_temp * u.AA 
    auxspec = Spectrum1D(spectral_axis=aux_lamb_temp * u.AA , flux= np.nan_to_num(spec_temp,nan=0)  * u.Unit('erg cm-2 s-1 AA-1'),
                         uncertainty = StdDevUncertainty( np.nan_to_num(error_temp,nan=0))) 
    aux_err_spec = Spectrum1D(spectral_axis=aux_lamb_temp* u.AA , flux= np.nan_to_num(error_temp,nan=0)  * u.Unit('erg cm-2 s-1 AA-1')) 
    return auxspec, aux_err_spec



def  get_hidro_lines(min_wav_rv,max_wav_rv):
    # Read Hidrogen lines
    # options are: Br, Hu , Pa, Hband, Ksband, Pf, All  

    if min_wav_rv > 19800: 
        tab= '/home/gonza/Proyecto_AGN_OCSVM/Codigos/NISCAL/database/hydrogen_transitions_Ks-band.txt'
        list_hydrogen = Table.read(tab, format='ascii')
        hydrogen_s = list_hydrogen['SERIES']
        hydrogen_t = list_hydrogen['TRANSITION']
        hydrogen_w = list_hydrogen['WAVELENGTH']
        x_hy = np.array(list_hydrogen['WAVELENGTH']) * 1e4
        y_hy = np.zeros(len(hydrogen_w))
        hband_hid_aux=x_hy[np.where(x_hy < max_wav_rv)]
        hband_hid=hband_hid_aux[np.where(hband_hid_aux > min_wav_rv)]
        #print("Wavelength of Hydrogen lines avoided: ",hband_hid)
        av1 = SpectralRegion((hband_hid[0] - 180.0) * u.AA, (hband_hid[0] + 180.0)* u.AA)
        avoid_reg=av1
    else:
        if min_wav_rv > 14000: 
            tab= '/home/gonza/Proyecto_AGN_OCSVM/Codigos/NISCAL/database/hydrogen_transitions_H-band.txt'
            list_hydrogen = Table.read(tab, format='ascii')
            hydrogen_s = list_hydrogen['SERIES']
            hydrogen_t = list_hydrogen['TRANSITION']
            hydrogen_w = list_hydrogen['WAVELENGTH']
            x_hy = np.array(list_hydrogen['WAVELENGTH']) * 1e4
            y_hy = np.zeros(len(hydrogen_w))
            hband_hid_aux=x_hy[np.where(x_hy < max_wav_rv)]
            hband_hid=hband_hid_aux[np.where(hband_hid_aux > min_wav_rv)]
            #print("Wavelength of Hydrogen lines avoided: ",hband_hid)
            #for aux_lambda in hband_hid:
            #    av_aux = SpectralRegion((aux_lambda - 180.0) * u.AA, (aux_lambda + 180.0)* u.AA)
            #avoid_reg = 
            if len(hband_hid) == 1:
                av1 = SpectralRegion((hband_hid[0] - 160.0) * u.AA, (hband_hid[0] + 160.0)* u.AA)
                avoid_reg=av1
            if  len(hband_hid) == 2:
                av1 = SpectralRegion((hband_hid[0] - 160.0) * u.AA, (hband_hid[0] + 160.0)* u.AA)
                av2 = SpectralRegion((hband_hid[1] - 160.0) * u.AA, (hband_hid[1] + 140.0)* u.AA)
                avoid_reg=[av1,av2]
            if  len(hband_hid) == 3:
                av1 = SpectralRegion((hband_hid[0] - 160.0) * u.AA, (hband_hid[0] + 160.0)* u.AA)
                av2 = SpectralRegion((hband_hid[1] - 160.0) * u.AA, (hband_hid[1] + 140.0)* u.AA)
                av3 = SpectralRegion((hband_hid[2] - 120.0) * u.AA, (hband_hid[2] + 140.0)* u.AA)
                avoid_reg=[av1,av2,av3]#,av4,av5,av7,av8]
            if  len(hband_hid) == 4:
                av1 = SpectralRegion((hband_hid[0] - 160.0) * u.AA, (hband_hid[0] + 160.0)* u.AA)
                av2 = SpectralRegion((hband_hid[1] - 160.0) * u.AA, (hband_hid[1] + 140.0)* u.AA)
                av3 = SpectralRegion((hband_hid[2] - 120.0) * u.AA, (hband_hid[2] + 140.0)* u.AA)
                av4 = SpectralRegion((hband_hid[3] - 100.0) * u.AA, (hband_hid[3] + 75.0)* u.AA)
                avoid_reg=[av1,av2,av3,av4]
            if len(hband_hid) == 5:
                av1 = SpectralRegion((hband_hid[0] - 160.0) * u.AA, (hband_hid[0] + 160.0)* u.AA)
                av2 = SpectralRegion((hband_hid[1] - 160.0) * u.AA, (hband_hid[1] + 140.0)* u.AA)
                av3 = SpectralRegion((hband_hid[2] - 120.0) * u.AA, (hband_hid[2] + 140.0)* u.AA)
                av4 = SpectralRegion((hband_hid[3] - 100.0) * u.AA, (hband_hid[3] + 75.0)* u.AA)
                av5 = SpectralRegion((hband_hid[4] - 240.0) * u.AA, (hband_hid[4] + 80.0)* u.AA)
                avoid_reg=[av1,av2,av3,av4,av5]
            #av7 = SpectralRegion((hband_hid[6] - 30.0) * u.AA, (hband_hid[6] + 30.0)* u.AA)
            #av8 = SpectralRegion((hband_hid[7] - 70.0) * u.AA, (hband_hid[7] + 20.0)* u.AA)
            #avoid_reg=[av1,av2,av3]#,av4,av5,av7,av8]
        else:
            if min_wav_rv > 10000: 
                tab= '/home/gonza/Proyecto_AGN_OCSVM/Codigos/NISCAL/database/hydrogen_transitions_J-band.txt'
                list_hydrogen = Table.read(tab, format='ascii')
                hydrogen_s = list_hydrogen['SERIES']
                hydrogen_t = list_hydrogen['TRANSITION']
                hydrogen_w = list_hydrogen['WAVELENGTH']
                x_hy = np.array(list_hydrogen['WAVELENGTH']) * 1e4
                y_hy = np.zeros(len(hydrogen_w))
                hband_hid_aux=x_hy[np.where(x_hy < max_wav_rv)]
                hband_hid=hband_hid_aux[np.where(hband_hid_aux > min_wav_rv)]
                #print("Wavelength of Hydrogen lines avoided: ",hband_hid)
                av1 = SpectralRegion((hband_hid[0] - 180.0) * u.AA, (hband_hid[0] + 180.0)* u.AA)
                avoid_reg=av1
    return avoid_reg


def show_region_rv(spe_telu,spe_temp,in_config):
    #read the tellurc spec from ITC
    min_wav_rv = in_config["rvwavmin"], 
    max_wav_rv = in_config["rvwavmax"], 
    norm_wav = in_config["which_wav_scale"]
    
    trans_filter_Ks = DataBase+'2MASS_Ks_RSR.txt'
    trans_filter_H = DataBase+'2MASS_H_RSR.txt'
    trans_filter_J = DataBase+'2MASS_J_RSR.txt'
    atm_Ks_spec = DataBase+'telluric_spec_Ks_band.txt'
    atm_H_spec = DataBase+'telluric_spec_H_band.txt'
    atm_J_spec = DataBase+'telluric_spec_J_band.txt'
    
    f2_filter_Ks = DataBase+'f-2_Ks_G0804_05Sep2008.txt'
    f2_filter_H = DataBase+'f-2_H_G0803_05Sep2008.txt'
    f2_filter_J = DataBase+'f-2_J_G0802_05Sep2008.txt'
    
    if norm_wav > 19900: 
        atm_table = Table.read(atm_Ks_spec, format='ascii')
        #  Load FILTER TRANSMISSION CURVE 
        filter_wav,filter_trans = np.loadtxt(trans_filter_Ks, skiprows=1, unpack=True)
        f2_wav,f2_trans = np.loadtxt(f2_filter_Ks, skiprows=1, unpack=True)

    else:
        if norm_wav > 14000: 
            atm_table = Table.read(atm_H_spec, format='ascii')
            filter_wav,filter_trans = np.loadtxt(trans_filter_H, skiprows=1, unpack=True)
            f2_wav,f2_trans = np.loadtxt(f2_filter_H, skiprows=1, unpack=True)

        else:
            if norm_wav > 10000: 
                atm_table = Table.read(atm_J_spec, format='ascii')
                filter_wav,filter_trans = np.loadtxt(trans_filter_J, skiprows=1, unpack=True)
                f2_wav,f2_trans = np.loadtxt(f2_filter_J, skiprows=1, unpack=True)


    Atm_backgrn_wav  = np.array(atm_table['WAVELENGTH']) * 10
    Atm_backgrn_flux = np.array(atm_table['COUNTS'])
    Atm_ref = Atm_backgrn_flux[np.where(abs(Atm_backgrn_wav - norm_wav) < 1.4)]

    #print('K  ref  : ',Atm_ref)
    temp_15000 = spe_temp.flux.value[np.where(abs(spe_temp.spectral_axis.value -norm_wav) < 1.3)]
    telu_15000 = spe_telu.flux.value[np.where(abs(spe_telu.spectral_axis.value -norm_wav) < 1.4)]
    print('temp_ref : ',temp_15000)
    print('telu_ref : ',telu_15000)
    scale_factor_temp = temp_15000 /  telu_15000 * 0.75
    scale_factor_atm = Atm_ref / telu_15000 * 1.8  

    # PLOT SPECTRA
    ######################################
    fn = plt.gcf()
    fn.set_size_inches(12,8)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gsn = gridspec.GridSpec(1, 1)
    gsn.update(left=0.1, right=0.95,bottom=0.15,top=0.95)
    ax21 = plt.subplot(gsn[0:1,0:1])
    kk = fn.sca(ax21)     
    ax21.step(spe_temp.spectral_axis, spe_temp.flux.value / scale_factor_temp ,color='g', linestyle='-',label='Template spectrum (input)') 
    reg_tel = SpectralRegion(min_wav_rv * u.AA, max_wav_rv * u.AA)
    av1_spectrum = extract_region(spe_temp, reg_tel)
    
    techo1=av1_spectrum.flux.value
    floor1=techo1 * 0.0
    plt.fill_between(av1_spectrum.spectral_axis.value, floor1, techo1/scale_factor_temp, facecolor='0.95',edgecolor='0.75')
    
    ax21.step(spe_telu.spectral_axis, spe_telu.flux.value, color='k', linestyle='-',label='Telluric star (input)') 
    av2_spectrum = extract_region(spe_telu, reg_tel)
    #techo2=av2_spectrum.flux.value
    #floor2=techo2 * 0.0
    #plt.fill_between(av2_spectrum.spectral_axis.value, floor2, techo2, facecolor='0.90',edgecolor='0.75')

    ax21.plot(Atm_backgrn_wav, Atm_backgrn_flux / scale_factor_atm, color='r', linestyle='-',label='Atmospheric transmission (example)')
    ax21.axvline( in_config["minwav"] , color='b', linestyle=':')   
    ax21.axvline( in_config["maxwav"], color='b', linestyle=':')   
    ax21.axvline(min_wav_rv, color='g', linestyle=':')   
    ax21.axvline(max_wav_rv, color='g', linestyle=':',label='Wavelength range for RV')   
    ax21.axvline(norm_wav, color='b', linestyle='--',label='Scaling wavelength' )  
    
    ax21.plot(filter_wav*1e4,filter_trans*telu_15000,'m--',label='2MASS Relative Spectral Response')
    ax21.plot(f2_wav*1e4,f2_trans*telu_15000,'b-',label='F2 Relative Spectral Response')
   
    plt.title(" Wavelength for RV estimate")
    plt.ylabel("Arbitrary scale")
    plt.legend()
    fn.savefig('region_rv.png', dpi=100, bbox_inches='tight')

def get_rv_one(s_telu,w_telu,spec_temp,min_wav_rv,max_wav_rv,which_temp):
    # 2- Get radial velocity of telluric star from the template
    # FOR ALL THE TEMPLATES
    # PRINT A TABLE
    mod_rv=Legendre1D(4)    
    delta_wav = w_telu.wcs.cd[0,0]
    delta_wav = 2.6
    new_disp_grid = np.arange(min_wav_rv, max_wav_rv, delta_wav) * u.AA
    fluxcon = FluxConservingResampler()
    ds_temp = fluxcon(spec_temp, new_disp_grid) 
    de_temp = np.sqrt(1.00/ds_temp.uncertainty.array) #fluxcon(error_temp, new_disp_grid)
    
    ds_telu = fluxcon(s_telu, new_disp_grid)     
    de_telu = np.sqrt(1.00/ds_telu.uncertainty.array) # fluxcon(er_telu, new_disp_grid)     
    #####################################################################
    # Fit the continuum of the telluric and the template
    # Select the absorption lines to avoid
    # this will depend on the band. 
    # Needs improvement for all bands
    avoid_lines = get_hidro_lines(min_wav_rv,max_wav_rv)
    telu_fit = fit_generic_continuum(ds_telu, model=mod_rv,
                                     exclude_regions=avoid_lines)#,av2,av3])#,av4,av5,av7,av8])
    ######################################
    if os.path.isdir('RV_plots'):
        plot_dirRV = "RV_plots/"
    else:
        plot_dirRV = "RV_plots/"
        os.mkdir("RV_plots") 
    #####################################################################
    telu_continuum = telu_fit(new_disp_grid)
    telu_norm = ds_telu / telu_continuum    
    e_telu_norm = de_telu / telu_continuum    
    temp_fit = fit_generic_continuum(ds_temp, model=mod_rv, 
                                     exclude_regions=avoid_lines)#,av2,av3])#,av4,av5,av7,av8])
    temp_continuum = temp_fit(new_disp_grid)
    plot_continuum_fit_Ks(which_temp,ds_telu,de_telu,telu_continuum,ds_temp,temp_continuum,avoid_lines, min_wav_rv, max_wav_rv, plot_dirRV)
    
    # 3- Normalize the telluric and the template
    temp_norm = ds_temp / temp_continuum
    e_temp_norm = de_temp / temp_continuum
        
    plot_norm_spec_Ks(which_temp,telu_norm, e_telu_norm, temp_norm,avoid_lines,min_wav_rv, max_wav_rv, plot_dirRV)#,av2,av3,min_wav_rv, max_wav_rv)
        
    # 4- Measure velocity difference
    # Carry out the cross-correlation.
    # The RV-range is -130 - +130 km/s in steps of 1 km/s.
    # The first and last 20 points of the data are skipped.
    dw = telu_norm.spectral_axis
    df = telu_norm.flux
    tw = temp_norm.spectral_axis
    tf = temp_norm.flux
    te = e_temp_norm #.flux
    rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, -500., 500., 2, mode='doppler', skipedge=20)
    # Find the index of maximum cross-correlation function
    maxind = np.argmax(cc)
    #print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
    #if rv[maxind] > 0.0:
    #  print("  A red-shift with respect to the template")
    #else:
    #  print("  A blue-shift with respect to the template")
    ######################################
    # PLOT SPECTRA
    ######################################
    frv = plt.gcf()
    frv.set_size_inches(12,8)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gsn = gridspec.GridSpec(2, 1)
    gsn.update(left=0.1, right=0.95,bottom=0.15,top=0.95,hspace=0.3,wspace=0.3)
    arv1 = plt.subplot(gsn[0:1,0:1])
    arv1.plot(rv, cc, 'b-', label= 'Telluric - Template')
    plt.title(" Cross-correlation function ")
    plt.legend()
    arv1.plot(rv[maxind], cc[maxind], 'ro')
    arv1.set_xlabel("Radial Velocity (km/s)")
    arv2 = plt.subplot(gsn[1:2,0:1])
    #arv1.set_yscale("log")
    rv_out = rv[maxind] # * u.km / u.s        
    # 5- Apply RV correction and plot to check
    # "firstlast" as edge handling method.
    nflux1, wlprime1 = pyasl.dopplerShift(tw, tf,rv[maxind], edgeHandling="firstlast")
    nerror1, wlprime1 = pyasl.dopplerShift(tw, te,rv[maxind], edgeHandling="firstlast")
    # Plot the outcome
    
    arv2.plot(dw, df, 'b-', label= 'Telluric spectrum')
    arv2.plot(tw, nflux1, 'r:',label='Shifted template')
    arv2.axis([min_wav_rv, max_wav_rv, 0.6, 1.2])         
    plt.title("Normalized Spectra")    
    techo1 = nflux1 + nerror1
    floor1 = nflux1 - nerror1
    plt.fill_between(tw, floor1, techo1, facecolor='0.75',edgecolor='0.5')
    plt.legend()
    #plt.plot(tw, nflux2, 'g.-')
    frv.savefig(plot_dirRV+'rv_corrected'+str(which_temp+1)+'.png', dpi=100, bbox_inches='tight')
    return rv_out

def get_rv(s_telu,w_telu,s_temp,e_temp, min_wav_rv,max_wav_rv,n_templ):
    # 2- Get radial velocity of telluric star from the template
    # FOR ALL THE TEMPLATES
    # PRINT A TABLE
    mod_rv=Legendre1D(4)    
    
    delta_wav = w_telu.wcs.cd[0,0]
    delta_wav = 2.6
    new_disp_grid = np.arange(min_wav_rv, max_wav_rv, delta_wav) * u.AA
    fluxcon = FluxConservingResampler()
    ds_temp = fluxcon(s_temp, new_disp_grid) 
    de_temp = fluxcon(e_temp, new_disp_grid)

    ds_telu = fluxcon(s_telu, new_disp_grid)     
    de_telu = np.sqrt(1.00/ds_telu.uncertainty.array) #fluxcon(er_telu, new_disp_grid)     
    #####################################################################
    # Fit the continuum of the telluric and the template
    # Select the absorption lines to avoid
    # this will depend on the band.
    # Needs improvement for all bands
    avoid_lines = get_hidro_lines(min_wav_rv,max_wav_rv)
    telu_fit = fit_generic_continuum(ds_telu, model=mod_rv,
                                     exclude_regions=avoid_lines)#,av2,av3])#,av4,av5,av7,av8])
    ######################################
    if os.path.isdir('RV_plots'):
        plot_dirRV = "RV_plots/"
    else:
        plot_dirRV = "RV_plots/"
        os.mkdir("RV_plots") 
    #####################################################################
    telu_continuum = telu_fit(new_disp_grid)
    telu_norm = ds_telu / telu_continuum    
    e_telu_norm = de_telu / telu_continuum    
    rv_out = np.zeros(n_templ)
    print('To Avoid: ',avoid_lines )

    for i in np.arange(len(rv_out)):
        #print('On the loop.....',s_temp[i])
        temp_fit = fit_generic_continuum(ds_temp[i], model=mod_rv,
                                     exclude_regions=avoid_lines)
        temp_continuum = temp_fit(new_disp_grid)
        plot_continuum_fit_Ks(i,ds_telu,de_telu,telu_continuum,ds_temp[i],temp_continuum,avoid_lines, min_wav_rv, max_wav_rv, plot_dirRV)
        
        # 3- Normalize the telluric and the template
        temp_norm = ds_temp[i] / temp_continuum
        e_temp_norm = de_temp[i] / temp_continuum
        
        plot_norm_spec_Ks(i,telu_norm, e_telu_norm, temp_norm,avoid_lines,min_wav_rv, max_wav_rv, plot_dirRV)#,av2,av3,min_wav_rv, max_wav_rv)
        
        # 4- Measure velocity difference
        # Carry out the cross-correlation.
        # The RV-range is -130 - +130 km/s in steps of 1 km/s.
        # The first and last 20 points of the data are skipped.
        dw = telu_norm.spectral_axis
        df = telu_norm.flux
        tw = temp_norm.spectral_axis
        tf = temp_norm.flux
        te = e_temp_norm.flux
        rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, -500., 500., 2, mode='doppler', skipedge=20)
        # Find the index of maximum cross-correlation function
        maxind = np.argmax(cc)
        #print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
        #if rv[maxind] > 0.0:
        #  print("  A red-shift with respect to the template")
        #else:
        #  print("  A blue-shift with respect to the template"
        ######################################
        # PLOT SPECTRA
        ######################################
        frv = plt.gcf()
        frv.set_size_inches(12,8)
        rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
        rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
        rc('text', usetex=True)
        gsn = gridspec.GridSpec(2, 1)
        gsn.update(left=0.1, right=0.95,bottom=0.15,top=0.95,hspace=0.3,wspace=0.3)
        arv1 = plt.subplot(gsn[0:1,0:1])

        arv1.plot(rv, cc, 'b-', label= 'Telluric - Template')
        plt.title(" Cross-correlation function ")
        plt.legend()
        arv1.plot(rv[maxind], cc[maxind], 'ro')
        arv1.set_xlabel("Radial Velocity (km/s)")
        arv2 = plt.subplot(gsn[1:2,0:1])
        #arv1.set_yscale("log")
        #plt.show()
        rv_out[i] = rv[maxind] # * u.km / u.s        
        # 5- Apply RV correction and plot to check
        # "firstlast" as edge handling method.
        nflux1, wlprime1 = pyasl.dopplerShift(tw, tf,rv[maxind], edgeHandling="firstlast")
        nerror1, wlprime1 = pyasl.dopplerShift(tw, te,rv[maxind], edgeHandling="firstlast")
       # Plot the outcome
        arv2.plot(dw, df, 'b-', label= 'Telluric spectrum')
        arv2.plot(tw, nflux1, 'r:',label='Shifted template')
        arv2.axis([min_wav_rv, max_wav_rv, 0.6, 1.2])         
        plt.title("Normalized Spectra")    
        techo1 = nflux1 + nerror1
        floor1 = nflux1 - nerror1
        plt.fill_between(tw, floor1, techo1, facecolor='0.75',edgecolor='0.5')
        plt.legend()
        #plt.plot(tw, nflux2, 'g.-')
        frv.savefig(plot_dirRV+'rv_corrected'+str(i+1)+'.png', dpi=100, bbox_inches='tight')
    mean_rv= np.mean(rv_out)
    print("Radial Velocity respect to the templates: ",rv_out)
    print("Mean Radial velocity shift:           ",mean_rv)
    return rv_out

def get_quality(diff_spec,min_wav_q,max_wav_q):
    # first number: residuals of the hidrogen line
    # 1- extract the region 
    reg_temp = SpectralRegion(min_wav_q * u.AA, max_wav_q * u.AA)
    av1_spectrum = extract_region(diff_spec, reg_temp)
    #print(av1_spectrum)
    #aux_output[3].spectral_axis, aux_output[3].flux 
    # 2- get RMS, min, max, skewnes, minwav, and maxwav. 
    min_temp = min(av1_spectrum.flux.value)
    max_temp = max(av1_spectrum.flux.value)
    std_temp = np.std(av1_spectrum.flux.value)
    skew_temp = skew(av1_spectrum.flux.value)
    krt_temp = kurtosis(av1_spectrum.flux.value)
    # Minimo, Maximo, Standard deviation, skewness, kurtosis
    quality_stats = np.array([min_temp, max_temp, std_temp, skew_temp , krt_temp])
    
    return quality_stats
  
def get_tell_cor(vshift, telluric, w_telu, template, in_config, wi):
    minwav = in_config["minwav"]
    maxwav = in_config["maxwav"] 
    tempname = in_config["template_list"][wi]["name"]
    scale_wav = in_config["which_wav_scale"]
    min_wav_q = in_config["min_wav_q"]
    max_wav_q = in_config["max_wav_q"]
    
    # Shift template in radial velocity      
    rv_s_temp_flux_1, rv_s_temp_w = pyasl.dopplerShift(template.spectral_axis, template.flux, vshift, edgeHandling="firstlast")
    rvs_temp = Spectrum1D(spectral_axis=template.spectral_axis, flux=rv_s_temp_flux_1* u.Unit('erg cm-2 s-1 AA-1')) 
    ######################################
    #  Resample template and  telluric star's spectrum
    # Copy the region of interest from the telluric star and the templates.
    #delt_wav_tel = w_telu.wcs.cd[0,0]
    new_disp_grid = np.arange(minwav, maxwav, 2.6) * u.AA
    
    fluxcon = FluxConservingResampler()
    ds_temp = fluxcon(rvs_temp, new_disp_grid) 
    ds_telu = fluxcon(telluric, new_disp_grid) 
    ########
    temp_scale_flux = ds_temp.flux.value[np.where(abs(ds_temp.spectral_axis.value - scale_wav) < 1.35)]
    telu_scale_flux = ds_telu.flux.value[np.where(abs(ds_telu.spectral_axis.value - scale_wav) < 1.35)]
    print('template flux scale : ',temp_scale_flux)
    print('telluric flux scale : ',telu_scale_flux)
    scale_factor_temp =  telu_scale_flux / temp_scale_flux 
    
    ######################################
    # Calculate differences and corrections
    ######################################
    s_difference = ds_telu - ds_temp * scale_factor_temp 
    ######################################
    s_ratio = ds_telu / (ds_temp * scale_factor_temp) 
    ######################################
    # update WCS
    #w_telu.wcs.crval[0] = minwav
    #my_wcs = wcs.WCS(header={'CDELT1': delt_wav_tel, 'CRVAL1': minwav, 'CUNIT1': 'Angstrom', 'CTYPE1': 'LINEAR', 'CRPIX1': 1.0})
    
    #, wcs=my_wcs) 
    # Calculate quality index
    q_stats = get_quality(s_difference,min_wav_q,max_wav_q)
    #print ( 'qmin  :  ', q_stats[0])
    #print( 'qmax  :  ',q_stats[1])
    #print( 'qstd  :  ',q_stats[2])
    #print( 'qskew  :  ',q_stats[3])
    #print( 'qkurt  :  ',q_stats[4])
        
    ######################################
    # Record the informatino of the quality of telluric correction
    ######################################
    if os.path.isdir('TC_plots'):
        plot_dir = "TC_plots/"
    else:
        plot_dir = "TC_plots/"
        os.mkdir("TC_plots")
    ######################################
    # PLOT SPECTRA
    ######################################
    fco = plt.gcf()
    fco.set_size_inches(12,12)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gsc = gridspec.GridSpec(3, 1)
    gsc.update(left=0.1, right=0.95,bottom=0.12,top=0.95,hspace=0.3,wspace=0.3)
    axc1 = plt.subplot(gsc[0:1,0:1])
    axc2 = plt.subplot(gsc[1:2,0:1])
    axc3 = plt.subplot(gsc[2:3,0:1])
    kk = fco.sca(axc1)     
    axc1.step(ds_telu.spectral_axis, ds_telu.flux.value ,color='k', linestyle='-',label='Telluric Star') 
    axc1.step(ds_temp.spectral_axis, ds_temp.flux.value * scale_factor_temp  ,color='g', linestyle='-',label='Template Star '+tempname) 
    axc1.axvline(scale_wav, color='g', linestyle=':',label='Scaling Wavelength')
    axc1.axvline(min_wav_q, color='r', linestyle='--',label='Quality assesement region')   
    axc1.axvline(max_wav_q, color='r', linestyle='--')
    plt.title(" Telluric correction")
    axc1.set_ylabel("Arbitrary Flux")
    plt.legend()
    kk = fco.sca(axc2)     
    axc2.step(s_difference.spectral_axis, s_difference.flux.value ,color='k', linestyle='-',label='Telluric - Template') 
    #roof2=s_difference.flux + s_error_diff.flux
    #floor2=s_difference.flux - s_error_diff.flux
    #plt.fill_between(s_difference.spectral_axis.value, floor2, roof2 , facecolor='0.75',edgecolor='0.5')
    axc2.set_ylabel("Arbitrary Flux")
    axc2.axvline(scale_wav, color='g', linestyle=':')   
    axc2.axvline(min_wav_q, color='r', linestyle='--')   
    axc2.axvline(max_wav_q, color='r', linestyle='--')   
    axc2.annotate('Std. Dev.:  '+str(np.around(q_stats[2], decimals=1)), xy =(max_wav_q + 100, -19000))
                  #arrowprops = dict(facecolor ='green',shrink = 0.05))
    axc2.annotate('Skewness:  '+str(np.around(q_stats[3], decimals=2)), xy =(max_wav_q + 100, -22000)) 
                  #xytext =(max_wav_q, -20000))
    axc2.annotate('Kurtosis:  '+str(np.around(q_stats[4], decimals=2)), xy =(max_wav_q + 100, -25000)) 
                  #xytext =(max_wav_q, -20000)) 
    plt.title("Diference")
    plt.legend()
    
    kk = fco.sca(axc3)     
    axc3.step(s_ratio.spectral_axis, s_ratio.flux.value ,color='k', linestyle='-',label='Telluric / Template') 
   
     
    #roof4= s_error_ratio_plu.flux 
    #floor4= s_error_ratio_min.flux 
    #plt.fill_between(s_ratio.spectral_axis.value, floor4, roof4 , facecolor='0.75',edgecolor='b')

    roof3=s_ratio.flux + np.sqrt(1.00/s_ratio.uncertainty.array) # s_error_ratio_plu.flux 
    floor3= s_ratio.flux - np.sqrt(1.00/s_ratio.uncertainty.array) #s_error_ratio_min.flux 
    plt.fill_between(s_ratio.spectral_axis.value, floor3, roof3 , facecolor='0.75',edgecolor='g')
 
    axc3.set_ylabel("Relative Flux")
    axc3.axvline(scale_wav, color='g', linestyle=':')   
    axc3.axvline(min_wav_q, color='r', linestyle='--')   
    axc3.axvline(max_wav_q, color='r', linestyle='--')   
    plt.title("Ratio")
    plt.legend()
    fco.savefig(plot_dir+'telluric_correction_'+str(wi+1)+'.png', dpi=150, bbox_inches='tight')
    return new_disp_grid, q_stats, ds_telu, s_difference, s_ratio

def apply_tell_cor(s_science, s_telu, tes_rat, which_wav_scale, new_disp_grid):
    fluxcon = FluxConservingResampler()
    ds_science = fluxcon(s_science, new_disp_grid)    
    tcor_sci = ds_science / tes_rat
    tcor_telu = s_telu / tes_rat
    #tcor_sci.meta = s_science.meta

    ######################################
    # PLOT S/N
    ######################################
    fsn = plt.gcf()
    fsn.set_size_inches(12,12)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gsn = gridspec.GridSpec(2, 1)
    gsn.update(left=0.1, right=0.95,bottom=0.12,top=0.95,hspace=0.3,wspace=0.3)
    axsn1 = plt.subplot(gsn[0:1,0:1])
    axsn2 = plt.subplot(gsn[1:2,0:1])
    #axsn3 = plt.subplot(gsn[2:3,0:1])
    kk = fsn.sca(axsn1)     
    
    axsn1.step(tcor_sci.spectral_axis, tcor_sci.flux.value ,color='b', linestyle='-',label='After TC') 
    roof5=tcor_sci.flux.value + np.sqrt(1.00/tcor_sci.uncertainty.array) 
    floor5=tcor_sci.flux.value - np.sqrt(1.00/tcor_sci.uncertainty.array) 
    plt.fill_between(tcor_sci.spectral_axis.value, floor5, roof5 , facecolor='0.75',edgecolor='0.5')

    axsn1.axvline(which_wav_scale, color='g', linestyle=':')   
    plt.title("Telluric corrected Science spectrum")
    axsn1.set_ylabel("Arbitrary Flux")   
    plt.legend()
    
    kk = fsn.sca(axsn2)     
    axsn2.step(ds_science.spectral_axis, (ds_science.flux.value/np.sqrt(1.00/ds_science.uncertainty.array)),color='k', linestyle='-',label='Input Science Spectrum S/N')  
    
    axsn2.step(s_telu.spectral_axis, (s_telu.flux.value/np.sqrt(1.00/s_telu.uncertainty.array)) ,color='r', linestyle='-',label='Telluric Spectrum S/N') 
 
    axsn2.step(tcor_sci.spectral_axis, (tcor_sci.flux.value/ np.sqrt(1.00/tcor_sci.uncertainty.array)) ,color='g', linestyle='-',label='Telluric corrected Science Spectrum S/N')      
    axsn2.axvline(which_wav_scale, color='g', linestyle=':')   
    plt.title("S/N")
    plt.legend()
    fsn.savefig('Tel_S2N.png', dpi=150, bbox_inches='tight')
    ######################################
    # PLOT SPECTRA
    ######################################
    fsc = plt.gcf()
    fsc.set_size_inches(12,12)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gtel = gridspec.GridSpec(3, 1)
    gtel.update(left=0.1, right=0.95,bottom=0.12,top=0.95,hspace=0.3,wspace=0.3)
    axtc1 = plt.subplot(gtel[0:1,0:1])
    axtc2 = plt.subplot(gtel[1:2,0:1])
    axtc3 = plt.subplot(gtel[2:3,0:1])
    kk = fsc.sca(axtc1)     
    axtc1.step(ds_science.spectral_axis, ds_science.flux.value ,color='k', linestyle='-',label='Before TC') 
    
    axtc1.step(tcor_sci.spectral_axis, tcor_sci.flux.value ,color='g', linestyle='-',label='After TC') 
    axtc1.axvline(which_wav_scale, color='g', linestyle=':')   
    plt.title("Science data")
    #plt.ylabel("Arbitrary scale")
    axtc1.set_ylabel("Arbitrary scale")
    plt.legend()
    
    kk = fsc.sca(axtc2)     
    axtc2.step(s_telu.spectral_axis, s_telu.flux.value ,color='k', linestyle='-',label='Before TC') 
    axtc2.step(tcor_telu.spectral_axis, tcor_telu.flux.value ,color='g', linestyle='-',label='After TC') 
    axtc2.axvline(which_wav_scale, color='g', linestyle=':')   
    axtc2.set_ylabel("Arbitrary scale")
    plt.title("Telluric Star")
    plt.legend()
    
    kk = fsc.sca(axtc3)  
    
    axtc3.step(tes_rat.spectral_axis, tes_rat.flux.value ,color='k', linestyle='-',label='Telluric / Template') 
    roof7= tes_rat.flux.value + np.sqrt(1.00/tes_rat.uncertainty.array) 
    floor7=  tes_rat.flux.value - np.sqrt(1.00/tes_rat.uncertainty.array) 
    plt.fill_between(tes_rat.spectral_axis.value, floor7, roof7 , facecolor='0.75',edgecolor='0.5')
    
    #axtc3.step(s_correction.spectral_axis, s_correction[6].flux.value ,color='k', linestyle='-',label='Telluric / Template') 
    #roof3=s_correction[7].flux
    #floor3=s_correction[8].flux
    #plt.fill_between(s_correction.spectral_axis.value, floor3, roof3 , facecolor='0.75',edgecolor='0.5')

    axtc3.axvline(which_wav_scale, color='g', linestyle=':')   
    plt.ylabel("Ratio")
    plt.title("Telluric Correction")

    plt.legend()
    fsc.savefig('Tel_corrected_science.png', dpi=150, bbox_inches='tight')
    return tcor_sci, tcor_telu
 
def load_templates(in_config):
    # text file with a list of templates and spectral type
    delta_wav = 2.6 
    new_disp_grid = np.arange(in_config["minwav"], in_config["maxwav"], delta_wav) * u.AA
    ###### needs to read a list of template fluxes without knowing how big the list is.
    template_fluxes = np.zeros((len(in_config["template_list"]), len(new_disp_grid)))
    template_err = np.zeros((len(in_config["template_list"]), len(new_disp_grid)))
    
    for ii in np.arange(len(in_config["template_list"])):
        n_template = in_config["template_list"][ii]["fits"]
        #s_temp = read_template(n_template)
        s_temp, e_temp = read_template(n_template)
        fluxcon = FluxConservingResampler()
        ds_temp = fluxcon(s_temp, new_disp_grid) 
        de_temp = fluxcon(e_temp, new_disp_grid) 
        template_fluxes[ii] = ds_temp.flux.value #* u.Unit('erg cm-2 s-1 AA-1')
        template_err[ii] = de_temp.flux.value #* u.Unit('erg cm-2 s-1 AA-1')
        
    flux_templates = Spectrum1D(spectral_axis=ds_temp.spectral_axis,flux=template_fluxes * u.Unit('erg cm-2 s-1 AA-1'))
    err_templates = Spectrum1D(spectral_axis=de_temp.spectral_axis,flux=template_err * u.Unit('erg cm-2 s-1 AA-1'))
    
    return flux_templates, err_templates

def mag2flux_2mass(m_in,er_m_in,ref_wav,DataBase):
    # Identify de observed band and load convertion factors
        # For Ks band in 2MAS system
    if (ref_wav > 19970):
        f0_lam = (4.283E-14*u.W /u.cm**2/u.micron) 
        flam_mag = (f0_lam* 10**(-0.4*(m_in))).to(u.erg/u.s/u.cm**2/u.AA)
        flam_mag_p = (f0_lam* 10**(-0.4*(m_in+er_m_in))).to(u.erg/u.s/u.cm**2/u.AA)
        flam_mag_m = (f0_lam* 10**(-0.4*(m_in-er_m_in))).to(u.erg/u.s/u.cm**2/u.AA)
        trans_filter=DataBase+'2MASS_Ks_RSR.txt'

    else:
        if (ref_wav < 19970 and ref_wav > 15000):
            # For H band in 2MAS system
            f0_lam = (1E-14*u.W /u.cm**2/u.micron) 
            flam_mag = (f0_lam* 10**(-0.4*(m_in))).to(u.erg/u.s/u.cm**2/u.AA)
            flam_mag_p = (f0_lam* 10**(-0.4*(m_in+er_m_in))).to(u.erg/u.s/u.cm**2/u.AA)
            flam_mag_m = (f0_lam* 10**(-0.4*(m_in-er_m_in))).to(u.erg/u.s/u.cm**2/u.AA)      
            trans_filter=DataBase+'2MASS_H_RSR.txt'
        else:
            if (ref_wav < 15000):    
                # For J band in 2MAS system
                f0_lam = (4.283E-14*u.W /u.cm**2/u.micron) 
                flam_mag = (f0_lam* 10**(-0.4*(m_in))).to(u.erg/u.s/u.cm**2/u.AA)
                flam_mag_p = (f0_lam* 10**(-0.4*(m_in+er_m_in))).to(u.erg/u.s/u.cm**2/u.AA)
                flam_mag_m = (f0_lam* 10**(-0.4*(m_in-er_m_in))).to(u.erg/u.s/u.cm**2/u.AA)
                trans_filter=DataBase+'2MASS_J_RSR.txt'
    return     flam_mag, flam_mag_p, flam_mag_m, trans_filter
    
def get_this_one(in_config):
    ######################################
    # 1 - Read spectra and their errors
    ######################################
    # file names of the telluric and the science objects
    s_telu, wcs_telu= read_spec(in_config["telluric_file"])    
    s_scie, wcs_scie = read_spec(in_config["science_file"])
    ######################################
    # 2 - Read templates
    #  from a list of template fluxes without knowing how big the list is.
    ######################################
    which_temp = in_config["which_template"] - 1
    n_template = in_config["template_list"][which_temp]["fits"]
    s_temp, e_temp = read_template(n_template)
    delta_wav = 2.6 
    new_disp_grid = np.arange(in_config["minwav"], in_config["maxwav"], delta_wav) * u.AA
    fluxcon = FluxConservingResampler()
    ds_temp = fluxcon(s_temp, new_disp_grid)
    de_temp = fluxcon(e_temp, new_disp_grid)

    #f_temp = load_templates(in_config) 
    if "rvspeed" in in_config:
        rad_vel = in_config['rvspeed']
        print(' Radial velocity provided!' )
    else:
        # Show region for RV calculation  
        print(' Calculating radial velocity!' )
        show_region_rv(s_telu,ds_temp, in_config)
        ######################################
        # 3 - Get radial velocity of telluric star from the templates
        rad_vel = get_rv_one(s_telu,wcs_telu,ds_temp, in_config["rvwavmin"], in_config["rvwavmax"],which_temp)
        
        #rad_vel = get_rv(s_telu,wcs_telu,f_temp, e_temp, in_config["rvwavmin"], in_config["rvwavmax"],which_temp)
        #vshift =np.mean(rad_vel)
    ######################################
    # 4 - Shift template in radial velocity and
    # resample template to match telluric star's spectrum
    # PLOT -  templates, ratios and difference with the telluric
    ######################################
    #------------------------------------
    # IF THE TEMPLATE IS KNOWN OR CHOOSEN USE
    final_disp, quality_stats, spec_telu, spec_dif, spec_rat = get_tell_cor(rad_vel, 
                                                                            s_telu, wcs_telu, 
                                                                            ds_temp, 
                                                                            in_config, which_temp)
    ######################################
    # Apply telluric correction to the science spectrum
    tcor_sci, tcor_telu = apply_tell_cor(s_scie,spec_telu,spec_rat, in_config["which_wav_scale"], final_disp)
    return tcor_sci,tcor_telu, spec_rat, spec_dif

def plot_quality(fflog):
 
    print(' Plotting quality assessment' )
    data = ascii.read(fflog)
    # find minimum Std. Dev
    id_minstdev,=np.where(data["q_stdev"] == min(data["q_stdev"]))

    q_distance = np.sqrt((data["q_stdev"]/min(data["q_stdev"])-1.0)**2 + (data["q_skew"])**2 + (data["q_kurt"])**2)
    id_mindis,=np.where(q_distance == min(q_distance))

    ######################################
    # PLOT Quality Indexes
    ######################################
    fqu = plt.gcf()
    fqu.set_size_inches(12,12)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gtel = gridspec.GridSpec(3, 1)
    gtel.update(left=0.1, right=0.95,bottom=0.12,top=0.95,hspace=0.3,wspace=0.3)
    axqu1 = plt.subplot(gtel[0:1,0:1])
    axqu2 = plt.subplot(gtel[1:2,0:1])
    axqu3 = plt.subplot(gtel[2:3,0:1])
    kk = fqu.sca(axqu1)     
    axqu1.plot(data["Index"],data["q_stdev"] ,color='k',marker='o',markersize=10)
    plt.title("Standar Deviation")
    axqu1.axhline(min(data["q_stdev"]), color='g', linestyle='--')   
    axqu1.axhline(1.1*min(data["q_stdev"]), color='r', linestyle=':')  
    axqu1.plot(data["Index"][id_minstdev],data["q_stdev"][id_minstdev] ,color='r',marker='s',markersize=10)
    axqu1.plot(data["Index"][id_mindis],data["q_stdev"][id_mindis] ,color='b',marker='*',markersize=15)
    plt.xlabel("Template Index number")
    axqu1.set_ylabel("Counts")    
    #axqu1.set_yscale("log")
    kk = fqu.sca(axqu2)     
    axqu2.plot(data["Index"],data["q_skew"] ,color='k',marker='o',markersize=10)
    axqu2.axhline(0, color='g', linestyle='-')   
    axqu2.axhline(0.2, color='r', linestyle=':')   
    axqu2.axhline(-0.2, color='r', linestyle=':')   
    
    axqu2.plot(data["Index"][id_minstdev],data["q_skew"][id_minstdev] ,color='r',marker='s',markersize=10)
    axqu2.plot(data["Index"][id_mindis],data["q_skew"][id_mindis] ,color='b',marker='*',markersize=15)

    axqu2.set_ylabel("Skewness")
    plt.xlabel("Template Index number")    
    kk = fqu.sca(axqu3)  
    axqu3.plot(data["Index"],data["q_kurt"] ,color='k',marker='o',markersize=10)
    axqu3.axhline(0, color='g', linestyle='-')           
    axqu3.plot(data["Index"][id_minstdev],data["q_kurt"][id_minstdev] ,color='r',marker='s',markersize=10)
    axqu3.plot(data["Index"][id_mindis],data["q_kurt"][id_mindis] ,color='b',marker='*',markersize=15)

    plt.xlabel("Template Index number")
    plt.ylabel("Kurtosis")
    fqu.savefig('quality_check_1.png', dpi=150, bbox_inches='tight')
 
    ######################################
    # PLOT large
    ######################################
    fqu2 = plt.gcf()
    fqu2.set_size_inches(12,12)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gqu = gridspec.GridSpec(3, 3)
    gqu.update(left=0.1, right=0.95,bottom=0.12,top=0.95,hspace=0.2,wspace=0.25)
    axqu26 = plt.subplot(gqu[0:1,0:1])    
    axqu23 = plt.subplot(gqu[1:2,0:1])
    axqu24 = plt.subplot(gqu[2:3,0:1]) 
    axqu21 = plt.subplot(gqu[1:2,1:2])
    axqu22 = plt.subplot(gqu[2:3,1:2])  
    axqu25 = plt.subplot(gqu[2:3,2:3])
    
    kk = fqu2.sca(axqu21)     
    axqu21.scatter(data["q_stdev"],data["q_skew"] ,color='k',marker='D')
    plt.xlabel("Std. Deviation")  
    #plt.ylabel("Skewness")
    axqu21.axis([0.5*min(data["q_stdev"]),2*min(data["q_stdev"]), -1, 1])         
    axqu21.plot(data["q_stdev"][id_minstdev],data["q_skew"][id_minstdev] ,color='r',marker='s',markersize=10)
    axqu21.plot(data["q_stdev"][id_mindis],data["q_skew"][id_mindis] ,color='b',marker='*',markersize=15)
    axqu21.axhline(0, color='g', linestyle='-')   
    axqu21.axhline(0.2, color='r', linestyle=':')   
    axqu21.axhline(-0.2, color='r', linestyle=':') 
    axqu21.axvline(min(data["q_stdev"]), color='g', linestyle='--')  
    axqu21.axvline(1.1*min(data["q_stdev"]), color='r', linestyle=':')   
    
    kk = fqu2.sca(axqu22)     
    axqu22.scatter(data["q_stdev"],data["q_kurt"] ,color='k',marker='D')
    axqu22.plot(data["q_stdev"][id_minstdev],data["q_kurt"][id_minstdev] ,color='r',marker='s',markersize=10)
    axqu22.plot(data["q_stdev"][id_mindis],data["q_kurt"][id_mindis] ,color='b',marker='*',markersize=15)
    axqu22.axvline(min(data["q_stdev"]), color='g', linestyle='--')  
    axqu22.axvline(1.1*min(data["q_stdev"]), color='r', linestyle=':')   
    axqu22.axhline(0, color='g', linestyle='-')   
    #axqu22.set_ylabel("Kurtosis")
    axqu22.axis([0.5*min(data["q_stdev"]),2*min(data["q_stdev"]),-1,1])         
    plt.xlabel("Std. Deviation")    
        
    kk = fqu2.sca(axqu23)     
    axqu23.scatter(q_distance,data["q_skew"] ,color='k',marker='D')
    plt.xlabel("Quality Distance")  
    plt.ylabel("Skewness")
    axqu23.axis([-0.1,2,-1,1])         
    axqu23.plot(q_distance[id_minstdev],data["q_skew"][id_minstdev] ,color='r',marker='s',markersize=10)
    axqu23.plot(q_distance[id_mindis],data["q_skew"][id_mindis] ,color='b',marker='*',markersize=15)
    axqu23.axhline(0, color='g', linestyle='-')   
    axqu23.axhline(0.2, color='r', linestyle=':')   
    axqu23.axhline(-0.2, color='r', linestyle=':') 
    axqu23.axvline(min(q_distance), color='g', linestyle='--')  
    axqu23.axvline(2*min(q_distance), color='r', linestyle=':')   
    
    kk = fqu2.sca(axqu24)     
    axqu24.scatter(q_distance,data["q_kurt"] ,color='k',marker='D')
    axqu24.plot(q_distance[id_minstdev],data["q_kurt"][id_minstdev] ,color='r',marker='s',markersize=10)
    axqu24.plot(q_distance[id_mindis],data["q_kurt"][id_mindis] ,color='b',marker='*',markersize=15)
    axqu24.axvline(min(q_distance), color='g', linestyle='--')  
    axqu24.axvline(2*min(q_distance), color='r', linestyle=':')   
    axqu24.axhline(0, color='g', linestyle='-')   
    axqu24.set_ylabel("Kurtosis")
    axqu24.axis([-0.1,2,-1 ,1])         
    plt.xlabel("Quality Distance")  
    
    kk = fqu2.sca(axqu26)     
    axqu26.scatter(q_distance,data["q_stdev"] ,color='k',marker='D')
    axqu26.plot(q_distance[id_minstdev],data["q_stdev"][id_minstdev] ,color='r',marker='s',markersize=10)
    axqu26.plot(q_distance[id_mindis],data["q_stdev"][id_mindis] ,color='b',marker='*',markersize=15)
    axqu26.axvline(min(q_distance), color='g', linestyle='--')  
    axqu26.axvline(2.*min(q_distance), color='r', linestyle=':')   
    axqu26.axhline(min(data["q_stdev"]), color='g', linestyle='--')  
    axqu26.axhline(1.1*min(data["q_stdev"]), color='r', linestyle=':')   
    axqu26.set_ylabel("Std. Deviation")  
    axqu26.axis([-0.1,2,0.5*min(data["q_stdev"]),2*min(data["q_stdev"])])         
    plt.xlabel("Quality Distance")      

    kk = fqu2.sca(axqu25)     
    axqu25.scatter(data["q_skew"],data["q_kurt"] ,color='k',marker='D')
    axqu25.plot(data["q_skew"][id_minstdev],data["q_kurt"][id_minstdev] ,color='r',marker='s',markersize=10)
    axqu25.plot(data["q_skew"][id_mindis],data["q_kurt"][id_mindis] ,color='b',marker='*',markersize=15)
    axqu25.axhline(0, color='g', linestyle='-')   
    axqu25.axvline(0, color='g', linestyle='-')   
    axqu25.axvline(0.2, color='r', linestyle=':')   
    axqu25.axvline(-0.2, color='r', linestyle=':') 
    axqu25.set_xlabel("Skewness")    
    axqu25.axis([-1,1,-1,1])
    fqu2.savefig('quality_check_2.png', dpi=150, bbox_inches='tight')
    ######################################
    # PLOT Quality Indexes
    ######################################
    fqu3 = plt.gcf()
    fqu3.set_size_inches(7,7)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gqu3 = gridspec.GridSpec(1, 1)
    gqu3.update(left=0.1, right=0.95,bottom=0.12,top=0.95,hspace=0.3,wspace=0.3)
    axqu31 = plt.subplot(gqu3[0:1,0:1])
    kk = fqu3.sca(axqu31)     
    axqu31.scatter(data["q_skew"],data["q_kurt"] ,color='k',marker='D')
    axqu31.plot(data["q_skew"][id_minstdev],data["q_kurt"][id_minstdev] ,color='r',marker='s',markersize=10)
    axqu31.plot(data["q_skew"][id_mindis],data["q_kurt"][id_mindis] ,color='b',marker='*',markersize=15)
    axqu31.axhline(0, color='g', linestyle='-')   
    axqu31.axvline(0, color='g', linestyle='-')   
    axqu31.axvline(0.2, color='r', linestyle=':')   
    axqu31.axvline(-0.2, color='r', linestyle=':') 
    axqu31.set_ylabel("Kurtosis")
    axqu31.set_xlabel("Skewness")
    fqu3.savefig('quality_check_3.png', dpi=150, bbox_inches='tight')
    return id_mindis[0]


def find_best_template(in_config):
    ######################################
    # 1 - Read spectra and their errors
    ######################################
    # file names of the telluric and the science objects
    s_telu, wcs_telu= read_spec(in_config["telluric_file"])    
    s_scie, wcs_scie = read_spec(in_config["science_file"])
    ######################################
    # 2 - Read templates
    #  from a list of template fluxes without knowing how big the list is.
    ######################################
    ntemplates = len(in_config["template_list"])
    f_temp, e_temp = load_templates(in_config) 
    if "rvspeed" in in_config:
        rad_vel = np.zeros(ntemplates) + in_config['rvspeed']
        vshift = in_config['rvspeed']
        print(' Radial velocity provided!' )
    else:
        # Show region for RV calculation
        print(' Calculating radial velocity!' )
        show_region_rv(s_telu,f_temp[0], in_config)
        ######################################
        #5/0
        # 3 - Get radial velocity of telluric star from the templates
        rad_vel = get_rv(s_telu,wcs_telu,f_temp,e_temp,
                         in_config["rvwavmin"], in_config["rvwavmax"],ntemplates)
        vshift =np.mean(rad_vel)
        ######################################
    with open("NISCAL_log.txt", "w") as flog:
            print("# Index Name      Template   ST    Wscale   Vrad  Wmin  Wmax  q_min q_max q_stdev q_skew q_kurt", file=flog)    
    with open("NISCAL_log.txt", "a") as flog:
        stdev_control = 1e4  # just a big number
        for ii in np.arange(ntemplates):
            ######################################
            # 4 - Shift template in radial velocity and
            # resample template to match telluric star's spectrum
            # PLOT -  templates, ratios and difference with the telluric
            
            disp_grid_i, quality_stats, spec_telu_i, spec_dif_i, spec_rat_i = get_tell_cor(rad_vel[ii], 
                                                                                    s_telu, wcs_telu, 
                                                                                    f_temp[ii], 
                                                                                    in_config, ii)
            print(ii+1, in_config["name_tel"],in_config["template_list"][ii]["name"],
                  in_config["template_list"][ii]["sp_type"],
                  in_config["which_wav_scale"],rad_vel[ii],in_config["min_wav_q"],
                  in_config["max_wav_q"], quality_stats[0], quality_stats[1], 
                  quality_stats[2], quality_stats[3], quality_stats[4], file=flog)
            # Automatize the selection of the best template based on standard 
            # deviation in the quality assesment region of the difference spetrum
            if (quality_stats[2] < stdev_control):
                stdev_control = quality_stats[2] 
                final_tel_correction = spec_rat_i
                final_tel_diff = spec_dif_i
                final_disp = disp_grid_i
                final_spec_telu = spec_telu_i
                final_template = ii
            #------------------------------------
            # TO FIND BEST TEMPLATE:
            # Calculate S_correction for all templates and calculate an index (e.g. Std.Dev) 
            # for ranking and choosing the best option.
            # This is the quality assesment index and it is calculated at a defined 
            # wavelength region of the difference spectrum.
            # ######
    small_dis_temp = plot_quality("NISCAL_log.txt")
    disp_grid_i, quality_stats, spec_telu_i, spec_dif_i, spec_rat_i = get_tell_cor(rad_vel[small_dis_temp], 
                                                                                    s_telu, wcs_telu, 
                                                                                    f_temp[small_dis_temp], 
                                                                                    in_config, small_dis_temp)
    stdev_control = quality_stats[2] 
    final_tel_correction = spec_rat_i
    final_tel_diff = spec_dif_i
    final_disp = disp_grid_i
    final_spec_telu = spec_telu_i
    final_template = small_dis_temp
    """
    FROM IRAF nsextract:
    The task displays the S/N for each spectrum.  This is calculated  in
    two ways.
    First,  using  the  variance  array,  by  generating a S/N spectrum,
    dividing the science data by the square root of the  variance.   The
    midpt statistic (median) is then calculated for the spectrum.
    Second,  by  generating  a  smooth  spectrum  with  a median filter,
    dividing the original by the smoothed  spectrum  and  measuring  the
    noise in the result (using IMSTAT with nclip=1).
    """
    # S_correction has 8 spectra:
    # [0] : the telluric spectrum, 
    # [1] : the telluric error, 
    # [2] : the template, 
    # [3] : the template error, 
    # [4] : the difference 
    # [5] : The error in the difference    
    # [6] : the ratio  -> Tellulric correction.
    # [7 y 8] : the errors in the ratio
    ######################################
    # Apply telluric correction to the science spectrum
    tcor_sci, tcor_telu = apply_tell_cor(s_scie,final_spec_telu,final_tel_correction, in_config["which_wav_scale"], final_disp)
    return final_template, tcor_sci, tcor_telu, final_tel_correction, final_tel_diff

def get_exp_time(filename):    
    h_some = fits.open(filename)
    #wcs_sci = wcs.WCS(h_telu[('sci',1)].header,h_telu)
    exptime = h_some[1].header['EXPTIME']
    print(' Exposure time : ', exptime)
    return exptime

def get_slit_cor(fullwidth, slit_width,name_source):
    
   
    psf = Gaussian2DKernel(fullwidth/2.3548)
    xc,yc = psf.center
    xsize,ysize =psf.shape
    slit_fraction = psf.array[yc-np.int(slit_width/2 - 0.5):yc+np.int(slit_width/2+0.5),0:xsize].sum()
    slit_c = 1.0 / slit_fraction #1.149 * fullwidth/slit_width
   
    fsl = plt.gcf()
    fsl.set_size_inches(12,8)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'14'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gsl = gridspec.GridSpec(1, 2)
    gsl.update(left=0.1, right=0.95,bottom=0.15,top=0.95,hspace=0.3,wspace=0.3)
    ax21sl = plt.subplot(gsl[0:1,0:1])
    ax22sl = plt.subplot(gsl[0:1,1:2])
    kk = fsl.sca(ax21sl)      
    plt.imshow(psf.array)

    ax21sl.axhline(yc-slit_width/2, color='r', linestyle='-')   
    ax21sl.axhline(yc+slit_width/2, color='r', linestyle='-')   
    ax21sl.set_title("Gaussian PSF, FWHM = "+str(np.around(fullwidth, decimals=2))+" pix")    
    ax21sl.annotate('Slit width: '+str(np.around(slit_width, decimals=1))+' pix', xy =(0,yc-2.0), color='r')  
    
    kk = fsl.sca(ax22sl)  
    plt.imshow(psf.array[yc-np.int(slit_width/2-0.5):yc+np.int(slit_width/2+0.5),0:xsize])
    ax22sl.set_title("Fraction of PSF inside the slit: "+str(np.around(slit_fraction, decimals=3)))    
    #ax22sl.annotate('Fraction of PSF: '+str(np.around(slit_fraction, decimals=3)), xy =(1,0), color='r')  
    fsl.savefig('Slit_loss_'+name_source+'.png', dpi=100, bbox_inches='tight')
    return slit_c

def PhotFlux(tcor_science,in_config, DataBase): 
    mag_in =   in_config["magnitude_sci"]
    er_mag_in = in_config["er_magnitude_sci"]
    scale_wav = in_config["which_wav_scale"]
    #############################
    ## Estimate or read the SLIT LOSS correction
    # this is a multiplicative factor 
    if "slitloss_sci" in in_config:
        slit_cor = in_config["slitloss_sci"]
    else:
        slit_cor = get_slit_cor(in_config["fwhm_sci"], in_config["slit_width"],in_config["name_sci"]) 
        #else:
        #    slit_cor = 1.0   
    slit_ts_science = tcor_science * slit_cor
    #############################
    ## Calculate expected flux from the 2MASS magnitude and error
    total_flux_from_mag, total_flux_from_mag_p, total_flux_from_mag_m, trans_filter = mag2flux_2mass(mag_in,er_mag_in,scale_wav, DataBase)
    print(' Expected Total Flux: ',total_flux_from_mag,', range: ',total_flux_from_mag_m,'-', total_flux_from_mag_p)
    #  Load FILTER TRANSMISSION CURVE 
    filter_wav,filter_trans = np.loadtxt(trans_filter, skiprows=1, unpack=True)
    #Integrate the science spectrum
    ################################
    # Create filtering function and calculate integrated flux in the filter
    filter_function = interp1d(filter_wav,filter_trans, kind='cubic')
    collected_flux = slit_ts_science.flux.value * filter_function(tcor_science.spectral_axis.value*1e-4)
    integrated_spec = np.trapz(collected_flux,tcor_science.spectral_axis.value) * u.Unit('erg cm-2 s-1 AA-1')    
    flu_scale = total_flux_from_mag / integrated_spec 
    flu_scale_p = total_flux_from_mag_p / integrated_spec 
    flu_scale_m = total_flux_from_mag_m / integrated_spec 
    ############################
    aux_spec_flux = slit_ts_science * flu_scale
    aux_spec_flux_p = slit_ts_science * flu_scale_p
    aux_spec_flux_m = slit_ts_science * flu_scale_m    
    collected_flux = aux_spec_flux.flux.value * filter_function(tcor_science.spectral_axis.value*1e-4)
    integrated_spec = np.trapz(collected_flux,tcor_science.spectral_axis.value) * u.Unit('erg cm-2 s-1 AA-1')    
    
    collected_flux_p = aux_spec_flux_p.flux.value * filter_function(tcor_science.spectral_axis.value*1e-4)
    integrated_spec_p = np.trapz(collected_flux_p,tcor_science.spectral_axis.value) * u.Unit('erg cm-2 s-1 AA-1')
    collected_flux_m = aux_spec_flux_m.flux.value * filter_function(tcor_science.spectral_axis.value*1e-4)
    integrated_spec_m = np.trapz(collected_flux_m,tcor_science.spectral_axis.value) * u.Unit('erg cm-2 s-1 AA-1')
    print(' measured Total Flux after calibration: ',integrated_spec,' range:',integrated_spec_m,'-', integrated_spec_p)
    fluxed_spec = aux_spec_flux 
    fluxed_spec.meta = tcor_science.meta
    fluxed_spec_p = aux_spec_flux_p
    fluxed_spec_m = aux_spec_flux_m 
    ############################
    # Plot  
    ############################
    ff = plt.gcf()
    ff.set_size_inches(9,6)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gtel = gridspec.GridSpec(1, 1)
    gtel.update(left=0.1, right=0.95,bottom=0.12,top=0.95)
    axf1 = plt.subplot(gtel[0:1,0:1])
    kk = ff.sca(axf1)     
    axf1.step(fluxed_spec_p.spectral_axis, fluxed_spec_p.flux ,color='r', linestyle=':' ,label='$\pm 1\sigma$ Phot. Error') 
    axf1.step(fluxed_spec_m.spectral_axis, fluxed_spec_m.flux ,color='r', linestyle=':' ) 
    
    # techo y piso del error mas el valor:
    roofi= fluxed_spec_p.flux.value - np.sqrt(1.00/fluxed_spec.uncertainty.array) #(fluxed_spec_p.flux+fluxed_er_p.flux)
    floori= fluxed_spec_m.flux.value + np.sqrt(1.00/fluxed_spec.uncertainty.array) #(fluxed_spec_p.flux-fluxed_er_p.flux)
    plt.fill_between(fluxed_spec.spectral_axis.value, floori, roofi,
                     facecolor='0.75',edgecolor='0.5' ,label='Total Error')
    # techo y piso del error mas el valor:
    #roofm=  fluxed_spec.flux.value + np.sqrt(1.00/fluxed_spec.uncertainty.array) #(fluxed_spec_m.flux+fluxed_er_m.flux)
    #floorm= fluxed_spec.flux.value - np.sqrt(1.00/fluxed_spec.uncertainty.array) #(fluxed_spec_m.flux-fluxed_er_m.flux)
    #plt.fill_between(fluxed_spec_m.spectral_axis.value, floorm, roofm, color='b',
    #                 facecolor='0.75',edgecolor='0.5')
    
    axf1.step(fluxed_spec.spectral_axis, fluxed_spec.flux ,color='k', linestyle='-' ,label='Flux calibrated spectrum') 
    plt.title("Flux calibrated science spectrum - PhotFlux output")
    #plt.ylabel("Relative Flux")
    plt.legend()
    ff.savefig('flux_photcal_science.png', dpi=150, bbox_inches='tight')
    
    ff2 = plt.gcf()
    ff2.set_size_inches(9,6)
    gtel2 = gridspec.GridSpec(1, 1)
    gtel2.update(left=0.1, right=0.95,bottom=0.12,top=0.95)
    axf2 = plt.subplot(gtel2[0:1,0:1])
    kk = ff.sca(axf2)     
    axf2.step(fluxed_spec_p.spectral_axis, fluxed_spec_p.flux ,color='r', linestyle='--',label='$\pm 1\sigma$ Phot. Error') 
    axf2.step(fluxed_spec_m.spectral_axis, fluxed_spec_m.flux ,color='r', linestyle='--' )
    axf2.step(fluxed_spec.spectral_axis, fluxed_spec.flux ,color='k', linestyle='-' ,label='Flux calibrated spectrum') 
    axf2.step(fluxed_spec_m.spectral_axis, collected_flux * u.Unit('erg cm-2 s-1 AA-1') ,color='k', linestyle=':' ,label='Transmitted flux') 
    axf2.plot(filter_wav*1e4,filter_trans * (max(fluxed_spec_m.flux.value) + min(fluxed_spec_m.flux.value))*0.8 ,'m-',label='Relative Spectral Response')
    plt.title("Science spectrum and filter response for flux calibration")
    #plt.ylabel("Relative Flux")
    plt.legend()
    ff2.savefig('science_and_calibration_filter.png', dpi=150, bbox_inches='tight')
    return fluxed_spec,fluxed_spec_p, fluxed_spec_m

    
def TellFlux(best_template, in_config, DataBase): 
    #  Las telluricas suelen ser observadas
    #  en Bright Mode osea Number of Reads =1
    #  la combinacion de ncombine suele ser "average".
    #  Esto esta en el Header en los keywords 
    #  NSCHLCOM= 'average'
    #  READMODE= 'Bright  '           / Detector Readout mode
    #  NREADS  =                    1 / Number of reads
    #  En este caso, solo deberiamos dividir el espectro por 
    #  el tiempo de exposicion para que quede en cts/s.
    # The idea is to scale the template to match the flux 
    # of the telluric star. Divide the telluric by the exposure time and then
    # by the scaled template to obtain ADU/s /F_nu
    
    mag_tel =   in_config["magnitude_tel"]
    er_mag_tel = in_config["er_magnitude_tel"]
    scale_wav = in_config["which_wav_scale"]
    s_telu, wcs_telu= read_spec(in_config["telluric_file"])    
    s_scie, wcs_scie = read_spec(in_config["science_file"])
    exp_time_tell = get_exp_time(in_config["telluric_file"])    
    exp_time_scie = get_exp_time(in_config["science_file"])    

    #############################
    ## Calculate expected flux from the 2MASS magnitude and error
    total_flux_from_mag, total_flux_from_mag_p, total_flux_from_mag_m, trans_filter = mag2flux_2mass(mag_tel,er_mag_tel,scale_wav, DataBase)
    print(' Expected Total Flux: ',total_flux_from_mag,', range: ',total_flux_from_mag_m,'-', total_flux_from_mag_p)
    #  Load FILTER TRANSMISSION CURVE 
    filter_wav,filter_trans = np.loadtxt(trans_filter, skiprows=1, unpack=True)
    # Load template
    
    #f_temp = load_templates(in_config) 
    n_template = in_config["template_list"][best_template]["fits"]
    s_temp, e_temp = read_template(n_template)
    delta_wav = 2.6 
    new_disp_grid = np.arange(in_config["minwav"], in_config["maxwav"], delta_wav) * u.AA
    fluxcon = FluxConservingResampler()
    ds_temp_aux = fluxcon(s_temp, new_disp_grid)
    de_temp_aux = fluxcon(e_temp, new_disp_grid)

    if "rvspeed" in in_config:
        rad_vel = in_config['rvspeed']
        print(' Radial velocity provided!' )
    else:
        # Show region for RV calculation  
        print(' Calculating radial velocity!' )
        show_region_rv(s_telu,ds_temp_aux, in_config)
        ######################################
        # 3 - Get radial velocity of telluric star from the templates
        rad_vel = get_rv_one(s_telu,wcs_telu,ds_temp_aux, in_config["rvwavmin"], in_config["rvwavmax"],best_template)
    ######################################
    # Shift template in radial velocity and
    # resample template to match telluric star's spectrum    
    minwav = in_config["minwav"]
    maxwav = in_config["maxwav"] 
    tempname = in_config["template_list"][best_template]["name"]
    scale_wav = in_config["which_wav_scale"]
    
    # Shift template in radial velocity      
    rv_s_temp_flux_1, rv_s_temp_w = pyasl.dopplerShift(ds_temp_aux.spectral_axis, ds_temp_aux.flux, rad_vel, edgeHandling="firstlast")
    
    rvs_temp = Spectrum1D(spectral_axis=ds_temp_aux.spectral_axis, flux=rv_s_temp_flux_1* u.Unit('erg cm-2 s-1 AA-1')) 
    
    ######################################
    #  Resample template and  telluric star's spectrum
    #delt_wav_tel = w_telu.wcs.cd[0,0]
    new_disp_grid = np.arange(minwav, maxwav, 2.6) * u.AA  
    fluxcon = FluxConservingResampler()
    ds_temp = fluxcon(rvs_temp, new_disp_grid) 
    ds_telu = fluxcon(s_telu, new_disp_grid) 
    ds_scie = fluxcon(s_scie, new_disp_grid) 
    ds_telu.meta = s_telu.meta
    ds_scie.meta = s_scie.meta
    #Integrate the template spectrum
    ################################
    # Create fintering function and calculate integrated flux in the filter
    filter_function = interp1d(filter_wav,filter_trans, kind='cubic')
    collected_flux = ds_temp.flux.value * filter_function(ds_temp.spectral_axis.value*1e-4)
    integrated_spec = np.trapz(collected_flux,ds_temp.spectral_axis.value) * u.Unit('erg cm-2 s-1 AA-1') 
    flu_scale = total_flux_from_mag / integrated_spec 
    flu_scale_p = total_flux_from_mag_p / integrated_spec 
    flu_scale_m = total_flux_from_mag_m / integrated_spec 
    ############################
    aux_spec_flux = ds_temp * flu_scale
    aux_spec_flux_p = ds_temp * flu_scale_p
    aux_spec_flux_m = ds_temp * flu_scale_m    
    collected_flux = aux_spec_flux.flux.value * filter_function(ds_temp.spectral_axis.value*1e-4)
    integrated_spec = np.trapz(collected_flux,ds_temp.spectral_axis.value) * u.Unit('erg cm-2 s-1 AA-1')    
    collected_flux_p = aux_spec_flux_p.flux.value * filter_function(ds_temp.spectral_axis.value*1e-4)
    integrated_spec_p = np.trapz(collected_flux_p,ds_temp.spectral_axis.value) * u.Unit('erg cm-2 s-1 AA-1')
    collected_flux_m = aux_spec_flux_m.flux.value * filter_function(ds_temp.spectral_axis.value*1e-4)
    integrated_spec_m = np.trapz(collected_flux_m,ds_temp.spectral_axis.value) * u.Unit('erg cm-2 s-1 AA-1')
    print(' Measured Total Flux of Template after calibration: ')
    print(integrated_spec,' range:',integrated_spec_m,'-', integrated_spec_p)
    ############################################################
    # Slit loss correction must be included in the flux calibration 
    # We need the transformation from counts/s to flux. So, we nkow the flux
    # of the star, and we must calculate the counts per second that it should
    # produce, based on what was measured and what was missing.
    # This is a multiplicative factor 
    
    if "slitloss_tel" in in_config:
        slit_cor_t = in_config["slitloss_tel"]
    else:
        slit_cor_t = get_slit_cor(in_config["fwhm_tel"], in_config["slit_width"],in_config["name_tel"]) 
    slit_ds_telu = ds_telu * slit_cor_t

    if "slitloss_sci" in in_config:
        slit_cor_s = in_config["slitloss_sci"]
    else:
        slit_cor_s = get_slit_cor(in_config["fwhm_sci"], in_config["slit_width"],in_config["name_sci"]) 
    slit_ds_science = ds_scie * slit_cor_s
    #############################
    flux_cal_func = (slit_ds_telu/exp_time_tell ) / aux_spec_flux
    flux_cal_func_p = (slit_ds_telu/exp_time_tell ) / aux_spec_flux_p
    flux_cal_func_m = (slit_ds_telu/exp_time_tell ) / aux_spec_flux_m
    ############################################################
    fts_sci = (slit_ds_science/exp_time_scie) / flux_cal_func
    fts_sci_p = (slit_ds_science/exp_time_scie) / flux_cal_func_p
    fts_sci_m = (slit_ds_science/exp_time_scie) / flux_cal_func_m 
    fts_sci.meta = ds_scie.meta
    
    fftf = plt.gcf()
    fftf.set_size_inches(9,6)
    rc('font',**{'family':'serif','serif':[' Times New Roman'],'size':'12'})
    rc('grid',**{'color':'black','linestyle':':','linewidth':'1.2','alpha':'0.5'})
    rc('text', usetex=True)
    gtelf = gridspec.GridSpec(1, 1)
    gtelf.update(left=0.1, right=0.95,bottom=0.12,top=0.95)
    axtf = plt.subplot(gtelf[0:1,0:1])
    kk = fftf.sca(axtf)    
    axtf.step(fts_sci_p.spectral_axis, fts_sci_p.flux ,color='r', linestyle=':' ,label='$\pm 1\sigma$ Calibration Error') 
    axtf.step(fts_sci_m.spectral_axis, fts_sci_m.flux ,color='r', linestyle=':' ) 
    # techo y piso del error mas el valor:
    rooft= fts_sci_p.flux.value - np.sqrt(1.00/fts_sci.uncertainty.array) #(fluxed_spec_p.flux+fluxedt_er_p.flux)
    floort= fts_sci_m.flux.value + np.sqrt(1.00/fts_sci.uncertainty.array) #(fluxed_spec_p.flux-fluxed_er_p.flux)
    plt.fill_between(fts_sci.spectral_axis.value, floort, rooft,
                     facecolor='0.75',edgecolor='0.5' ,label='Total Error')
    # techo y piso del error mas el valor:
    #roofm=  fluxed_spec.flux.value + np.sqrt(1.00/fluxed_spec.uncertainty.array) #(fluxed_spec_m.flux+fluxed_er_m.flux)
    #floorm= fluxed_spec.flux.value - np.sqrt(1.00/fluxed_spec.uncertainty.array) #(fluxed_spec_m.flux-fluxed_er_m.flux)
    #plt.fill_between(fluxed_spec_m.spectral_axis.value, floorm, roofm, color='b',
    #                 facecolor='0.75',edgecolor='0.5')
    axtf.step(fts_sci.spectral_axis, fts_sci.flux ,color='k', linestyle='-' ,label='Flux calibrated spectrum') 
    plt.title("Flux calibrated science spectrum - TellFlux output")
    #plt.ylabel("Relative Flux")
    plt.legend()
    fftf.savefig('flux_telcal_science.png', dpi=150, bbox_inches='tight')    
    
    return fts_sci,fts_sci_p,fts_sci_m


############################################
###   PROGRAMA
##########################################    
if __name__ == '__main__':
    ######################################
    # 0 - Read configuration file
    ######################################
    #config_file = sys.argv[1]
    #config_file = 'niscal_conf.yaml'
    #config_file = 'niscal_conf_K1.yaml'    
    #config_file = 'niscal_conf_K2.yaml'    
    #config_file = 'niscal_conf_K3.yaml'    
    #config_file = 'niscal_conf_H_st.yaml'    
    config_file = 'niscal_conf_H.yaml'    

    stream = open(config_file, 'r')
    in_config = yaml.load(stream, Loader=yaml.FullLoader)
    #------------------------------------------
    # ---- Definitions and setup 
    # working directory
    DataBase='/home/gonza/Proyecto_AGN_OCSVM/Codigos/NISCAL/database/'
    wdir= in_config["wdir"] 
    os.chdir(wdir)   
    ######################################
    # 1 - Telluric correction. if you know the template just go get it.
    if "which_template" in in_config:
        print('Getting template: ',in_config['which_template'])
        tcor_science, tcor_telluric, spec_rat, spec_dif  = get_this_one(in_config)
        best_template = in_config['which_template'] - 1
    # If not, find me the best template:
    else:
        print('Finding the best template for you!')
        best_template, tcor_science, tcor_telluric, spec_rat, spec_dif   = find_best_template(in_config)
        print('Your best option is Template ',best_template+1,':',in_config["template_list"][best_template]["name"] )
    ######################################
    # 2 - Flux calibration
    if in_config["fluxing"]:
        if "magnitude_sci" in in_config:
            ######################################
            # Flux calibration 1: Known magnitude in the same band
            ######################################
            # If the magnitude in the 2MASS catalog is known,
            # the code will correct the flux level to reproduce 2MASS magnitude
            # using the 2MASS filter response.    
            #-------
            ftcor_sci_2mass , ftcor_sci_2mass_p, ftcor_sci_2mass_m = PhotFlux(tcor_science, in_config, DataBase)
            generic_fits(ftcor_sci_2mass,"ftc_sci_2mass.fits")
            generic_fits(ftcor_sci_2mass_p,"ftc_sci_2mass_p.fits")
            generic_fits(ftcor_sci_2mass_m,"ftc_sci_2mass_m.fits")
        if "magnitude_tel" in in_config:
            ######################################
            # Flux calibration 2: Telluric standard
            ######################################
            # If you know the magnitude of the telluric standard in 2MASS, 
            # you can use TellFlux 
            # Then, the ratio between the fluxed spectrum and the
            # teluric corrected ct/s spectrum (i.e. divided exp time)
            # should give us the flux calibration function.
            # The telluric corrected science spectrum
            # in ct/s can be transformed in f_lambda by
            # multiplying by the flux calibration function
            # from the telluric star.
            #-------
            print('Fluxed by telluric star spectrum ')
            ftcor_sci_tel , ftcor_sci_tel_p, ftcor_sci_tel_m  = TellFlux(best_template, in_config, DataBase) 
            generic_fits(ftcor_sci_tel,"ftc_sci_tel.fits")
            generic_fits(ftcor_sci_tel_p,"ftc_sci_tel_p.fits")
            generic_fits(ftcor_sci_tel_m,"ftc_sci_tel_m.fits")
        if "magnitude_stand" in in_config:
            print('Some day...')
            ######################################
            # Flux calibration 3: Spectroscopic standard observed a different day
            ######################################
    else:
        print('Flux Calibration Disabled')
    ######################################
    ######################################
    # # Write fits files:
    #  1- Telluric calibrated science and error
    generic_fits(tcor_science,"tc_science_out.fits")
    #  2- telluric correction
    generic_fits(spec_rat,"telluric_correction.fits")
    print('The End')
