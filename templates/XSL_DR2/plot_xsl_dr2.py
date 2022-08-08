############################################################################
#
# plot_xsl_dr2.py
#
#
# Description:
#       Read the binary DR2 XSL spectra and plot all the arms for a given XSL id per page
#
# Functions:
#       -- read_bin_spec: Read a binary spectrum
#	-- plot_one: Plot a binary spectrum
#	-- plot_all_arms: Plot the three arms on the same figure 
#       -- MAIN
#
#
# History:
#       - 27 / 11 / 19 : Creation 
#
#
############################################################################

## Load some Python modules
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


############################################################################

# **********************************************************
#
# 	Read a binary spectrum
#
# **********************************************************

def read_bin_spec(spec_name, ang='', star_kw='', loss_corr_kw=''):


	# If the file exists
	try: 
	
		###########

		# Print some info
		print(' ')
		print('File to read:', spec_name)


		###########

		# Open the spectrum
		hdu = fits.open(spec_name)
		

		###########

		# Define the columns
		flux = hdu[1].data['FLUX']
		waves = hdu[1].data['WAVE']
		

		###########

		# Get the units of the waves
		prim_hdr = hdu[0].header
		sec_hdr = hdu[1].header

		unit = ''
		unit = sec_hdr['TUNIT1']
		

		###########

		# Change the units to Angstrom
		if (len(ang) != 0):
			if (unit == 'nm'):
				unit = 'A'
				waves = waves * 10.


		###########

                # Convert the arrays to numpy arrays 
		waves_final = np.array(waves)
		flux_final = np.array(flux)


		###########

                # Get the star name
		if (len(star_kw) != 0):
                        star = ''
                        star = prim_hdr['HNAME']
            
		###########

                # Get the flux-loss kw
		if (len(loss_corr_kw) != 0):
                        loss_corr = ''
                        loss_corr = prim_hdr['LOSS_COR']
            

	###########

        # File not found 
	except IOError:
		print('=> File not found: ', spec_name)
		flux_final = 999
		waves_final = 999
		unit = '999'


	###########

        # Define the final results
	list_final = [flux_final, waves_final, unit]

	# Return the flux, waves, unit
	if (len(star_kw) != 0):
            list_final.append(star)

	if (len(loss_corr_kw) != 0):
            list_final.append(loss_corr)


        # Return the results
	return list_final
	
	


############################################################################

# **********************************************************
#
#       Plot one arm
#
# **********************************************************

def plot_one(count_plot, frame, ytitle, xslid=''):


        ##################

        # Read the frame
        [flux_temp, waves_temp, unit, star, loss_corr] = read_bin_spec(frame, ang='on', star_kw='on', loss_corr_kw='on')
	

        ##################

        # Restrict the wavevelength
        if (ytitle == 'NIR'):
            waves = waves_temp[(waves_temp> 10500) & (waves_temp < 24500)]
            flux = flux_temp[(waves_temp> 10500) & (waves_temp < 24500)]
        else:
            waves = waves_temp[waves_temp> waves_temp[0]+110]
            flux = flux_temp[waves_temp> waves_temp[0]+110]


        ##################

        # Define the subfigure
        sub = plt.subplot(3, 1, count_plot)

        # Plot the tellurics in grey
        if (ytitle == 'NIR'):

            # Plot the telluric parts in grey
            list_tell_beg = [11050, 13450, 17950, 22700]
            list_tell_end = [11700, 14500, 19500, 22780]

        # Plot the dichroics in grey
        elif (ytitle == 'UVB'):

            # Plot the telluric parts in grey
            list_tell_beg = [5400]
            list_tell_end = [waves[-1]]

        # Plot the dichroics in grey
        elif (ytitle == 'VIS'):

            # Plot the telluric parts in grey
            list_tell_beg = [waves[0]]
            list_tell_end = [5800]


        # Do a loop on all the wavelengths
        for iwa in range(len(list_tell_beg)):

                ############
                # TEll part
                idx = [(waves > list_tell_beg[iwa]) & (waves < list_tell_end[iwa])]
                sub.plot(waves[idx], flux[idx], color='grey')

                ############

                # First waves => below
                if (iwa == 0):
                        if (ytitle == 'VIS'): 
                                idx = [(waves > list_tell_end[iwa])]
                                if (len(idx) != 0 ): sub.plot(waves[idx], flux[idx], color='blue', label='LOSS_CORR = ' + str(loss_corr))
                        else:
                                idx = [(waves< list_tell_beg[iwa])]
                                if (len(idx) != 0 ): sub.plot(waves[idx], flux[idx], color='blue', label='LOSS_CORR = ' + str(loss_corr))

                else:
                        # Inter waves: Below
                        idx = [(waves > list_tell_end[iwa-1]) & (waves < list_tell_beg[iwa])]
                        if (len(idx) != 0 ): sub.plot(waves[idx], flux[idx], color='blue')

                        # Last waves => above
                        if (iwa == len(list_tell_beg)-1):
                                idx = [(waves > list_tell_end[iwa])]
                                if (len(idx) != 0 ): sub.plot(waves[idx], flux[idx], color='blue')


        ##################

	# Find the maximum for the y-axis
        flux_beg = np.nanmax(flux[(waves > waves[0]+200) & (waves < waves[0]+700)])
        waves_mid = (waves[-1]-waves[0])/2. + waves[0]
        flux_mid = np.nanmax(flux[(waves > waves_mid-250) & (waves < waves_mid+250)])
        flux_end = np.nanmax(flux[(waves > waves[-1]-600) & (waves < waves[-1]-100)])
        ymax = np.nanmax([flux_beg, flux_mid, flux_end]) * 1.2
        ymin = np.nanmin(flux) * 0.8


        ##################

        # Add the minor points axis
        sub.minorticks_on()

        # Set the limits
        sub.set_xlim(waves[0]-2, waves[-1]+2)
        sub.set_ylim(ymin, ymax)


        ##################

        # Add the labels
        sub.set_xlabel('Wavelength [' + unit + ']')
        sub.set_ylabel(ytitle)

        # Add the legend
        plt.legend()

        # Add upper text
        if (len(xslid) != 0):
            plt.title(xslid + ': ' + star)




############################################################################

# **********************************************************
#
#       Plot the three arms on the same figure
#
# **********************************************************

def plot_all_arms(pdf_doc, xslid, uvb_frame, vis_frame, nir_frame):

        # Define the figure instance
        fig = plt.figure(figsize=(11.69,8.27))


        ##################

        # UVB
        if (uvb_frame != []):
                plot_one(1, uvb_frame, 'UVB', xslid=xslid)


        ##################

        # VIS
        if (vis_frame != []):
                if (uvb_frame == []):
                    plot_one(2, vis_frame, 'VIS', xslid=xslid)
                else:
                    plot_one(2, vis_frame, 'VIS')


        ##################

        # NIR
        if (nir_frame != []):
                if (uvb_frame == [] and vis_frame == []):
                    plot_one(3, nir_frame, 'NIR', xslid=xslid)
                else:
                    plot_one(3, nir_frame, 'NIR')


        ##################

        # Add some space
        plt.tight_layout()

        # Save the figure
        pdf_doc.savefig()

        # Close plt
        plt.close()



############################################################################

# **********************************************************
#
#                    MAIN
#
# **********************************************************

################

# DEfine the path for the data
path_data = ''

# Define the path for the plots
path_out = ''


################

# Open the output plot
output = path_out + 'plot_xsl_dr2.pdf'

# Open the output file
pdf_doc = PdfPages(output)


################

# Do a loop on all the frames
for x in range(911):

	# Define the spectrum
	xsl_short = 'X' + '{:04d}'.format(x+1)

	# Define the full frame
	frame_uvb = 'xsl_spectrum_' + xsl_short + '_uvb.fits'
	frame_vis = 'xsl_spectrum_' + xsl_short + '_vis.fits'
	frame_nir = 'xsl_spectrum_' + xsl_short + '_nir.fits'

	frame_uvb2 = 'xsl_spectrum_' + xsl_short + '_uvb_nlc.fits'
	frame_vis2 = 'xsl_spectrum_' + xsl_short + '_vis_nlc.fits'
	frame_nir2 = 'xsl_spectrum_' + xsl_short + '_nir_nlc.fits'


        ##################

        # File not found
	uframe = []
	if os.path.isfile(path_data+frame_uvb):
		uframe = path_data + frame_uvb
	elif os.path.isfile(path_data+frame_uvb2):
		uframe = path_data + frame_uvb2

	vframe = []
	if os.path.isfile(path_data+frame_vis):
		vframe = path_data + frame_vis
	elif os.path.isfile(path_data+frame_vis2):
		vframe = path_data + frame_vis2

	nframe = []
	if os.path.isfile(path_data+frame_nir):
		nframe = path_data + frame_nir
	elif os.path.isfile(path_data+frame_nir2):
		nframe = path_data + frame_nir2


        ##################
	
	# Launch the prog
	if (uframe != [] or vframe != [] or nframe != []):


                # Launch the prog
                plot_all_arms(pdf_doc, xsl_short, uframe, vframe, nframe)



##################

# Close the file
pdf_doc.close()

# Print some info
print(' ')
print('Output file: ', output)
print(' ')
