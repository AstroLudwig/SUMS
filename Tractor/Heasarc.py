import numpy as np
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table,Column
import glob
import subprocess
import sys
import matplotlib.pyplot as plt
import time
from astropy.io.fits import update

class HeasarcRoutines:
    
    
        def __init__(self,tractor_fits_string,uv_filter):
            self.file_location = tractor_fits_string
            
            
            self.uvot_coincidence_on_source()
            
            self.uvot_coincidence_on_background()
            
            self.systematic_error()

            self.sss()
            
            self.lss()
            
            self.senscor(uv_filter)
            
            self.uvot_flux(uv_filter)
            
            self.write_file(uv_filter)
            
            
        def uvot_coincidence_on_source(self):
            #uvot coincidence on source:
            coin_command1 = 'uvotcoincidence infile=' +self.file_location + '+1 coinfile=CALDB'
            coin_command2 = ' ratecol=TOTAL_RATE errcol=TOTAL_RATE_ERR prefix=TOTAL_ chatter=1'
            coin_command = coin_command1 + coin_command2

            self.bash_command('heainit; %s' %coin_command,print_output=False)
            
            print('----finished uvotcoincidence on source----')
        
        
        def uvot_coincidence_on_background(self):
            #Run background command
            coin_command1 = 'uvotcoincidence infile=' +self.file_location + '+1 coinfile=CALDB'
            coin_command2 = ' ratecol=BACK_RATE errcol=BACK_RATE_ERR prefix=BACK_ chatter=1'
            coin_command = coin_command1 + coin_command2

            self.bash_command('heainit; %s' %coin_command,print_output=False)

            print('----finished uvotcoincidence on background----')
            
            #Corrected for coincidence loss:
            tab1=fits.open(self.file_location)
            dat1=tab1[1].data
            coi_rate = dat1['TOTAL_COI_RATE'] - dat1['BACK_COI_RATE']
            coi_rate_err = np.sqrt(dat1['TOTAL_COI_RATE_ERR'] ** 2. + dat1['BACK_COI_RATE_ERR'] ** 2.)

            # Add these to a new extension in new file:

            orig_cols = dat1.columns
            new_cols = fits.ColDefs([
                fits.Column(name='COI_SRC_RATE', format='1E', array=coi_rate),
                fits.Column(name='COI_SRC_RATE_ERR', format='1E', array=coi_rate_err),
            ])
            hdu_new = fits.BinTableHDU.from_columns(orig_cols + new_cols)
            
            #Make new header updates. 
            hdr1 = tab1[1].header
            
            hdr1['TTYPE26'] = 'COI_SRC_RATE'                                                        
            hdr1['TFORM26'] = '1E      '                                                            
            hdr1['TTYPE27'] = 'COI_SRC_RATE_ERR'                                                    
            hdr1['TFORM27'] = '1E      ' 
            
            update(self.file_location,hdu_new.data,hdr1,1)
            
            tab1.close()
        
            
        
        def systematic_error(self):
            '''Apply a systematic error on the count rate to account for variations in the PSF shape. 
            Currently a value of 5% is recommended. This is just for the error itself.'''    
            
            tab1=fits.open(self.file_location)
            dat1=tab1[1].data
            
            coi_rate = dat1['COI_SRC_RATE']
            coi_rate_err = dat1['COI_SRC_RATE_ERR']
            sys_err = coi_rate*0.025
            
            ap_coi_rate = coi_rate
            ap_coi_rate_err = np.sqrt(coi_rate_err**2. + sys_err**2.)
            
            orig_cols = dat1.columns
            new_cols = fits.ColDefs([
                fits.Column(name='AP_COI_SRC_RATE', format='1E', array=ap_coi_rate),
                fits.Column(name='AP_COI_SRC_RATE_ERR', format='1E', array=ap_coi_rate_err),
            ])
            hdu_new = fits.BinTableHDU.from_columns(orig_cols + new_cols)
       
            #Make new header updates. 
            hdr1 = tab1[1].header
            
            hdr1['TTYPE28'] = 'AP_COI_SRC_RATE'                                                        
            hdr1['TFORM28'] = '1E      '                                                            
            hdr1['TTYPE29'] = 'AP_COI_SRC_RATE_ERR'                                                    
            hdr1['TFORM29'] = '1E      ' 
            
            update(self.file_location,hdu_new.data,hdr1,1)
            
            tab1.close()
            
            print('----applied systematic error----')
            
        def sss(self):
            '''Apply the small scale sensitivity correction'''
            
            sss_command = 'uvotlss input=TABLE infile=' +self.file_location + '+1 lssfile=../Data/Calibration/sssfile5.fits.gz chatter=1'
    
            self.bash_command('heainit; %s' %sss_command,print_output=False)
            
            #Grab the 'LSS Factor' and make a new column for that. (Those keywords will get overwritten below.)
            tab1=fits.open(self.file_location)
            dat1=tab1[1].data
            
            sss_factor = dat1['LSS_FACTOR']
            
            orig_cols = dat1.columns
            new_cols = fits.ColDefs([
                fits.Column(name='SSS_FACTOR', format='1E', array=sss_factor),
            ])
            hdu_new = fits.BinTableHDU.from_columns(orig_cols + new_cols)
       
            #Make new header updates. 
            hdr1 = tab1[1].header
            
            hdr1['TTYPE33'] = 'SSS_FACTOR'                                                        
            hdr1['TFORM33'] = '1E      '                                                            
            
            update(self.file_location,hdu_new.data,hdr1,1)
            
            tab1.close()
            
            print('----finished sss correction-----')
            
            
        def lss(self):
            '''Apply the large scale sensitivity correction'''
            
            lss_command = 'uvotlss input=TABLE infile=' +self.file_location + '+1 lssfile=CALDB chatter=1'
    
            self.bash_command('heainit; %s' %lss_command,print_output=False)
            
            print('----finished lss correction----')
            
        
        def senscor(self,uv_filter):
            '''Figure out the detector sensitivity correction and apply that to all columns.'''
            
            tab1=fits.open(self.file_location)
            dat1=tab1[1].data
            hdr1=tab1[1].header
            
            tmid = (hdr1['TSTART']+hdr1['TSTOP'])/2.
            
            # Read in the calibration file:
            tab2=fits.open('../Data/Calibration/swusenscorr20041120v005.fits')
            
            if uv_filter == 'UVW1':
                dat2=tab2[4].data
            elif uv_filter == 'UVW2':
                dat2=tab2[6].data
            else:
                dat2=tab2[5].data
            
            index = np.where(dat2['TIME'] < tmid)[0]
            time = dat2['TIME'][index[-1]]
            offset = dat2['OFFSET'][index[-1]]
            slope = dat2['SLOPE'][index[-1]]
            
            # Calculate senscorr factor
            duration_year_s = 24.*60.*60.*365.25
            exponent = (tmid-time)/duration_year_s
            senscor_factor = (1+offset)*(1+slope)**exponent
            
            # Update columns
            rate = dat1['LSS_RATE']
            rate_err = dat1['LSS_RATE_ERR']
            
            senscor_rate = rate * senscor_factor
            senscor_rate_err = rate_err * senscor_factor
            
            # Update things:
            orig_cols = dat1.columns
            new_cols = fits.ColDefs([
                fits.Column(name='SENSCORR_RATE', format='1E', array=senscor_rate),
                fits.Column(name='SENSCORR_RATE_ERR', format='1E', array=senscor_rate_err),
            ])
            hdu_new = fits.BinTableHDU.from_columns(orig_cols + new_cols)
       
            #Make new header updates. 
            hdr1['TTYPE34'] = 'AP_COI_SRC_RATE'                                                        
            hdr1['TFORM34'] = '1E      '                                                            
            hdr1['TTYPE35'] = 'AP_COI_SRC_RATE_ERR'                                                    
            hdr1['TFORM35'] = '1E      '  
            
            hdr1['SENS_FAC'] = senscor_factor
            
            update(self.file_location,hdu_new.data,hdr1,1)
            
            tab1.close()
            
            print('----finished senscor correction----')
        
        
        def uvot_flux(self,uv_filter):
            #Run UVOT FLUX on this:
            flux_command1 = 'uvotflux infile=' + self.file_location + '+1 zerofile=CALDB filter=' + uv_filter
            flux_command2 = ' syserr=yes ratecol=SENSCORR_RATE errcol=SENSCORR_RATE_ERR chatter=1'
            flux_command = flux_command1 + flux_command2
            

            self.bash_command('heainit; %s' %flux_command,print_output=False)
            print('----finished uvotflux----')

            
        def write_file(self,uv_filter):
            #Write out to a csv file which I can use within TopCat:
            tab1 = fits.open(self.file_location)
            dat1 = tab1[1].data
            out_csv = self.file_location[:-5]+'.csv' 
            
            resid_frac = dat1['RESID_RATE']/dat1['TOTAL_RATE']

            data = [dat1['KEY'],dat1['RA'], dat1['DEC'], dat1['PIX_X'],
                    dat1['PIX_Y'], dat1['MAG'],dat1['MAG_ERR'],resid_frac,
                    dat1['TOTAL_SATURATED'],dat1['SSS_FACTOR'],dat1['EDGE_FLAG']]
            
            names = ['KEY','RA', 'DEC', 'PIX_X', 'PIX_Y', 'MAG',
                     'MAG_ERR','RESID_FRAC','SATURATED','SSS','EDGE']
            ascii.write(data, out_csv, names=names, delimiter=',', overwrite=True)
            tab1.close()
            
        def bash_command(self,cmd,print_output=True):
            output = subprocess.check_output(['/bin/bash','-i', '-c',cmd])
            if print_output:
                print(output)