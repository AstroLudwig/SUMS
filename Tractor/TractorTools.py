# -*- Copyright (c) 2018, Bethany Ann Ludwig, All rights reserved. -*-
"""
NAME:
    Photometry Retrieve Source 
PURPOSE:
    Automate process of pulling swift fields and finding the Zaritsky sources in them.
    Blue Source Retrieval is a class. You give it an ra/dec and it gives you the SWIFT field
    with the objects in them. It includes the coordinates, regions, catalog information,
    and intial guesses for flux. The entire object could be saved with pickle. 
Notes: 
"""	

from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table,Column
import astropy.units as u
from astropy.io import fits
import pandas as pd 
from astropy.wcs import WCS
import numpy as np
from tractor import *
from pathlib import Path
from photutils import CircularAperture
from photutils import aperture_photometry


class PhotometryTools:
    
    def __init__(self,source,background,psf_filename,fits_table_sname,save_output=False,fit_positions=np.nan,threshold=1.5):
        
        # Background Handling 
        sky_background=background.background
        error=background.error_image
        subtraction_bkgd=background.median2
        # Estimated bkgd with PhotUtils, subtracting off from image.
        self.image = np.copy(source.data) - sky_background
        # Basically 0, the median of the subtracted image, makes tractor happier than null background
        self.sky = ConstantSky(subtraction_bkgd)
        
        # PSF Handling
        if type(psf_filename) == np.ndarray: # Default, unique image to image
            self.psf = psf_filename
        elif Path(psf_filename).suffix == ".txt": # For assuming psf is constant to save time. 
            self.psf = np.loadtxt(psf_filename)
        elif Path(psf_filename).suffix == ".fits": # For pyraf handling
            self.psf_data = fits.open(psf_filename)[0].data
            # normalize out to 5 arcsecs
            n = np.shape(self.psf_data)[0] 
            to_pix = 1  #  Pixel / ArcSecond
            x0 = np.floor(n/2); y0 = np.floor(n/2)
            inner_sum = 0
            size = n * to_pix
            pixel_psf = np.copy(self.psf_data)
            for i in range(n):
                for j in range(n):
                    distance = np.sqrt((i-y0)**2+(j-x0)**2)
                    if distance < 5. * to_pix:
                        inner_sum += pixel_psf[i,j]
            self.psf = self.psf_data / inner_sum
          
        self.pixel_scale = source.cdelt * 3600
         
        ## Tractor 
        self.tractor_image = self.get_tractor_image(self.image,error,self.sky)
        self.tractor_sources = self.get_tractor_sources(source,background,threshold)
        self.tractor_object = self.get_tractor_object(self.tractor_image,self.tractor_sources)
       
        # Get zeroth order result
        self.zeroth_model = self.tractor_object.getModelImage(0)
        self.zeroth_chi = self.tractor_object.getChiImage(0)
        
        # Freeze images
        self.tractor_object.freezeParam('images')
        
        # Freeze Positions or put Gaussian priors:
        if np.isnan(fit_positions):
            # Forced Photometry
            for source_object in self.tractor_object.catalog:
                source_object.freezeParam('pos')
        else:
            # Semi-Forced Photometry
            for source_object in self.tractor_object.catalog:
                source_object.pos.addGaussianPrior('x',source_object.pos.x,fit_positions)
                source_object.pos.addGaussianPrior('y',source_object.pos.y,fit_positions)
        
        # Run optimization    
        self.optimize(self.tractor_object)
        
        self.model = self.tractor_object.getModelImage(0)
        self.chi = self.tractor_object.getChiImage(0)
        
        self.tractor_catalog = self.tractor_object.catalog
        
        # Dustin Suggestion to thaw everything at the end.
        self.tractor_object.thawAllRecursive()
        
        ################################################
        ## Setting up Fits Table to Input into Heasarc #
        ################################################
        
        self.x,self.y,self.flux = self.get_library(self.tractor_catalog)
        
        # Figure out what count rate is left in the residual image:
        self.resid_rate,self.resid_rate_err = self.calc_resid_photom(self.x,self.y,self.image,self.model,source.header["EXPOSURE"])
        
        # Filter other sources to match input to tractor:
        index = np.where((source.source_intensities > threshold*background.BACK_RATE) & (source.outside > 0.))[0]
        self.ra,self.dec = np.array(source.ra),np.array(source.dec)
        self.ra,self.dec = self.ra[index],self.dec[index]
        
        self.key = np.arange(source.catalog.shape[0])[index]
        
        self.detx,self.dety = source.detector_positions
        self.detx,self.dety = self.detx[index],self.dety[index]
        
        self.background_rate =  np.array(background.BACK_RATE[index])
        self.background_rate_err = np.array(background.BACK_RATE_ERR[index])
        
        self.edge = source.edge[index]
        
        self.exposure_time = source.header["EXPOSURE"]
        
        self.flux = np.array(self.flux)
        self.flux_err = np.sqrt(np.abs(np.copy(self.flux)) * self.exposure_time) / self.exposure_time
        
        self.total_rate = np.copy(self.flux) + self.background_rate;
        self.total_rate_err = np.sqrt(self.flux_err**2+self.background_rate_err**2)
        
        self.fitstable = Table([self.key,self.ra,self.dec,self.x,self.y,self.detx,self.dety,self.flux,
                                self.flux_err,self.total_rate,self.total_rate_err,
                                self.background_rate,self.background_rate_err,
                                self.resid_rate,self.resid_rate_err,self.edge],
                                names=('KEY','RA','DEC','PIX_X','PIX_Y','DETX','DETY',
                                       'TRACTOR_FLUX','TRACTOR_DFLUX',
                                      'TOTAL_RATE','TOTAL_RATE_ERR',
                                      'BACK_RATE','BACK_RATE_ERR',
                                      'RESID_RATE','RESID_RATE_ERR','EDGE_FLAG'))
        self.save_fitstable(fits_table_sname)
        self.update_header(fits_table_sname,source)
    
        if save_output:
            self.save_fits(fits_table_sname,source,self.model,self.chi,sky_background)
            
    def get_tractor_image(self,image,error,sky):
        # sky can equal NullSky() if you want to subtract the sky beforehand.
        # For the a18 field maria set ConstantSky(0.00418)
        return Image(data=image, invvar=np.ones_like(image) / (error**2),
                         psf=PixelizedPSF(self.psf),
                            wcs=NullWCS(), photocal=NullPhotoCal(),
                                sky=sky)
    
    
    def get_tractor_sources(self,source,background,threshold):
        X,Y = source.pixel_positions
        # Initial photometric guesses
        intensity = source.source_intensities
        # Photometry on the background
        back_rate = background.BACK_RATE
        # Remove objects outside image, I.E. in black diagonal area of fits
        outside = source.outside
        # Filter sources to give tractor based on some threshhold over the background. 
        index = np.where((intensity > threshold*back_rate) & (outside > 0.))[0]

        X2 = X[index]
        Y2 = Y[index]
        intensity2 = intensity[index]
        
        return [PointSource(PixPos(x,y),Flux(guess)) 
                        for x, y, guess 
                        in zip(X2,Y2,intensity2)]
    
    def get_tractor_object(self,tractor_image,tractor_sources):
        return Tractor([tractor_image], tractor_sources)
    
    def optimize(self,tractor_object):
            for i in range(100):
                dlnp,X,alpha = tractor_object.optimize()
                print ('dlnp', dlnp)
                # If Likelihood condition is reached, stop. 
                if dlnp < 1e-3:
                    var = tractor_object.optimize(variance=True,just_variance=True)
                    break   
    
    def get_library(self,tractor_catalog):
        PosX = []; PosY = []; Flux = [];
        for i in range(len(tractor_catalog)):
            r,d,f = tractor_catalog[i].getParams() 
            PosX.append(r); PosY.append(d); Flux.append(f)
        return PosX,PosY,Flux
    
    def calc_resid_photom(self,x,y,image,model,exposure):
        resid = image - model
        positions = [(x[i],y[i]) for i in range(len(x))]
        aperture = CircularAperture(positions,r=5.)
        phot_table = aperture_photometry(resid,aperture)
        resid_counts = phot_table['aperture_sum']
        resid_counts_err = np.sqrt(resid_counts * exposure) / exposure
        return resid_counts,resid_counts_err
    
    def save_fitstable(self,savename):
        self.fitstable.write(savename,format="fits",overwrite=True)
    
    def save_fits(self,savename,source,model,chi,background):
        fits.writeto(savename[:-5] + "_img.fits",source.data,source.header,overwrite=True)
        fits.writeto(savename[:-5] + "_model.fits",model,source.header,overwrite=True)
        fits.writeto(savename[:-5] + "_chi.fits",chi,source.header,overwrite=True)
        
        if type(background) == numpy.ndarray:
            fits.writeto(savename[:-5] + "_bkgd_subtracted_img.fits",self.image,source.header,overwrite=True)
        
    def update_header(self,fits_table_sname,source):
        hdr = fits.open(fits_table_sname)
        header = hdr[1].header
        header['TSTART'] = source.header['TSTART']
        header['TSTOP'] = source.header['TSTOP']
        header['EXPOSURE'] = source.header["EXPOSURE"]
        header['DEADC'] = source.header['DEADC']
        header['FRAMTIME'] = source.header['FRAMTIME']
        header['TELESCOP'] = source.header['TELESCOP']
        header['INSTRUME'] = source.header['INSTRUME']
        header['FILTER'] = source.header['FILTER']
        header['DETNAM'] = source.header['DETNAM']
        header['OBS_ID'] = source.header['OBS_ID']
        header['TARG_ID'] = source.header['TARG_ID']
        header['OBJECT'] = source.header['OBJECT']
        header['DATAMODE'] = source.header['DATAMODE']
        header['DATE-OBS'] = source.header['DATE-OBS']
        header['DATE-END'] = source.header['DATE-END']
        
        hdr.writeto(fits_table_sname, overwrite=True)
        hdr.close()