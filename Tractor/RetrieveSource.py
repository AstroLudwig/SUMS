# -*- Copyright (c) 2020, Bethany Ann Ludwig, All rights reserved. -*-
"""
NAME:
    Photometry Retrieve Source 
PURPOSE:
    Given a swift field, find the Zaritsky sources in them and do an initial photometric guess.
    get_meta() is a class. You give it an ra/dec and it gives you the SWIFT field
    with the objects in them. It includes the coordinates, regions, catalog information,
    and intial guesses for flux. The entire object could be saved with pickle. 
Notes: 
"""

import astropy.units as u
import numpy as np
import pandas as pd 

from astropy.io import fits
from astropy.wcs import WCS
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord, Angle
from photutils import aperture_photometry, SkyCircularAperture

################
# SWIFT Class  #
################

# Inputs:
# hdu: 
#              Name of the file itself, opened with fits
# Umag_cutoff: 
#              First cutoff, how bright do sources need to be in Umag to be considered
# Bmag_cutoff: 
#              Second cutoff, before removing sources with no Umag, check if Bmag is present + above a threshhold
# fits_origin:
#              Most files start at 0, some start at 1. 
# aperture_size:
#              How big to make the photometric aperture (the circle around the source)
# xdim:
#              Pixel coordinates for the xrange you want to run retrieve source functions on
# ydim:
#              Pixel coordinates for the yrange you want to run retrieve source functions on
# optical_catalog: 
#              For speed, pick an optical catalog based on what galaxy you are in. 

# Returns: 
# A class object with 
# - general information about the file (header,exposure_time,cdelt,wcs,filter)
# - The patch of sky or data you are running on.
# - An optical catalog of the sources in that patch
# - UV photometry based on optical astrometry, meant to be a first guess
# - Some detector positions, flags for if region is off the image. 

class get_meta(): 
    
    
    
    # Callable Functions  
    #############################
    def with_hdu(self,hdu,
                 Umag_cutoff=np.nan,
                 Bmag_cutoff=np.nan,
                 fits_origin=0,
                 aperture_size=2.5*2,
                 xdim=[0,300],
                 ydim=[0,300],
                 optical_catalog="../Data/Optical/table1_smc.dat"):
        
        # Get general information from the header 
        self.header = hdu.header
        
        self.exposure_time = self.header["EXPOSURE"]
        
        self.cdelt = np.abs(self.header["CD1_1"])

        self.wcs= WCS(self.header)
        
        self.filter = self.header["FILTER"]
        
        # Get data based on patch of sky defined in xdim ydim and divide by exposure time to get count rate
        
        self.data = hdu.data[ydim[0]:ydim[1],xdim[0]:xdim[1]] / self.exposure_time
        
        
        # Get optical sources for that patch of UV sky
        
        self.optical_catalog_fname = optical_catalog
        
        self.catalog = self.get_catalog_sources(Umag_cutoff,Bmag_cutoff,fits_origin,xdim,ydim)
        
        self.catalog['KEY'] = np.arange(self.catalog.shape[0])
        
        self.pixel_positions = self.get_positions(hdu,self.catalog,fits_origin)
        
        self.pixel_positions[0] = self.pixel_positions[0] - xdim[0]
        
        self.pixel_positions[1] = self.pixel_positions[1] - ydim[0]
        
        self.ra = self.catalog.Ra
        
        self.dec = self.catalog.Dec
        
        
        # Do some initial photometry to get an initial guess 
        
        self.source_intensities = np.array(self.get_intensities(self.catalog,aperture_size))
        
        
        # Get position with some correction from the detector
        
        self.detector_positions = self.get_det_positions(hdu,self.catalog,fits_origin)
        
        self.outside,self.edge=self.get_edge(self.detector_positions)
             
        
        return self
    
    # Dependent Functions
    #####################
    
    # Open Optical Catalog and Reduce Number of Sources by Some Cutoff. 
    def get_optical_catalog(self,Umag_cutoff,Bmag_cutoff):
        # Read in the optical catalog 
        labels = ["Ra","Dec","Umag","e_Umag",
                     "Bmag","e_Bmag","Vmag","e_Vmag","Imag","e_Imag",
                     "Flag","Jmag","e_Jmag","Hmag","e_Hmag","Ksmag","e_Ksmag"]
        optical_catalog = pd.read_csv(self.optical_catalog_fname,sep="\s+",names=labels)
        
        # For optical catalog: hour angle degrees to regular degrees
        optical_catalog.Ra = optical_catalog.Ra * 15
        
        if np.isfinite(Umag_cutoff):
            print(f'Reducing using Umag Cutoff{Umag_cutoff}')
            optical_catalog = optical_catalog.drop(optical_catalog[optical_catalog.Umag > Umag_cutoff].index)
            
            if np.isfinite(Bmag_cutoff):
                print(f'Reducing using Bmag Cutoff{Bmag_cutoff}')
                # If we have a not nan Bmag cut off, only drop rows with no Umag if Bmag is above a certain cut off
                optical_catalog = optical_catalog.drop((optical_catalog[optical_catalog.Umag == 0. ].index) & 
                                                       (optical_catalog[optical_catalog.Bmag > Bmag_cutoff].index))
                return optical_catalog
        else:  
            # Otherwise just drop things that don't have Umag.
            optical_catalog = optical_catalog.drop(optical_catalog[optical_catalog.Umag == 0. ].index)
            return optical_catalog

    # Gets Ra/Dec of the Four Corners of the UV Image
    def get_corners(self,fits_origin,xdim,ydim):
        n_rows, n_cols = np.shape(self.data)
        n_rows = n_rows + ydim[0]
        n_cols = n_cols + xdim[0]
        return WCS(self.header).all_pix2world([xdim[0],xdim[0],n_cols,n_cols],
                                      [ydim[0],n_rows,n_rows,ydim[0]],fits_origin)

    # Find all optical sources within the 4 corners of the uv image.
    def get_catalog_sources(self,Umag_cutoff,Bmag_cutoff,fits_origin,xdim,ydim):
        optical_catalog = self.get_optical_catalog(Umag_cutoff,Bmag_cutoff)
        ra,dec = self.get_corners(fits_origin,xdim,ydim)
        objects = optical_catalog.loc[(optical_catalog.Ra < ra[0]) & (optical_catalog.Ra > ra[2])
                        & (optical_catalog.Dec > dec[0]) & (optical_catalog.Dec < dec[1])]
        return objects

    # Get an estimate for flux around optical source in uv image
    def get_intensities(self,catalog,aperture_size):
        
        positions = SkyCoord.from_pixel(self.pixel_positions[0],self.pixel_positions[1],self.wcs)
        
        apertures = SkyCircularAperture(positions, r = aperture_size * u.arcsec)
        
        pix_apertures = apertures.to_pixel(self.wcs)
                
        photometry = aperture_photometry(self.data, pix_apertures)['aperture_sum']
        
        return photometry
    
    # Get X,Y positions
    def get_positions(self,hdu,catalog,fits_origin):
        
        return WCS(hdu).all_world2pix(catalog.Ra,catalog.Dec,fits_origin)
    
    
    # Get DETX,DETY positions
    def get_det_positions(self,hdu,catalog,fits_origin):
        
        px,py = WCS(hdu).all_world2pix(catalog.Ra,catalog.Dec,0)
        mx,my = WCS(hdu,key='D').all_pix2world(px,py,0)
        detx = mx/0.009075 + 1100.5
        dety = my/0.009075 + 1100.5
        
        return [detx,dety]
    
    def get_edge(self,positions):
        '''This figures out if it is within an approximation of 5" of the edge
        It also figures out if it is outside the data area itself (so we can drop from tractor)
        
        
        Outside = -99 means it is outside the data region
        Edge = -99 means it is outside a region that is 5" in from the edge (so a super-set of outside)
        '''
        
        detx,dety = positions
        
        detx2 = (detx - 1100.5) *0.009075
        dety2 = (dety - 1100.5) *0.009075
        outside = np.zeros(len(detx2))+1. #This says it is outside the data area.
        edge = np.zeros(len(detx2))+1. #This says it is within 5" of edge.
        edge_space = 0.09 #This is 5"

        # Define various edges in detector coordinates:
        tl = 9.128 #y
        tc = 9.303 #y
        tr = 9.040 #y
        
        bl = -8.989 #y
        br = -9.178 #y 
        bc = -9.246 #y
        
        lt = -8.988 #x
        lb = -8.713 #x
        lc = -9.018 #x
        
        rt = 8.810 #x
        rb = 8.891 #x
        rc = 9.012 #x
        
        #Define the eight lines based on these points:
        
        #Top Left:
        m_tl = (tc-tl)/(0.-lt)
        b_tl = tc
        
        #Top Right:
        m_tr = (tr-tc)/(rt)
        b_tr = tc
        
        #Right Top:
        m_rt = (tr)/(rt-rc)
        b_rt = -m_rt*rc
        
        #Right Bottom:
        m_rb = (br)/(rb-rc)
        b_rb = -m_rb*rc
        
        #Bottom Right:
        m_br = (br-bc)/(rb)
        b_br = bc
        
        #Bottom Left:
        m_bl = (bc-bl)/(0.-lb)
        b_bl = bc
        
        #Left Bottom:
        m_lb = (bl)/(lb-lc)
        b_lb = -m_lb*lc
        
        #Left Top:
        m_lt = (tl)/(lt-lc)
        b_lt = -m_lt*lc
        
        #Series of eight conditions for the edges for each point:
        
        for i in range(len(detx2)):
            
            #Tops:
            if dety2[i] > (detx2[i]*m_tl + b_tl): outside[i] = -99.
            if dety2[i] > (detx2[i]*m_tl + b_tl - edge_space): edge[i] = -99.
                
            if dety2[i] > (detx2[i]*m_tr + b_tr): outside[i] = -99.
            if dety2[i] > (detx2[i]*m_tr + b_tr - edge_space): edge[i] = -99.
             
            #Bottoms:
            if dety2[i] < (detx2[i]*m_bl + b_bl): outside[i] = -99.
            if dety2[i] < (detx2[i]*m_bl + b_bl + edge_space): edge[i] = -99.
                
            if dety2[i] < (detx2[i]*m_br + b_br): outside[i] = -99.
            if dety2[i] < (detx2[i]*m_br + b_br + edge_space): edge[i] = -99.
            
            #Rights:
            if detx2[i] > ((dety2[i]-b_rt)/m_rt): outside[i] = -99.
            if detx2[i] > ((dety2[i]-b_rt)/m_rt - edge_space): edge[i] = -99.
                
            if detx2[i] > ((dety2[i]-b_rb)/m_rb): outside[i] = -99.
            if detx2[i] > ((dety2[i]-b_rb)/m_rb - edge_space): edge[i] = -99.
            
            #Lefts:
            if detx2[i] < ((dety2[i]-b_lt)/m_lt): outside[i] = -99.
            if detx2[i] < ((dety2[i]-b_lt)/m_lt + edge_space): edge[i] = -99.
                
            if detx2[i] < ((dety2[i]-b_lb)/m_lb): outside[i] = -99.
            if detx2[i] < ((dety2[i]-b_lb)/m_lb + edge_space): edge[i] = -99.
        
        return outside,edge
    
    
    
    
        
