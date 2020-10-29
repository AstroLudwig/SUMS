# -*- Copyright (c) 2019, Bethany Ann Ludwig, All rights reserved. -*-
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
import astropy.units as u
from astropy.io import fits
import pandas as pd 
from astropy.wcs import WCS
import numpy as np
import wget
from photutils import aperture_photometry, SkyCircularAperture

################
# SWIFT Class  #
################

# Inputs:
#         Ra, Dec of the center of the Swift image you want.
#         Catalog of sources that may or may not be in that image.

class BlueSourceRetrieval:
    ## SMC
    #optical_catalog_filename = "../Optical/Zaritsky/ZaritskyCatalog.dat"
    ## LMC
    #optical_catalog_filename = "../Optical/Zaritsky/table1_lmc.dat"

    radius = 2.5 * u.arcsec

    distance = 60.6 * u.kpc

    origin = 0

    def __init__(self,uv_center_ra_deg,uv_center_dec_deg,
                 cutoff,r_arcmin,
                 optical_catalog_filename,
                 uvfilter='UVM2',
                 xdim=[0,300],
                 ydim=[0,300],
                 exposure_map = False):
        
        self.optical_catalog_filename = optical_catalog_filename
        self.cutoff = cutoff
        self.filter = uvfilter
        self.r_arcmin = r_arcmin
        # Swift Sky Coords
        self.uv_center_ra = uv_center_ra_deg
        self.uv_center_dec = uv_center_dec_deg

        # Swift File
        self.image_file = self.get_SWIFT_file(self.uv_center_ra,self.uv_center_dec,self.r_arcmin)
        self.data = self.image_file.data[ydim[0]:ydim[1],xdim[0]:xdim[1]]
        self.header = self.image_file.header
        self.cdelt = np.abs(self.header["CDELT1"])
        self.wcs = WCS(self.header)
        self.swift_x, self.swift_y = self.get_pixel_coordinates(self.uv_center_ra,self.uv_center_dec,self.wcs)
        
        # Correct Swift File for Exposure
        exp_time = self.header["EXPOSURE"]
        Sources.data = self.data/self.header["EXPOSURE"]
        Sources.source_intensities = Sources.source_intensities/exp_time
        
        
        # Optical Sources in Swift File, this is the current time constraint.  
        self.source_df = self.get_catalog_sources(self.data,self.wcs,self.optical_catalog_filename, self.cutoff)
        self.source_ra = self.source_df.Ra
        self.source_dec = self.source_df.Dec 
        self.source_skycoord = SkyCoord(self.source_ra,self.source_dec,
                                        frame="fk5",unit=(u.deg,u.deg))
        
        # Photutils, 5 arcsecond radius hard coded
        self.source_intensities = self.get_intensities(self.source_skycoord,self.data,
                                                           self.wcs,aperture_size = 5)

        # Optical Pixel Coordinates
        self.pixel_positions = self.get_pixel_coordinates(self.source_ra,self.source_dec,self.wcs)
        self.blue = self.source_df.Umag - self.source_df.Vmag
        
        if exposure_map:
            self.exposure_url, self.exposure_map = self.get_exposure_map(self.filter,self.uv_center_ra,self.uv_center_dec)

    # Pull Swift UVOT image out of the SkyView Catalog
    def get_SWIFT_file(self,ra_deg,dec_deg,r_arcmin):
        coord = SkyCoord(ra_deg,dec_deg,frame="fk5",unit=(u.deg,u.deg)) 
        img = SkyView.get_images(position=coord,survey=['UVOT '+self.filter+' Intensity'],radius=r_arcmin * u.arcmin)
        return img[0][0]

    # Convert RA/Dec into Pixel
    def get_pixel_coordinates(self,ra,dec,wcs):
        return wcs.all_world2pix(ra,dec,self.origin)

    # Read in Zaritsky Catalog while fixing some headers and choosing a cutoff.
    def get_optical_catalog(self,filename,cutoff):
        labels = ["Ra","Dec","Umag","e_Umag",
                     "Bmag","e_Bmag","Vmag","e_Vmag","Imag","e_Imag",
                     "Flag","Jmag","e_Jmag","Hmag","e_Hmag","Ksmag","e_Ksmag"]
        optical_catalog = pd.read_csv(filename,sep="\s+",names=labels)
        # Change from hourangle to angle
        optical_catalog.Ra = optical_catalog.Ra * 15
        if np.isfinite(cutoff):
            optical_catalog = optical_catalog.drop(optical_catalog[optical_catalog.Umag > cutoff].index)
            optical_catalog = optical_catalog.drop(optical_catalog[optical_catalog.Umag == 0. ].index)
        return optical_catalog

    # Gets Ra/Dec of the Four Corners of the UV Image
    def get_corners(self,data,wcs):
        n_rows, n_cols = np.shape(data)
        return wcs.all_pix2world([0,0,n_cols,n_cols],[0,n_rows,n_rows,0],self.origin)

    # Find all optical sources within the 4 corners of the uv image.
    def get_catalog_sources(self,swift_data,swift_wcs,optical_catalog_filename,cutoff):	
        optical_catalog = self.get_optical_catalog(optical_catalog_filename,cutoff)
        ra,dec = self.get_corners(swift_data,swift_wcs)
        objects = optical_catalog.loc[(optical_catalog.Ra < ra[0]) & (optical_catalog.Ra > ra[2])
                        & (optical_catalog.Dec > dec[0]) & (optical_catalog.Dec < dec[1])]
        return objects

    # Get an estimate for flux around optical source in uv image
    def get_intensities(self,skycoords, data, wcs, aperture_size):
        
        apertures = SkyCircularAperture(skycoords, r = aperture_size * u.arcsec)
        
        pix_apertures = apertures.to_pixel(wcs)
        
        photometry = aperture_photometry(data, pix_apertures)['aperture_sum']
        
        return photometry
        
    def get_exposure_map(self,uv_filter,ra,dec):
        url = ("https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?survey=swiftuvot%sexp&lon=%s&lat=%s&sampler=Default&projection=Tan&size=0.083331,0.083331&pixels=300&coordinates=J2000.0&Return=FITS" % (uv_filter,ra,dec))
        sname = "../ExposureMaps/%s_%s_%s_expmap.fits" % (uv_filter,ra,dec)
        file = wget.download(url,sname)
        return url,fits.open(file)
        

# Experimental. Trying to get more functionality for Maria.        
class get_meta(): 
    
    
    
    # Callable Functions  
    #############################
    def with_hdu(self,hdu,
                 cutoff=np.nan,
                 fits_origin=0,
                 aperture_size=2.5*2,
                 xdim=[0,300],
                 ydim=[0,300],
                 optical_catalog = "../Optical/Zaritsky/table1_lmc.dat",
                 astrometry=False):
        
        self.header = hdu.header
        
        if astrometry:
            self.cdelt = np.abs(self.header["CD1_1"])
        else:
            self.cdelt = np.abs(self.header["CDELT1"])
        
        self.wcs= WCS(self.header)
        
        self.filter = self.header["FILTER"]
        
        self.data = hdu.data[ydim[0]:ydim[1],xdim[0]:xdim[1]]
        
        self.optical_catalog_fname = optical_catalog
        
        self.catalog = self.get_catalog_sources(cutoff,fits_origin,xdim,ydim)
        
        self.catalog['KEY'] = np.arange(self.catalog.shape[0])
        
        self.pixel_positions = self.get_positions(hdu,self.catalog,fits_origin)
        
        self.detector_positions = self.get_det_positions(hdu,self.catalog,fits_origin)
        
        self.outside,self.edge=self.get_edge(self.detector_positions)
        
        self.pixel_positions[0] = self.pixel_positions[0] - xdim[0]
        self.pixel_positions[1] = self.pixel_positions[1] - ydim[0]
        
        
        self.ra = self.catalog.Ra
        self.dec = self.catalog.Dec
        
        self.source_intensities = np.array(self.get_intensities(self.catalog,cutoff,aperture_size))
        
        return self
    
    # Dependent Functions
    #####################
    
    # Open Zaritsky File and Reduce Number of Sources by some U Mag Cutoff. 
    def get_optical_catalog(self,cutoff):
        labels = ["Ra","Dec","Umag","e_Umag",
                     "Bmag","e_Bmag","Vmag","e_Vmag","Imag","e_Imag",
                     "Flag","Jmag","e_Jmag","Hmag","e_Hmag","Ksmag","e_Ksmag"]
        optical_catalog = pd.read_csv(self.optical_catalog_fname,sep="\s+",names=labels)
        # Change from hourangle to angle
        optical_catalog.Ra = optical_catalog.Ra * 15
        if np.isfinite(cutoff):
            optical_catalog = optical_catalog.drop(optical_catalog[optical_catalog.Umag > cutoff].index)
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
    def get_catalog_sources(self,cutoff,fits_origin,xdim,ydim):	
        optical_catalog = self.get_optical_catalog(cutoff)
        ra,dec = self.get_corners(fits_origin,xdim,ydim)
        objects = optical_catalog.loc[(optical_catalog.Ra < ra[0]) & (optical_catalog.Ra > ra[2])
                        & (optical_catalog.Dec > dec[0]) & (optical_catalog.Dec < dec[1])]
        return objects

    # Get an estimate for flux around optical source in uv image
    def get_intensities(self,catalog,cutoff,aperture_size):
        
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
        
    
#####################
# Future Dust Work  #
#####################

        # Started doing dust stuff here but got distracted
#         # Optical Catalog 
#         self.optical_catalog = self.get_optical_catalog(filename=self.optical_catalog_filename, cutoff=self.cutoff)
        
#         # Dereddening info
#         self.effective_wavelengths = {"UVW2" : 2085.7 * u.Angstrom,
#                                       "UVM2" : 2245.8 * u.Angstrom,
#                                       "UVW1" : 2684.1 * u.Angstrom, 
#                                       "U" : 3678.9 * u.Angstrom,
#                                       "B" : 4333.3 * u.Angstrom,
#                                       "V" : 5321.6 * u.Angstrom,
#                                       "I" : 8567.6 * u.Angstrom}
#         self.av = self.get_av_values(self.optical_catalog,Av_filename)"


#     def get_av_values(self,optical_catalog,av_fname):
#         # Get Av value by coordinate in av host stars fits file
#         # 1. Get pixel values in the av image from ra/dec coordinates in catalog
#         x,y = WCS(av_fname).all_world2pix(optical_catalog.Ra,optical_catalog.Dec,0)
#         # 2. To index, pixel must be an integer, round and then convert. 
#         x = [int(np.round(x)) for x in x]; y = [int(np.round(y)) for y in y] 
#         # 3. Add the Av values as a column in the catalog. 
#         optical_catalog["Av"] = fits.open(av_fname)[0].data[y,x]