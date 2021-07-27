# -*- Copyright (c) 2019, Bethany Ann Ludwig, All rights reserved. -*-
"""
NAME:
    Ogle SubField Finder
PURPOSE:
    My version of astroquery for OGLE DR3. 
    Inputs RA,DEC, and Galaxy
    Outputs Ogle Image
Notes: 
	
"""
import numpy as np 
from astropy.coordinates import SkyCoord
import pandas as pd 
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt 
from astropy.visualization import ZScaleInterval
from astroquery.skyview import SkyView
import astropy.units as u
import os
import wget
#from Plotter import plotter
################################################################


################################################################
################################################################


class Ogle :

    ################################################################
    # Defaults: Don't save fits of the image
    #           Search radius is half an arcsecond 
    
    def __init__(self,ra,dec,attempts=3,save_fits_name=None,save_png_name=None,
                 radius=0.5,next_file=False):
        
        if type(ra) == float or type(ra) == np.float64:
            # Assume units are degrees
            coordinate = SkyCoord(ra=ra,dec=dec,unit=u.deg)
        elif type(ra) == str:
            # Assume units are hourangle, degree. Seperated by colon          
            coordinate = SkyCoord(ra=ra,dec=dec,unit=(u.hourangle,u.deg))
        
        self.galaxy = self.which_galaxy(coordinate)
        
        self.all_fields = self.scrape_field_list(self.galaxy)
        
        self.star, self.image, self.field, self.subfield = self.locate_subfield(coordinate,
                                                                                self.galaxy,
                                                                                next_file,
                                                                                attempts)        
        self.X = self.star.X
        
        self.Y = self.star.Y
        
        if type(save_fits_name) == str:
             # If you want to save ogle field as fits
             self.get_field_image(self.galaxy,
                                  self.field,
                                  self.subfield).writeto(f"{save_fits_name}.fits" ,
                                                         overwrite = True)
            
        if type(save_png_name) == str:
            # If you want to save an image of the ogle field and/or another survey
             plotter(self.image,self.X,self.Y,
                     coordinate,self.field,self.subfield,
                     save_png_name = save_png_name,sky_view = True)

    ################################################################

    # For a given coordinate, which galaxy is it in? 

    def which_galaxy(self,source_coord):

        # From Simbad: LMC Coords:	05 23 34.6 -69 45 22, distance 49.97 kpc
        lmc = SkyCoord("05:23:34.6","-69:45:22",
                       unit=(u.hourangle,u.degree))
        # From Simbad: SMC Coords: 00 52 38 -72 48 01, distance 60.6 kpc
        smc = SkyCoord("00:52:38","-72:48:01",
                       unit=(u.hourangle,u.degree))

        # Calculate distance to each galaxy
        distances = np.array([source_coord.separation(galaxy).arcsecond for galaxy in [lmc,smc]])     
        
        if np.min(distances) == distances[0]:
            print("Target found in LMC")
            return "lmc"
        print("Target found in SMC")
        return "smc"

    ################################################################
    # Note, there is no, non-scraped version. 
    # If Ogle Goes Down, Just replace the URL by the Path to your fields.txt file.
    # For a galaxy, get a list of all the fields and their coordinates.

    def scrape_field_list(self, galaxy_string):

        return pd.read_csv('http://www.astrouw.edu.pl/ogle/ogle3/maps/%s/fields.txt' % galaxy_string ,
                           sep = '\s+', skiprows=[1])
    
    ################################################################
    def check_field_for_source(self,galaxy,source_coord,ref,ref_separation,next_file):
        
        min_distance_index = np.where(ref_separation==np.min(ref_separation))[0][0]

        if min_distance_index > (ref.shape[0] - 1):

            # There are two column combined into one in ref_seperation
            # Need to remove both coordinates from the same field 

            first_index  = min_distance_index - ref.shape[0]
            
            second_index =  min_distance_index 
            
            # Remove both coordinates from subfield distance list
            ref_separation[first_index] = 1e5
            
            ref_separation[second_index] = 1e5 
        
        else:

            first_index = min_distance_index
            
            second_index = min_distance_index + ref.shape[0]
    
            
            ref_separation[first_index] = 1e5 
            
            ref_separation[second_index] = 1e5 
              
        # Get the subfield for the coordinate you are closest to
        field = ref.iloc[first_index]['field']
        
        subfield = ref.iloc[first_index]['sub_field_index']
        
        field_map = self.scrape_field_map(galaxy,field,subfield)
        
        # Check that star is there
        subfield_coord = SkyCoord(field_map.RA,field_map.DEC, 
                                            unit=(u.hourangle,u.degree))
        
        subfield_distances = source_coord.separation(subfield_coord).arcsecond
        
        star_distance =  np.min(subfield_distances)
        
        star_index = np.where(subfield_distances == star_distance)[0][0]
        
        star = field_map.iloc[star_index]
        
        print(f"Chosen Star is {str(star_distance)} from source in {field}.{subfield}")
        
        if star_distance < 1  and next_file == False:
            
            print("Finder Chart Aquired")
            
            image = self.scrape_field_image(galaxy,field,subfield)
            
            return [star,image,field,subfield]
        
        print(f"Chosen Star not close enough to subfield. Recalculating")
        
        return [ref_separation]
    
    # For a given coordinate, which field is it in? 
    def locate_subfield(self,source_coord,galaxy,next_file,attempts):
        df = pd.read_csv("Subfield_ref.csv")
        
        ref = df[df.galaxy==galaxy]
        
        all_coords = SkyCoord(np.append(ref.ra_1,ref.ra_2),
                              np.append(ref.dec_1,ref.dec_2),
                              unit=(u.hourangle,u.deg))
      
        ref_separation = source_coord.separation(all_coords).arcsecond
        
        
        for i in range(attempts):
    
            result = self.check_field_for_source(galaxy,source_coord,ref,
                                                   ref_separation,next_file)
     
            if len(result) > 1: 
           
                return result
      
            ref_separation = result[0]
        
        print(f"{attempts} tries, no field found")
        
        
    def which_field(self,ra,dec):
        # Compute distances to field
        field_ra,field_dec = self.convert_to_radian(self.all_fields.RA,self.all_fields.DEC)
        angles = self.compute_angular_distance(ra , dec,field_ra,field_dec) 

        return np.array(self.all_fields[angles==np.min(angles)]["Field"])[0].lower()

        
    ################################################################

    # Compute angular distance between two sources
    # Sometimes it's faster to do the math yourself
    # ra/dec need to be radian 

    def compute_angular_distance(self,ra1,dec1,ra2,dec2):     

        return np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2))
    
    ################################################################

    # Convert ra/dec strings to radians

    def convert_to_radian(self,ra_hms,dec_dms):

        # For a pandas data frame
        if type(ra_hms) == pd.core.series.Series: 
            h,m,s = np.array(ra_hms.str.split(":",expand=True)).T.astype(float)
            d,m_,s_ = np.array(dec_dms.str.split(":",expand=True)).T.astype(float)
        # For an individual source.     
        else:
            h,m,s = np.array(ra_hms.split(":")).astype(float)
            d,m_,s_ = np.array(dec_dms.split(":")).astype(float)

        # Calculate degrees, convert to radians
        ra = 15 * (h + m/60 + s/3600) * np.pi / 180
        dec = d/np.abs(d) * (np.abs(d) + m_/60 + s_/3600) * np.pi / 180

        return ra,dec
    
    ################################################################
    # Get text file of all stars in a subfield 
    
    # Downloaded Data Mode
    def get_field_map(self,galaxy,field,subfield):
        names = ["RA","DEC","X","Y","U1","U2","U3","U4","U5","U6","U7","U8","U9"]
        
        path=f'{galaxy}/maps/{field.lower()}.{subfield}.map.bz2'
        
        if os.path.exists(path):
            return pd.read_csv(path,sep='\s+',header=None,names=names)
        print("Please fix path to downloaded maps in get_field_map.")
        print("Current Path Set To: ",path)
    
    # Web Scrape Mode 
    def scrape_field_map(self,galaxy,field,subfield):
        names = ["RA","DEC","X","Y","U1","U2","U3","U4","U5","U6","U7","U8","U9"]
    
        url = 'http://www.astrouw.edu.pl/ogle/ogle3/maps/%s/maps/%s.%s.map.bz2' % (galaxy, field.lower(), subfield) 
        print(url)
        try:
            return pd.read_csv(url,sep='\s+',header=None,names=names)
        except:
            print("Issue with scraping map from OGLE site. Please check scrape_field_map")
         
    ################################################################
    # For a given galaxy, field, and subfield, return the image
    
    # Web Scrape Mode 
    def scrape_field_image(self,galaxy,field,subfield):
        url = 'http://www.astrouw.edu.pl/ogle/ogle3/maps/%s/ref_images/%s.i.%s.fts.bz2' % (galaxy,field.lower(),subfield)
       
        try:
            return fits.open(url)[0]
    
        except:
            print("Issue with scraping map from OGLE site. Please check scrape_field_map")
            
    # Downloaded Data Mode
    def get_field_image(self,galaxy,field,subfield):
        path = f'{galaxy}/ref_images/{field.lower()}.i.{subfield}.fts.bz2'
        
        if os.path.exists(path):
            return fits.open(path)[0]    
        print("Please fix path to downloaded images in get_field_image.")
            

################################################################
class Brute_Force : 
    
    def __init__(self,ra,dec,galaxy,save_png_name,mode="downloaded"):
        
        if type(ra) == float :
            # Assume units are degrees
            coordinate = SkyCoord(ra=ra,dec=dec,unit=u.deg)
            
        elif type(ra) == str :
            # Assume units are hourangle, degree. Seperated by colon
            # Currently not supporting hms, dms strings. 
            
            coordinate = SkyCoord(ra=ra,dec=dec,unit=(u.hourangle,u.deg))
    
        self.field,self.subfield = self.map_loop(galaxy,coordinate,mode)
            
        if mode == "downloaded" :
            self.image = self.get_field_image(galaxy,self.field,self.subfield)
                
        if mode == "scrape" :    
            self.image = self.scrape_field_image(galaxy,self.field,self.subfield)
        
        if type(save_png_name) == str :
            plotter(self.image,self.X,self.Y,
                    coordinate,self.field,self.subfield,
                    save_png_name=save_png_name,sky_view=True)
    
    # Download Data Mode - Map
    def get_field_map(self,galaxy,field,subfield):
        names = ["RA","DEC","X","Y","U1","U2","U3","U4","U5","U6","U7","U8","U9"]
        
        path=f'{galaxy}/maps/{field.lower()}.{subfield}.map.bz2'  
        
        if os.path.exists(path):
            return pd.read_csv(path,sep='\s+',header=None,names=names)
        
        print("Please fix path to downloaded maps in get_field_map.")
        print("Current Path Set To: ",path)
    
    # Web Scrape Mode - Map
    def scrape_field_map(self,galaxy,field,subfield):
        names = ["RA","DEC","X","Y","U1","U2","U3","U4","U5","U6","U7","U8","U9"]
    
        url = 'http://www.astrouw.edu.pl/ogle/ogle3/maps/%s/maps/%s.%s.map.bz2' % (galaxy, field, subfield) 
    
        try:
            return pd.read_csv(url,sep='\s+',header=None,names=names)
        except:
            print("Issue with scraping map from OGLE site. Please check scrape_field_map")
            
    # Loop through all maps 
    def map_loop(self,galaxy,coordinate,mode):
        
        for j in range(100,141):
            
            for i in range(1,9):
                
                if mode == "downloaded":
                    field_map = self.get_field_map(galaxy,f"{galaxy.upper()}{j}",f"{i}")
                    
                if mode == "scrape":
                    field_map = self.scrape_field_map(galaxy,f"{galaxy.upper()}{j}",f"{i}")
                
                field_coord = SkyCoord(field_map.RA,
                                          field_map.DEC, 
                                          unit=(u.hourangle,u.degree))
                
                field_distances = coordinate.separation(field_coord).arcsecond

                star_distance =  np.min(field_distances)
                
                print(f"{galaxy} {str(j)}.{(i)} closest object: {str(star_distance)} AS")
                
                if star_distance < 1 : 
                    
                    return str(j),str(i)
                
                
    ################################################################
    # For a given galaxy, field, and subfield, return the image
    
    # Web Scrape Mode 
    def scrape_field_image(self,galaxy,field,subfield):
        url = 'http://www.astrouw.edu.pl/ogle/ogle3/maps/%s/ref_images/%s.i.%s.fts.bz2' % (galaxy,field,subfield)
       
        try:
            return fits.open(url)[0]
    
        except:
            print("Issue with scraping map from OGLE site. Please check scrape_subfield_map")
            
    # Downloaded Data Mode
    def get_field_image(self,galaxy,field,subfield):
        path = f'{galaxy}/ref_images/{field.lower()}.i.{subfield}.fts.bz2'
        
        if os.path.exists(path):
            return fits.open(path)[0]    
        print("Please fix path to downloaded images in get_field_image.")
################################################################

class Subfield_Reference :
    
    ################################################################
    
    def __init__(self, galaxy, field, sub_field_index) :
        self.galaxy = galaxy
        self.field = field
        self.sub_field_index = sub_field_index

        self.ra_1 = None
        self.ra_2 = None
        self.dec_1 = None
        self.dec_2 = None
        
        names = ["RA","DEC","X","Y","U1","U2","U3","U4","U5","U6","U7","U8","U9"]
        dim = [2180,4176]
        mid_x = dim[0] / 2
        first_y = dim[1] / 3
        second_y = 2 * dim[1] / 3
        
        url = self.get_url()
        df = pd.read_csv( url,sep='\s+',header=None,names=names)
        self.set_first_field_coordinate(df, mid_x, first_y)
        self.set_second_field_coordinate(df, mid_x, second_y)

    ################################################################
        
    def get_field_coordinate(df,x,y):
        pixelRange = 20
        nearby = np.where(np.isclose(df.X,x,atol=pixelRange) & np.isclose(df.Y,y,atol=pixelRange) )[0]
        
        if len(nearby) == 0:
            # Move up dy pixels and try again
            dy = 5; sign = 1
            go_down = False
            while len(nearby) == 0:
                if go_down :
                    nearby = np.where(np.isclose(df.X,x,atol=pixelRange) & np.isclose(df.Y,y-dy,atol=pixelRange) )[0]
                    dy += 5
                    go_down = False
                else :
                    nearby = np.where(np.isclose(df.X,x,atol=pixelRange) & np.isclose(df.Y,y+dy,atol=pixelRange) )[0]
                    go_down = True
                
            if go_down != True: 
                sign = -1

            print("Adjusted by %s " % (sign * dy))
        
        if len(nearby == 1):
            nearby = nearby[0]
            return df.RA[nearby], df.DEC[nearby]
                

        if len(nearby) > 1: 
            dist = np.zeros(len(nearby))

            for i,index in enumerate(nearby):
                dist[i] = np.sqrt((df.iloc[index].X - x)**2 + (df.iloc[index].Y - y)**2)

            nearby = nearby[dist == np.min(dist)][0]
            print(nearby)
        return df.RA[nearby], df.DEC[nearby]
        
    ################################################################
        
    def get_url(self) :
        return 'http://www.astrouw.edu.pl/ogle/ogle3/maps/%s/maps/%s.%s.map.bz2' % (self.galaxy , self.field.lower(), self.sub_field_index)
    
    ################################################################
    
    def set_first_field_coordinate(self, df, x, y) :
        self.ra_1, self.dec_1 = Subfield_Reference.get_field_coordinate(df,x,y)
    
    ################################################################
    
    def set_second_field_coordinate(self, df, x, y) :
        self.ra_2, self.dec_2 = Subfield_Reference.get_field_coordinate(df,x,y)
        
    ################################################################

class check_field : 
    
    def __init__(self,galaxy,field,subfield,ra,dec,save_png_name=None):  
        
        coordinate = SkyCoord(ra,dec,unit=u.deg)
        
        self.map = self.get_field_map(galaxy,field,subfield)
        
        field_coord = SkyCoord(self.map.RA, self.map.DEC, unit=(u.hourangle,u.degree))
                
        field_distances = coordinate.separation(field_coord).arcsecond

        self.star_distance =  np.min(field_distances)
        
        index = np.where(field_distances == self.star_distance)
        
        self.star = self.map.iloc[index]
        
        self.hdu = self.get_field_image(galaxy,field,subfield)
        
        self.image = self.hdu[0].data
        
        if type(save_png_name) == str : 
            
            vmin,vmax = ZScaleInterval().get_limits(self.image)
            n = 100

            plt.figure(figsize=(10,10))

            plt.imshow(self.image,vmin=vmin,vmax=vmax)
            plt.scatter(self.star.X.iloc[0],self.star.Y.iloc[0],c='r',s=5)
            plt.xlim(self.star.X.iloc[0] - n,self.star.X.iloc[0] + n)
            plt.ylim(self.star.Y.iloc[0] - n,self.star.Y.iloc[0] + n)
            plt.savefig(save_png_name)
        
    def get_field_image(self,galaxy,field,subfield):
        path = f'{galaxy}/ref_images/{field.lower()}.i.{subfield}.fts.bz2'
        
        return fits.open(path)
        
    # Download Data Mode - Map
    def get_field_map(self,galaxy,field,subfield):
        names = ["RA","DEC","X","Y","U1",
                 "U2","U3","U4","U5","U6","U7","U8","U9"]
        
        path=f'{galaxy}/maps/{field.lower()}.{subfield}.map.bz2'  
        
        if os.path.exists(path):
            return pd.read_csv(path,sep='\s+',header=None,names=names)
        
        print("Please fix path to downloaded maps in get_field_map.")
        print("Current Path Set To: ",path)
# ######################################    

