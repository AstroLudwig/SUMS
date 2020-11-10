import numpy as np 
import PSF_Models as Models
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
import scipy.integrate as integrate
from scipy.integrate import quad

class psf_fit():
    
    
    def __init__(self,pixel_per_arsecond,uvfilter,width,save=False,cog_file="../Data/Calibration/swureef20041120v104.fits",fwhm=2.5):
    
        
        if uvfilter == "UVM2" or uvfilter == "uvm2": 
            self.filter_n = 6
            
        if uvfilter == "UVW2" or uvfilter == "uvw2": 
            self.filter_n = 5
            
        if uvfilter == "UVW1" or uvfilter == "uvw1": 
            self.filter_n = 4
            
        self.CoG = fits.open(cog_file)
        
        self.R = self.CoG[self.filter_n].data.RADIUS
        
        self.Frac = self.CoG[self.filter_n].data.REEF
        
        self.FWHM = fwhm # Arcseconds
        ## Fit a Cumulative Integrated Moffat to the Data

        index_to_5 = np.where(np.isclose(self.R,5))[0][0] + 1

        Moff_R, Moff_F, Moff_popt = self.fit(self.R[:index_to_5],self.Frac[:index_to_5],self.cumul_moff,[2])

        Gauss_R, Gauss_F, Gauss_popt = self.fit(self.R[index_to_5 - 1 :],self.Frac[index_to_5 - 1 :],self.cumul_gauss,[1,1,1])
        
        ## Use fitted parameters to get psf 

        pMoff_F = self.moff(Moff_R,*Moff_popt)

        pGauss_F = self.two_gauss(Gauss_R,*Gauss_popt)

        self.r = np.append(Moff_R,Gauss_R)
        self.psf = np.append(pMoff_F,pGauss_F)

        # ///////////////////////
        # Create Pixelated PSF
        #////////////////////////

        # Decide how big you want the range to be. 
        n = width # This is an n arcsecond psf

        # Decide arcsecond to pixel conversion
        to_pix = pixel_per_arsecond #  Pixel / ArcSecond

        # Needs to be odd and an integer.
        size = n * to_pix
        
        print(size)

        if size - np.round(size) != 0:
            size = int(np.ceil(size))
        if size % 2 != 1:
            size = int(size + 1)
        if size - np.round(size) == 0:
            size = int(size)
            
        print(size)
        print("Scaling is "+str(1/to_pix)+" ArcSeconds per Pixel")
        print("Image needs to be "+str(size)+" pixels to be "+str(n)+" ArcSeconds in size.")	

        # Decide how fine you want to sample it.
        sample_rate = .1; expand_n = int(size * 1 / sample_rate)

        # Get a larger psf that samples it at a finer resolution
        large_psf = np.zeros(expand_n*expand_n).reshape(expand_n,expand_n)
        x0 = np.floor(expand_n/2); y0 = np.floor(expand_n/2)
        for i in range(expand_n):
            for j in range(expand_n):
                distance = sample_rate * np.sqrt((i-y0)**2+(j-x0)**2)
                if distance < 5:
                    large_psf[i,j] = Models.moff(distance, *Moff_popt)
                if distance > 5: 
                    large_psf[i,j] = Models.two_gauss(distance, *Gauss_popt)

        # Regrid by summing things back into the original resolution
        print(size)
        pixel_psf = np.zeros(size*size).reshape(size,size)
        shrink = 1 / sample_rate
        for i in range(size):
            for j in range(size):
                i_start = int(i*shrink) ; i_stop = int(i*shrink + shrink)
                j_start = int(j*shrink) ; j_stop = int(j*shrink + shrink)
                pixel_psf[i,j] = np.sum(large_psf[i_start:i_stop,j_start:j_stop])

        # Normalize the psf
        inner_sum = 0
        x0 = np.floor(n/2); y0 = np.floor(n/2)
        for i in range(n):
            for j in range(n):
                distance = np.sqrt((i-y0)**2+(j-x0)**2)
                if distance < 5 * to_pix:
                    inner_sum += pixel_psf[i,j]

        pixel_psf = pixel_psf / inner_sum
        print("Sum Before Normalization ",inner_sum)

        # Check that it was normalized
        inner_sum = 0
        x0 = np.floor(n/2); y0 = np.floor(n/2)
        for i in range(n):
            for j in range(n):
                distance = np.sqrt((i-y0)**2+(j-x0)**2)
                # Convert to pixel distances
                if distance < 5 * to_pix:
                    inner_sum += pixel_psf[i,j]
        print("Sum After Normalization ",inner_sum)
        
        # Saving PSF as Float32 to be Friendly with Dustin's Code
        self.psf = np.array(pixel_psf,np.float32)
        
        if save:
            np.savetxt("PSF/PSF_"+str(n)+"x"+str(n)+"_arcsec_image_"+str(1/to_pix)+"_arcsec_per_pixel_normalized.txt",pixel_psf)
        
    def fit(self,x,y,func,initial_guess):
        popt,pcov = curve_fit(func,x,y,p0=initial_guess)
        X = np.arange(np.min(x),np.max(x),0.001)
        return X, func(X,*popt),popt
        
        
    def cumul_moff(self,r,B):
        a = self.FWHM / (2 * np.sqrt(2**(1/B) - 1))
        top = (1+(25/a**2))**B * (1 + (r**2/a**2))**(-B) * (-r**2 + (-1 + (1+(r**2/a**2))**B)*a**2)

        bottom = (-25 + (-1 + (1+(25/a**2))**B) * a**2)

        return top / bottom 
        
    def cumul_gauss(self,r,sigma,A1,A2):     
        # Assuming r0 = 0   #  sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

        return (A1 + A2) * (np.exp(-25/(2 *sigma**2))-np.exp(-r**2 / (2 *sigma**2))) * np.sqrt(2 * np.pi) * sigma + 1
    
    def moff(self,r,Beta):
        alpha = self.FWHM / (2 * np.sqrt(2**(1/Beta) - 1))
    
        return (2 * (Beta -1) / alpha **2) * (1 + r**2/alpha**2)**(-Beta)

 
    def two_gauss(self,r,sigma,A1,A2):
        r0 = 0 

        #sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

        gauss1 = (A1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(r-r0)**2 / (2 *sigma**2))
        gauss2 = (A2 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(r-r0)**2 / (2 *sigma**2))

        return gauss1 + gauss2
