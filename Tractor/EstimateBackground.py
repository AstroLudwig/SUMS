from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils import make_source_mask
from photutils import Background2D, SExtractorBackground
import numpy as np
from photutils import CircularAperture
from photutils import aperture_photometry

class BkgdEstimator:

    def __init__(self,source,n_pix):
        '''
        param: source: the output from Retrive Source. Currently assumes that data was divided by exposure time before input (such that the units of the image are counts/s)
        param: n_pix: the size of the 'subimage' for the 2D background estimator
        '''
        
        exposure = source.header["EXPOSURE"]
        data = source.data
        
        # Initial Masks for data:
        self.mask = make_source_mask(data, snr=5, npixels=5, dilate_size=11)
        self.mask_rotate =  (data == 0)
        mask=self.mask+self.mask_rotate
        self.mean, self.median, self.std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
        print('Masked Sigma Clipped Stats Original Data:',self.mean,self.median,self.std)
        
        # Sigma clip bright obvious things to avoid biasing the background estimate
        # TO DO: New photutils need to change sigma to snr
        self.sigma_clip = SigmaClip(sigma=3) 
        # Apply the SExtractor algorithm to our estimation
        self.bkg_estimator = SExtractorBackground() 
        self.bkg = Background2D(
            data, (n_pix[0],n_pix[1]),
            mask=self.mask+self.mask_rotate,
            filter_size=(3, 3),
            exclude_percentile=75,
            sigma_clip=self.sigma_clip,
            bkg_estimator=self.bkg_estimator)
        
        self.background = self.bkg.background #* ~self.mask_rotate
        self.background_rms = self.bkg.background_rms #* ~self.mask_rotate
        
        # Cacluate the remaining stats on the background after subtracting this off.
        datasub = data - self.background
        self.mask2 = make_source_mask(datasub, snr=5, npixels=5, dilate_size=11)
        self.mask2b = self.mask2+self.mask_rotate
        self.mean2, self.median2, self.std2 = sigma_clipped_stats(datasub, sigma=3.0, mask=self.mask2b)
        print('Masked Sigma Clipped Stats Back Sub Data:',self.mean2,self.median2,self.std2)
        
        # Calculate the error image for input into tractor:
        error_image = np.sqrt((self.background_rms*exposure)**2 +(data*exposure))/exposure
        error_image[error_image == 0.0] = 0.0001
        error_image[np.argwhere(np.isnan(error_image))] = 0.0001
        self.error_image = error_image
        
        # Calculate the background count rate and count rate error with a 5" aperture at each source location. Needed for calibration
        X,Y = source.pixel_positions
        positions = [(X[i],Y[i]) for i in range(len(X))]
        aperture = CircularAperture(positions,r=5.)
        phot_table = aperture_photometry(self.background,aperture)
        self.BACK_RATE = phot_table['aperture_sum']
        self.BACK_RATE_ERR = np.sqrt(self.BACK_RATE * exposure) / exposure
        