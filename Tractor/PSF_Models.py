import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import quad
from scipy.optimize import curve_fit
FWHM = 2.5 # Arcseconds

def mean(x,y):
    n = len(x)                         
    mean = sum(x*y)/n
    return mean


def sigma(x,y):
    n=len(x)
    sigma = sum(y*(x-mean(x,y))**2)/n    
    return sigma


def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def gaussgauss(x,a,x0,sigma,aa,xx0,ssigma):
    return gauss(x,a,x0,sigma) + gauss(x,aa,xx0,ssigma)


def gaussmoffat(x,a,x0,sigma,A,B):
    return gauss(x,a,x0,sigma) + moffat(x,A,B)


def gaussgaussmoffat(x,a,x0,sigma,aa,xx0,ssigma,A,B):
    return gauss(x,a,x0,sigma) + gauss(x,aa,xx0,ssigma) + moffat(x,A,B)


def moffat(x,a,B):
    prefactor = 2 * (B - 1) / ( a ** 2)
    factor = (1 + ((x**2) / a**2)) ** (-B)
    return prefactor * factor


def testfit(initial_x,fit_x,fit,dy):
    index = [(np.where(np.isclose(initial_x[i],fit_x,atol=0.01)))[0][0] for i in range(len(initial_x))]
    return np.sum((dy - fit[index])**2 / dy)


def createpsf(dy):
    return (dy.reshape(1,len(dy)).T * dy)


def psf(x,xs,ys):
    index = np.where(xs > x)[0]


def cumulintegrate(x,y,func,initial_guess):
    IntY = integrate.cumtrapz(y* (np.pi * x**2),x,initial=0) 
    return x, IntY


# /////////////////////////////////////////////////////////////////
def moff(r,Beta):
    alpha = FWHM / (2 * np.sqrt(2**(1/Beta) - 1))

    return (2 * (Beta -1) / alpha **2) * (1 + r**2/alpha**2)**(-Beta)


def cumul_moff(r,B):
    a = FWHM / (2 * np.sqrt(2**(1/B) - 1))
    top = (1+(25/a**2))**B * (1 + (r**2/a**2))**(-B) * (-r**2 + (-1 + (1+(r**2/a**2))**B)*a**2)

    bottom = (-25 + (-1 + (1+(25/a**2))**B) * a**2)
    
    return top / bottom 
 

# /////////////////////////////////////////////////////////////////
def two_gauss(r,sigma,A1,A2):
    r0 = 0 

    #sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

    gauss1 = (A1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(r-r0)**2 / (2 *sigma**2))
    gauss2 = (A2 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(r-r0)**2 / (2 *sigma**2))

    return gauss1 + gauss2


def cumul_gauss(r,sigma,A1,A2):     
    # Assuming r0 = 0     

  #  sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

    return (A1 + A2) * (np.exp(-25/(2 *sigma**2))-np.exp(-r**2 / (2 *sigma**2))) * np.sqrt(2 * np.pi) * sigma + 1


# /////////////////////////////////////////////////////////////////
def fit(x,y,func,initial_guess):
    popt,pcov = curve_fit(func,x,y,p0=initial_guess)
    X = np.arange(np.min(x),np.max(x),0.001)
    return X, func(X,*popt),popt


def cof_plot(moff_x,moff_y,gauss_x,gauss_y,original_x,original_y,xlabel):
    plt.figure()
    plt.scatter(original_x,original_y,color="purple")
    plt.scatter(moff_x,moff_y,color="orange",alpha = .01)
    plt.scatter(gauss_x,gauss_y,color="aqua",alpha = .01)
    plt.xlabel("ArcSeconds")
 

def psf_plot(x,y):
    plt.figure()
    plt.scatter(x,y,s=0.1)
    plt.scatter(-x,y,s=0.1)
    plt.xlim(-20,20)
