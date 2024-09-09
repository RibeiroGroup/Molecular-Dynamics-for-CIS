import numpy as np

from scipy import signal
import math, os, sys, time

def check_path(path):
    """
    Check if path exist
    """
    try: 
        assert os.path.isdir(path)
    except AssertionError:
        warnings.warn('Path {} does not exist!'.format(path))

def profiling_rad(omega_list,Hrad):
    """
    Calculate the energy of the field for each unique wavenumbers.
    """

    unique_omega = list(set(np.round(omega_list,decimals = 6)))
    unique_omega = np.sort(unique_omega)
    rad_profile = []

    for i, omega in enumerate(unique_omega):
        rad_profile.append(
                np.sum(Hrad[np.isclose(omega_list, omega)])
                )

    return unique_omega, rad_profile


def field_spectra(result_dict, limit = None):
    rad_profile = []
    
    for i, rd in result_dict.items():

        if limit and i > limit : 
            continue
        
        Afield = rd["field"]
        
        rad_energy = red.convert_energy(np.array(Afield.history["energy"][-1]), "ev") 
        omega = red.convert_wavenumber(Afield.k_val)
        omega_profile, final_rad_profile = profiling_rad(omega, rad_energy)
        
        foo = np.argsort(omega_profile)
        omega_profile = np.array(omega_profile)[foo]
        final_rad_profile = np.array(final_rad_profile)[foo]
        
        rad_profile.append(final_rad_profile)
        
    rad_profile = np.mean(rad_profile, axis = 0)
    
    return omega_profile, np.array(rad_profile)


def fft_autocorr(t,dp_vel,dtps,windows='Gaussian'):
    autocorr = calc_ACF(dp_vel)
    yfft = calc_FFT(autocorr, windows)
    yfft = np.nan_to_num(yfft)
    intensity = np.sum(yfft, axis=1)[0:int(len(yfft)/2)]
    
    delta_t = dtps * 1e-12 * 3e10
    wvn = np.fft.fftfreq(len(yfft), delta_t)[0:int(len(yfft)/2)]
    
    return wvn, intensity

def dipole_spectra(
    atoms, dtps, time_frame = None, 
    quant = 'dipole_velocity', windows = 'Gaussian'):
    #h = result_dict['h']

    if time_frame: 
        ti, tf = time_frame
    dipole_velocity = np.sum(
        atoms.observable[quant], axis = 1)
    dp = np.array(dipole_velocity)
    #dp_vel = np.sum(
    #    result_dict['atoms'].observable['dipole'], axis = 1
    #)

    if time_frame: 
        time = np.array(atoms.observable['t'])
        time -= time[0]
        dp = dp[(time > ti) * (time < tf)]

    wavenumber, ir = fft_autocorr(time, dp, dtps, windows)

    if quant == 'dipole':
        ir *= wavenumber**2

    return np.array(wavenumber), np.array(ir)

"""
Calculating ACF and FFT functions
"""

def zero_padding(sample_data):
    '''
      A series of Zeros will be padded to the end of the dipole moment array 
    (before FFT performed), in order to obtain a array with the length which
    is the "next power of two" of numbers.
    #### Next power of two is calculated as: 2**np.ceil(log2(x))
    #### or Nfft = 2**int(math.log(len(data_array)*2-1, 2))
    '''
    N = 2**int(math.log(len(sample_data)*2-1, 2))
    return N

def calc_ACF(array):
    '''
    Calculating Autocorrelation function ACF). 
    The original calculate ACF of array A by convolving A with time (~array index) 
    inversed of itself.
    Arg:
    + array (np.array): input array for calculating ACF
    '''
    # normalization
    yunbiased = array - np.mean(array, axis=0)
    ynorm = np.sum(np.power(yunbiased,2), axis=0)
    # print "the average value of input data array", ynorm
    autocor = np.zeros(np.shape(array))

    for i in range(3):
        autocor[:,i] = signal.fftconvolve(array[:,i],
                                          array[:,i][::-1],
                                          mode='full')[len(array)-1:]/ynorm[i]
    return autocor

def choose_window(data, kind='string'):
    if kind == 'Gaussian':
        sigma = 2 * math.sqrt(2 * math.log(2))
        window = signal.gaussian(len(data), std=4000.0/sigma, sym=False)
    elif kind == 'BH':
        window = signal.blackmanharris(len(data), sym=False)
    elif kind == 'Hamming':
        window = signal.hamming(len(data), sym=False)
    elif kind == 'Hann':
        window = signal.hann(len(data), sym=False)
    return window


def calc_FFT(data, window):
    '''
    This function is for calculating the "intensity" of the ACF at each 
    frequency by using the discrete fast Fourier transform.
    
####
#### http://stackoverflow.com/questions/20165193/fft-normalization
####
    '''
    window = choose_window(data, kind=window)
    WE = sum(window) / len(data)
    wf = window / WE
    # convolve the window function. 
    sig = data * wf[None,:].T

    # A series of number of zeros will be padded to the end of the DACF \
    # array before FFT.
    N = zero_padding(sig)
	
    yfft = np.fft.fft(sig, N, axis=0) / len(sig)
# without window function
#    yfft = np.fft.fft(data, n=int(N_fft), axis=0) / len(data)
    return np.square(np.absolute(yfft))

