import numpy as np

def Kolmogorov_PSD(f: np.ndarray, wavelength: float):
    """
        spectral filter Z_x(f) (Ec. 3.8).
    
        Args:
            f (ndarray)       : spatial frecuency in X.
            r0 float          : strength of the turbulent in the layer  
            glob (Bool)       : if you want to use a Phi_K with a global r0 or not
            
        Returns:

            Phi_K(F) (ndarray): kolmogorov_PSD.
    """

    k = 2 * np.pi / wavelength
    k2 = k*k

    Phi_K = np.zeros_like(f)
    # x/0 prevention
    mask_f = f > 0 
    Cn2_dh = 1           # asumes a layer with cn2 = 1
    #Kolmogorov PSD eq (2.11): layer aproach
    Phi_K[mask_f] = 9.7e-3 * k2 * (f[mask_f] ** (-11/3)) * Cn2_dh
    # piston remove
    Phi_K[0,0] = 0

    return Phi_K



def calculate_Zx_spectral_filter(fx: np.ndarray, fy: np.ndarray, wavelength: float, d: float):
    """
    spectral filter Z_x(f) (Ec. 3.8).
    
    Args:
        fx (ndarray)       : spatial frecuency in X.
        fy (ndarray)       : spatial frecuency in Y.
        d (float)          : sub-aperture size.
        
    Returns:

        Zx(F) (ndarray): spatial filter.
    """
    scalingFactor =((np.sqrt(3) * wavelength) / (np.pi*d)) ** 2
    y_term = 3 * (np.sinc(d * fy)) ** 2

    # [u*cos(u) - sin(u)]^2 / u^4, where u = pi*d*fx
    u = np.pi * d * fx
    u2 = u*u
    u4 = u2*u2 # @dev_note: I found this way of obtaining u^4 curious but interesting.
    
    # init X term with zeros
    x_term = np.zeros_like(u)
    mask = np.abs(u) > 0 #<-- mask preventing x/0 cases
    
    # numerator: [u cos(u) - sin(u)]^2
    numerator = ( u[mask] * np.cos(u[mask]) - np.sin(u[mask])) ** 2
    
    # [u cos(u) - sin(u)]^2 / u^4
    x_term[mask] = numerator / u4[mask]
    
    Zx =  y_term * x_term * scalingFactor   
    return Zx


def weighting_function(d: float,
                       dx: float,
                       nsubx: int,
                       samp: int,
                       wavelength: float =500e-9,
                       h:float =0.,
                       scalingFactor: float = 1.,
                       ):

    """
    calculate weighting function (cov) for SHIMM.
    
    Args:

        dx(float)           : steps in phychic aperture [m]
        h (float)           : layer altitude    [m].
        d (float)           : sub-aperture size [m].
        samp(int)           : frequency oversampling
        glob (Bool)         : if you want to use a Phi_K with a global r0 or not
        nsubx (int)         : grid size.
        cn2r0(float)        : value of r0 or cn2 in K_PSD depends on glob
        sclaingFactor(float): if you want to get the result in rad2 or arcsec2

    Returns:
    return  W_z (ndarray): [2,N,N]. where N = 2*nsubx-1 

    cov matrix for x-slope and y-slope .

    cov[0, di, dj] -> cov_xx  
    cov[1, di, dj] -> cov_yy 

    """
    
    # C: matriz espacial (p.ej. covarianza) tamaño N0xN0
    Nfft = max(256, samp) 

    # 1. spatio-frequency domain (fx, fy)
    freq = np.fft.fftfreq(Nfft, d=dx)   #<-- spatio-frequency vector based on sub-aperture grid size [ cycles / (m)]

    fx, fy = np.meshgrid(freq, freq) 


    # spatio-frequency magnitude 
    f = np.hypot(fx,fy)   # f = ||f||
    f2 = f*f

    
    # ---------------------------------------------------------
    
    # 3. Shack-Hartmann aperture filter (Ec. 3.8)
    A_fx = calculate_Zx_spectral_filter(fx, fy, wavelength, d)
    A_fy = calculate_Zx_spectral_filter(fy ,fx ,wavelength, d)
    

    # 4. propagation ( sinusoidal term in Ec. 3.9) fresnel phase param
    Fresnel_term = np.cos(np.pi * wavelength * h * f2)**2

    #5. Kolmogorov PSD obtain (Ec. 2.11)
    Phi_K = Kolmogorov_PSD(f,wavelength)

    # 6. Filter applied
    spectral_densityX =  Phi_K * Fresnel_term * A_fx  
    spectral_densityY =  Phi_K * Fresnel_term * A_fy 

    # 7. ifft
    df = freq[1] - freq[0]
    fft_integral_scale = (Nfft**2) * (df**2)

    covariance_mapX = np.fft.ifft2(spectral_densityX) *  fft_integral_scale
    covariance_mapY = np.fft.ifft2(spectral_densityY) *  fft_integral_scale
    
    covariance_mapX = np.fft.fftshift(covariance_mapX) 
    covariance_mapY = np.fft.fftshift(covariance_mapY) 
    

    mid = int(Nfft/2)
    stride = int(round(d / dx))   # si dx=d/2 -> stride=2
    k = np.arange(-(nsubx-1), nsubx)  # [-5..+5] para nsubx=6
    idx = mid + stride * k

    W_zX = np.real(covariance_mapX[np.ix_(idx,idx)]) * scalingFactor  # X slope on [Rad^2] -> [arcsec^2]
    W_zY = np.real(covariance_mapY[np.ix_(idx,idx)]) * scalingFactor  # Y slope on [Rad^2] -> [arcsec^2]
    W_f = scintillation_weighting_function(d,dx,nsubx,samp,wavelength,h,scalingFactor)

    return W_zX, W_zY, W_f

def scintillation_weighting_function(
                             d: float,
                             dx: float,
                             nsubx: int,
                             samp: int,
                             wavelength: float =500e-9,
                             h:float =0.,
                             scalingFactor: float = 1.,
                             ):

    """
    calculate weighting function (cov) for SHIMM.
    
    Args:

        dx(float)           : steps in phychic aperture [m]
        h (float)           : layer altitude    [m].
        d (float)           : sub-aperture size [m].
        samp(int)           : frequency oversampling
        glob (Bool)         : if you want to use a Phi_K with a global r0 or not
        nsubx (int)         : grid size.
        cn2r0(float)        : value of r0 or cn2 in K_PSD depends on glob
        sclaingFactor(float): if you want to get the result in rad2 or arcsec2

    Returns:
    return  W_z (ndarray): [N,N]. where N = 2*nsubx-1 

    cov matrix for Scintillation index .

    cov[di, dj] -> cov_I 

    """
    
    # 1. spatio-frequency domain (fx, fy)
    Nfft = max(256, samp) 

    # 1. spatio-frequency domain (fx, fy)
    freq = np.fft.fftfreq(Nfft, d=dx)   #<-- spatio-frequency vector based on sub-aperture grid size [cycles/m]
    fx, fy = np.meshgrid(freq, freq) 

    # spatio-frequency magnitude 
    f = np.hypot(fx,fy)   # f = ||f||
    f2 = f*f
    
    # ---------------------------------------------------------
    
    # 3. aperture filter Shack-Hartmann (Ec. 2.44)
    A_f =  (np.sinc(d * fx)**2) * (np.sinc(d * fy)**2)
    
    # 4. propagation ( sinusoidal term in Ec. 2.42/2.45) fresnel phase param
    Fresnel_term = np.sin(np.pi * wavelength * h * f2)**2

    #5. Kolmogorov PSD obtain
    Phi_K = Kolmogorov_PSD(f,wavelength)

    spectral_density = Phi_K * Fresnel_term * A_f * 4

    # 6. Numerical solution using IFFT 
    df = freq[1] - freq[0]
    fft_integral_scale = (Nfft**2) * (df**2)

    covariance_map = np.fft.ifft2(spectral_density) * fft_integral_scale
    covariance_map = np.fft.fftshift(covariance_map) 
    
    # fftshift centra la frecuencia cero en el medio de la imagen para visualización

    mid = int(Nfft/2) 
    stride = int(round(d / dx))   # si dx=d/2 -> stride=2
    k = np.arange(-(nsubx-1), nsubx)  # [-5..+5] para nsubx=6
    idx = mid + stride * k

    # W_z = np.real(covariance_map[mid:mid+N0,mid:mid+N0]) * scalingFactor  # Y slope on [Rad^2] -> [arcsec^2]
    W_z = np.real(covariance_map[np.ix_(idx,idx)]) * scalingFactor  

    return W_z



