# %%
from hcipy import *
from tqdm import tqdm
from OpticalSystem import ShimmOptic
from atm import InfiniteVonKarman, LocalMultiLayerAtmosphere
from atmosfera import KolmogorovGenerator

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from matplotlib import animation

%matplotlib inline

# %%

telescope_diameter = 0.28          # diametro telescopio en metros
central_obscuration_ratio = 0.34   # obstrucción central
# central_obscuration_ratio = 0.   # obstrucción central
# pixel_size = 4*2.9e-6              # Tamaño del pixel camara ZWO con bining * 4
pixel_size = 4*2.9e-6              # Tamaño del pixel camara ZWO con bining * 4
telescope_fnumber = 10             # El de Seetrue es 6.9
fach = 30e-3                       # El nuestro es 25.4
sensorSize = 258                   # Roi del sensor al arreglo de microlentes
focal_length = 15.4e-3             #<-- shack Hartmann focal length

Shimm = ShimmOptic(telescope_diameter,
                   central_obscuration_ratio,
                   fach,
                   pixel_size,
                   sensorSize,
                   SHcentered=True, #<-- Shack-Hartmann centrado o no. Nota: El nuestro de 7 no esta centrado
                   telescope_fnumber=telescope_fnumber,
                   focal_length=focal_length)

Shimm.pupil_info(True)
print(Shimm.oversizing_factor * telescope_diameter)

# %%

%matplotlib inline
# wavelength = 652e-9                        #<-- Longitud de onda utilizada
# wavelength = 1290e-9                        #<-- Longitud de onda utilizada
wavelength = 550e-9                        #<-- Longitud de onda utilizada
wf = Wavefront(Shimm.aperture, wavelength) #type:ignore <-- Frente de onda
wf.total_power = 3.3e8                     #<-- Magnitud 0.7 de estrella en terminos de potencia

num_lenslets = 20                                       #<-- Numero de lenslets en Shack-Hartmann
lenslet_diameter = 0.5e-3                               #<-- Diametro de cada sub-apertura
shwfs = Shimm.ShackHartman(lenslet_diameter,num_lenslets,focal_length,wavelength) #<-- Definicion de Shack-Hartmann
magnifier = Shimm.magnifier                             #<-- Magnificador: Es la relacion Focal_acromatico/focal_telescopio
image_ref = shwfs(Shimm.magnifier(wf)).power            #<-- propagacion: frente_de_onda => telescopio => acromatico => Shwfs => Sensor
shwfse = Shimm.estimator(shwfs,image_ref)               #<-- Estimador con algoritmo de centro de gravedad con umbral
slope_ref ,flux ,_ = shwfse.estimate([image_ref], 0.00) #<--intensidades por sub-apertura y ubicacion de centroides en metros


# %%
mask = np.array([[0,0,1,1,0,0],
                 [0,1,1,1,1,0],
                 [1,1,0,0,1,1],
                 [1,1,0,0,1,1],
                 [0,1,1,1,1,0],
                 [0,0,1,1,0,0]])

tcovs = Shimm.TheoricalCov(mask, 6, wavelength, h=0 )

plt.figure()
plt.imshow(tcovs["sx"][0])
