# %%
from hcipy import *
from tqdm import tqdm
from OpticalSystem import ShimmOptic
import numpy as np
from matplotlib import animation

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage import io
from skimage.transform import rotate
%matplotlib inline

video_path = "260618220053.mp4"

newconf = Configuration()
newconf.update({'core':{"use_new_style_fields":True}})
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video")

fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("FPS:", fps)
print("Frames:", num_frames)
print("Resolución:", int(width), "x", int(height))

# %%
# Convertir a monocromático si OpenCV lo cargó como BGR
# Ahora gray es un array NumPy 2D
frames = []

for i in tqdm(range(4000)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

frames = np.array(frames)
print(frames.shape)
# %%
telescope_diameter = 0.5          # diametro telescopio en metros
central_obscuration_ratio = 0.34   # obstrucción central
# central_obscuration_ratio = 0.   # obstrucción central
pixel_size = 5e-6              # Tamaño del pixel camara ZWO con bining * 4
telescope_fnumber = 6.8             # El de Seetrue es 6.8
fach = (25.4e-3)*(2.6/3.78)                      # El nuestro es 25.4
sensorSize = 500                   # Roi del sensor al arreglo de microlentes
focal_length = 32.8e-3             #<-- shack Hartmann focal length

Shimm = ShimmOptic(telescope_diameter,
                   central_obscuration_ratio,
                   fach,
                   pixel_size,
                   sensorSize,
                   SHcentered=False, #<-- Shack-Hartmann centrado o no. Nota: El nuestro de 7 no esta centrado
                   telescope_fnumber=telescope_fnumber,
                   focal_length=focal_length)

Shimm.pupil_info(True)


print(Shimm.oversizing_factor * telescope_diameter)

wavelength = 550e-9                        #<-- Longitud de onda arcturus
wf = Wavefront(Shimm.aperture, wavelength) #type:ignore <-- Frente de onda
wf.total_power = 3.3e8                     #<-- Magnitud 0.7 de estrella en terminos de potencia

# %%
%matplotlib qt

num_lenslets = 20                                       #<-- Numero de lenslets en Shack-Hartmann
lenslet_diameter = (0.5e-3)*(2.6/3.78)                            #<-- Diametro de cada sub-apertura
nsubx = 7
shwfs = Shimm.ShackHartman(lenslet_diameter,num_lenslets,focal_length,wavelength) #<-- Definicion de Shack-Hartmann
magnifier = Shimm.magnifier                             #<-- Magnificador: Es la relacion Focal_acromatico/focal_telescopio
# mag2 = Magnifier(35/45)

camera = Shimm.NoislessCamera()
image = camera.integrate(shwfs(magnifier(wf)),0.2e-3)
image_ref = camera.read_out()

shwfse = Shimm.estimator(shwfs,image_ref,0.9)               #<-- Estimador con algoritmo de centro de gravedad con umbral
slope_ref ,flux ,_ = shwfse.estimate([image_ref]) #<--intensidades por sub-apertura y ubicacion de centroides en metros

# camera = Shimm.IRCameraObs()

#=======================
# Shack-Hartmann con centroides detectados
#=======================


#
%matplotlib qt
fram = frames.mean(axis=0)
new = Shimm.image_pipeline(fram,angle=0.8,crop_box=(82,0,82+500,500),shift=(0,-15))

plt.figure()
plt.subplot(1,2,1)
imshow_field(image_ref,cmap="gray")
plt.plot(slope_ref[0,:],slope_ref[1,:],".",color="r")# type: ignore
plt.xlabel("X position [m]")
plt.ylabel("y position [m]")
plt.title("Shack-Hartmann Wavefront sensor")
print(frames[0,:,:].shape)
newFrame = Shimm.image_pipeline(fram,angle=0.8,crop_box=(82,0,82+500,500),shift=(0,-15)).astype(np.float64)
imtest = Field(newFrame.ravel(),Shimm.sensor_grid)
slope ,flux ,_ = shwfse.estimate([imtest],0.8) #<--intensidades por sub-apertura y ubicacion de centroides en metros
plt.subplot(1,2,2)
imshow_field(imtest,cmap="gray")
# plt.imshow(new,cmap="gray")
plt.plot(slope[0,:],slope[1,:],".",color="g")# type: ignore
plt.plot(slope_ref[0,:],slope_ref[1,:],".",color="r")# type: ignore
plt.xlabel("X position [m]")
plt.ylabel("y position [m]")
# %%
F = 60
steps = F*30

matx1 = np.zeros([steps, len(slope_ref[0,:])])
maty1 = np.zeros([steps, len(slope_ref[0,:])])
matf1 = np.zeros([steps, len(slope_ref[0,:])])

def SlopeDenoise(pcov):
    row_mean = pcov.mean(axis=1, keepdims=True)  # (nsubtot,1,2)
    col_mean = pcov.mean(axis=0, keepdims=True)  # (1,nsubtot,2)
    glob_mean = pcov.mean(axis=(0, 1), keepdims=True)  # (1,1,2)
    return pcov - row_mean - col_mean + glob_mean

def scintillation_noise_bias(
    S,
    n_pix,
    B=0.0,
    dark_current_rate=0.0,
    exposure_time=1.0,
    read_noise_rms=0.0,
):
    """
    Calcula el sesgo de ruido fotométrico para la covarianza de centelleo.

    Parámetros
    ----------
    S : float or np.ndarray
        Señal de la estrella por subapertura [counts].
    n_pix : int or float
        Número de píxeles por subapertura.
    B : float or np.ndarray
        Ruido de fondo por píxel [counts/pixel].
    dark_current_rate : float
        Corriente oscura [counts/pixel/s].
    exposure_time : float
        Tiempo de exposición [s].
    read_noise_rms : float
        Ruido de lectura RMS [counts RMS/pixel].

    Retorna
    -------
    bias : float or np.ndarray
        Sesgo diagonal de la covarianza de centelleo.
    """

    S = np.asarray(S, dtype=float)

    D = dark_current_rate * exposure_time          # [counts/pixel]
    read_noise_var = read_noise_rms**2            # [counts^2/pixel]

    bias = (S + n_pix * (B + D + read_noise_var)) / S**2

    return bias

for t in tqdm(range(steps)):
    newFrame = Shimm.image_pipeline(frames[t,:,:],angle=0.8,crop_box=(82,0,82+500,500),shift=(0,-15)).astype(np.float64)
    wfs_image = Field(newFrame.ravel(),Shimm.sensor_grid)
    slopes, flux, _= shwfse.estimate([wfs_image], 0.8)  # type: ignore <--Estimacion
    slopes = ((slopes) - (slope_ref)) * Shimm.m2r          # type: ignore <--Conversion a rad

    matx1[t,:] = slopes[0,:]
    maty1[t,:] = slopes[1,:]
    matf1[t,:] = flux

# %%
flu = np.mean(matf1, axis=0)
matff1 = (matf1 - flu) / flu

pcovx = np.cov(matx1,rowvar = False)
pcovy = np.cov(maty1,rowvar = False)
pcovf = np.cov(matff1,rowvar = False)

dark_current_rate = 14.40   # counts/s, por ejemplo @20°C
read_noise_rms = 4.76       # counts RMS

S = 50000                   # counts por subapertura
n_pix = 100                 # píxeles por subapertura
B = 20                      # counts/pixel
exposure_time = 128e-3        # segundos

bias = scintillation_noise_bias(
    S=S,
    n_pix=n_pix,
    B=B,
    dark_current_rate=dark_current_rate,
    exposure_time=exposure_time,
    read_noise_rms=read_noise_rms,
)
print(bias)
Na = len(slopes[0,:])
e = np.mean(matx1.std(axis=0)/np.sqrt(steps))**2

biasmat = np.ones(Na) * bias
pcovf -= np.diag(biasmat)

pcovx2 = SlopeDenoise(pcovx)
pcovy2 = SlopeDenoise(pcovy)
for i in range(20):
    for j in range(20):
        if j == i:
            pcovx2[i,j] -= (1-(1/Na)) * e
            pcovy2[i,j] -= (1-(1/Na)) * e
        else:
            pcovx2[i,j] -= -(1/Na) * e
            pcovy2[i,j] -= -(1/Na) * e
plt.figure()
plt.plot(matx1[:,1])
# plt.figure(figsize=[10,10])
# plt.subplot(1,3,1)
# plt.imshow(pcovx2)
# plt.colorbar()
# plt.subplot(1,3,2)
# plt.imshow(pcovy2)
# plt.colorbar()
# plt.subplot(1,3,3)
# plt.imshow(pcovf)
# plt.colorbar()


# %%
import numpy as np
from scipy.optimize import nnls, lsq_linear
import matplotlib.pyplot as plt

mask = np.array([
                [0,0,1,1,1,0,0],
                [0,1,1,1,1,1,0],
                [1,1,0,0,0,1,1],
                [1,1,0,0,0,1,1],
                [1,1,0,0,0,1,1],
                [0,1,1,1,1,1,0],
                [0,0,1,1,1,0,0],
                ])

cx = pcovx2.ravel()
cy = pcovy2.ravel()
cf = pcovf.ravel()

b = np.concat([cf,cx,cy])
# %%
tcovs1 = Shimm.TheoricalCov(mask, nsubx, wavelength, h=0e3)
tcovs2 = Shimm.TheoricalCov(mask, nsubx, wavelength, h=4e3)
tcovs3 = Shimm.TheoricalCov(mask, nsubx, wavelength, h=12e3)
tcovs4 = Shimm.TheoricalCov(mask, nsubx, wavelength, h=20e3)

psfx1,wx1,tcovx1 = tcovs1["sx"]
psfx2,wx2,tcovx2 = tcovs2["sx"]
psfx3,wx3,tcovx3 = tcovs3["sx"]
psfx4,wx4,tcovx4 = tcovs4["sx"]

psfy1,wy1,tcovy1 = tcovs1["sy"]
psfy2,wy2,tcovy2 = tcovs2["sy"]
psfy3,wy3,tcovy3 = tcovs3["sy"]
psfy4,wy4,tcovy4 = tcovs4["sy"]

wi1,tcovi1 = tcovs1["si"]
wi2,tcovi2 = tcovs2["si"]
wi3,tcovi3 = tcovs3["si"]
wi4,tcovi4 = tcovs4["si"]

tcovx1,_ = Shimm.w2cov(psfx1,mask)
tcovx2,_ = Shimm.w2cov(psfx2,mask)
tcovx3,_ = Shimm.w2cov(psfx3,mask)
tcovx4,_ = Shimm.w2cov(psfx4,mask)

tcovy1,_ = Shimm.w2cov(psfy1,mask)
tcovy2,_ = Shimm.w2cov(psfy2,mask)
tcovy3,_ = Shimm.w2cov(psfy3,mask)
tcovy4,_ = Shimm.w2cov(psfy4,mask)

W0  = np.concat([tcovi1.ravel(), tcovx1.ravel(), tcovy1.ravel()])
W4  = np.concat([tcovi2.ravel(), tcovx2.ravel(), tcovy1.ravel()])
W12 = np.concat([tcovi3.ravel(), tcovx3.ravel(), tcovy1.ravel()])
W20 = np.concat([tcovi4.ravel(), tcovx4.ravel(), tcovy1.ravel()])

W = np.column_stack([W0,W4,W12,W20])
print(np.linalg.cond(W))
# prev = np.linalg.cond(W)

# %%
W_np = W
c_np = b

# E = np.diag(1/sigma)

sol = lsq_linear(
    W_np,
    c_np,
    tol=0,
    bounds=(0, np.inf),
    verbose=1,
    method="bvls"
)
j = sol.x

total1 = np.sum(j)

from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="resultados estimacion")

table.add_column("parametro")
table.add_column("estimacion")

r0est = fried_parameter_from_Cn_squared(total1,wavelength=wavelength)

table.add_row("r0", f"{r0est:.3f} [m]")
table.add_row("h1: 0km  ", f"{j[0]:.3e}" + r"[m^-3]" )
table.add_row("h2: 4km  ", f"{j[1]:.3e}" + r"[m^-3]" )
table.add_row("h3: 12km ", f"{j[2]:.3e}" + r"[m^-3]" )
table.add_row("h4: 20km ", f"{j[3]:.3e}" + r"[m^-3]" )
console.print(table)
