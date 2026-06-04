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
wavelength = 1290e-9                        #<-- Longitud de onda utilizada
# wavelength = 550e-9                        #<-- Longitud de onda utilizada
wf = Wavefront(Shimm.aperture, wavelength) #type:ignore <-- Frente de onda
wf.total_power = 3.3e8                     #<-- Magnitud 0.7 de estrella en terminos de potencia


#=============================
# Grafica PSF del frente de onda: Solo es una visualizacion
#=============================
spatial_resolution = (wavelength / telescope_diameter)
focal_grid = make_focal_grid(q=8, num_airy=16, spatial_resolution=spatial_resolution)

propagator = FraunhoferPropagator(Shimm.sensor_grid, focal_grid)

unaberrated_PSF = propagator.forward(wf).power  #type: ignore
# print(unaberrated_PSF.shaped.shape)
#
plt.figure()
imshow_psf(unaberrated_PSF, cmap='inferno', normalization="peak")
# imshow_field(wf.electric_field, cmap='inferno')
plt.colorbar()
# %%
num_lenslets = 20                                       #<-- Numero de lenslets en Shack-Hartmann
lenslet_diameter = 0.5e-3                               #<-- Diametro de cada sub-apertura
shwfs = Shimm.ShackHartman(lenslet_diameter,num_lenslets,focal_length,wavelength) #<-- Definicion de Shack-Hartmann
magnifier = Shimm.magnifier                             #<-- Magnificador: Es la relacion Focal_acromatico/focal_telescopio
image_ref = shwfs(Shimm.magnifier(wf)).power            #<-- propagacion: frente_de_onda => telescopio => acromatico => Shwfs => Sensor
shwfse = Shimm.estimator(shwfs,image_ref)               #<-- Estimador con algoritmo de centro de gravedad con umbral
slope_ref ,flux ,_ = shwfse.estimate([image_ref], 0.00) #<--intensidades por sub-apertura y ubicacion de centroides en metros


#=======================
# Shack-Hartmann con centroides detectados
#=======================

plt.figure()
imshow_psf(image_ref,scale="linear",normalization="peak", cmap="gray", vmin=0, vmax=0.01)
plt.plot(slope_ref[0,:],slope_ref[1,:],".",color="r")# type: ignore
plt.xlabel("X position [m]")
plt.ylabel("y position [m]")
plt.title("Shack-Hartmann Wavefront sensor")
# %%
F = 500
T = 1/F
print(1/T)
glob_cn = Cn_squared_from_fried_parameter(0.047,wavelength=wavelength)

#============================
# Pantalla de fase Basada en OOPAO
#============================


layer1 = InfiniteVonKarman(input_grid=Shimm.pupil_grid,
                          height=0e3,
                          reference_wavelength=wavelength,
                          Cn_squared=glob_cn*0.2,
                          direction=45,
                          fps=F,
                          L0=25,
                          l0=1e-20,
                          speed=3,#<-- velocidad
                          D_tel=telescope_diameter,
                          N = Shimm.sensorSize,
                          )

layer2 = InfiniteVonKarman(input_grid=Shimm.pupil_grid,
                          height=4e3,
                          reference_wavelength=wavelength,
                          Cn_squared=glob_cn * 0.2,
                          direction=30,
                          fps=int(1/T),
                          L0=25,
                          l0=1e-20,
                          speed=3,             #<-- velocidad
                          D_tel=telescope_diameter * Shimm.oversizing_factor,
                          N = Shimm.sensorSize)

layer3 = InfiniteVonKarman(input_grid=Shimm.pupil_grid,
                          height=12e3,
                          Cn_squared=glob_cn * 0.3,
                          direction=60,
                          reference_wavelength=wavelength,
                          fps=int(1/T),
                          L0=25,
                          l0=1e-20,
                          speed=3,#<-- velocidad
                          D_tel=telescope_diameter* Shimm.oversizing_factor,
                          N = Shimm.sensorSize)

layer4 = InfiniteVonKarman(input_grid=Shimm.pupil_grid,
                          height=20e3,
                          reference_wavelength=wavelength,
                          Cn_squared=glob_cn*0.4,
                          direction=90,
                          fps=int(1/T),
                          L0=25,
                          l0=1e-20,
                          speed=3,
                          D_tel=telescope_diameter* Shimm.oversizing_factor,
                          N = Shimm.sensorSize)

#=============j=======================================
# Propagar pantallas de fase mediante angular spectrum
#=======================================================
atm = LocalMultiLayerAtmosphere([layer1,layer2,layer3,layer4],scintillation=True) #<--- atmosfera multicapa
# atm = LocalMultiLayerAtmosphere([layer1],scintillation=True) #<--- atmosfera multicapa

#===========================
# Creacion de detector sin ruido
#===========================

camera = Shimm.NoislessCamera()

#===========================
# Creacion de detector con ruido: opcional
#===========================

# camera = Shimm.ZWOCamera()
# camera = Shimm.IRCamera()


#===========================
# Mascara con sub-aperturas activas sin vigneting
#===========================

mask = np.array([[0,0,1,1,0,0],
                 [0,1,1,1,1,0],
                 [1,1,0,0,1,1],
                 [1,1,0,0,1,1],
                 [0,1,1,1,1,0],
                 [0,0,1,1,0,0]])

#===========================
# segundos de medicion
#===========================

steps = F*3

#============================
#  Matrices de autocovarianza
#============================

matx = np.zeros([11 ,11])
maty = np.zeros([11 ,11])
matf = np.zeros([11 ,11])

fluxe = []
slopex = []
slopey = []

#=======================
# Adquicision de datos
#=======================

for i in tqdm(range(steps)):
       centsx = np.zeros_like(mask, dtype=np.float64)
       centsy = np.zeros_like(mask, dtype=np.float64)
       fluxes = np.zeros_like(mask, dtype=np.float64)

       camera.integrate((shwfs(magnifier(atm(wf)))),2e-3)  # captura con tiempo de expocision de 2 ms
       wfs_image = camera.read_out()                       # Generacion de imagen

       slopes, flux, _= shwfse.estimate([wfs_image], 0.0)  # type: ignore <--Estimacion
       # slopes = (slopes - slope_ref) * Shimm.m2arc        # type: ignore <--Conversion a arcsec
       slopes = (slopes - slope_ref) * Shimm.m2r            # type: ignore <--Conversion a rad

       x_slopes = slopes[0,:]                              #<----- slopes x
       y_slopes = slopes[1,:]                              #<----- slopes y

       # slopex.append(x_slopes)
       # slopey.append(y_slopes)


       #============================
       # Ordenamiento datos espaciales a formato de mascara
       #============================
       centsx[mask == 1] = x_slopes
       centsy[mask == 1] = y_slopes
       fluxes[mask == 1] = flux                            # type: ignore

       fluxe.append(fluxes)
       #=======================================
       # Autocovarianza espacial mediante furier
       #=========================================

       Cx,_ = Shimm.spatial_autocov_fft(centsx, mask,1)     #type:ignore
       Cy,_ = Shimm.spatial_autocov_fft(centsy, mask,1)     #type:ignore

       matx += Cx
       maty += Cy
       # Evolucion capas atmsfericas

       atm.evolve()


# %%
#
nsubx = 6
tcovs = Shimm.TheoricalCov(mask, nsubx, wavelength, h=10e3)
mattx = tcovs["si"][1]
mat,_ = Shimm.w2cov(matx/steps, mask)
plt.figure()
plt.imshow(mattx)
plt.colorbar()
# plt.plot([i for i in range(-5, 6)], matx[:,5]/steps)
# plt.plot([i for i in range(-5, 6)], mattx[:,5]/(np.pi**2))
# plt.grid("both")



# %%
#============================================
# Matriz de covarianza de centelleo
#============================================

fluxe = np.array(fluxe)
flu = np.mean(fluxe,axis=0)

for i in tqdm(fluxe):
       val = (i-flu) /flu   #<-------- Indices de centelleo
       Cf,_ = Shimm.spatial_autocov_fft(val, mask,1)#type:ignore
       matf += Cf


fig, ax = plt.subplots(1, 2, figsize=(15, 6))

im0 = ax[0].imshow(flu)
ax[0].set_title("Ubicaciones espaciales de intensidad en subaperturas")
fig.colorbar(im0, ax=ax[0], orientation='horizontal', pad=0.12)

im1 = ax[1].imshow(matf/steps)
ax[1].set_title("Matriz de autocovarianza")
fig.colorbar(im1, ax=ax[1], orientation='horizontal', pad=0.12)


plt.tight_layout

# %%
mask = np.array([
                [0,0,1,1,0,0],
                [0,1,1,1,1,0],
                [1,1,0,0,1,1],
                [1,1,0,0,1,1],
                [0,1,1,1,1,0],
                [0,0,1,1,0,0],
                ])

covx = (matx/steps)
covy = (maty/steps) # covf = (matf/steps)#type: ignore
covf =  matf/steps

pcovx,_ = Shimm.w2cov(covx,mask)
pcovy,_ = Shimm.w2cov(covy,mask)
pcovf,_ = Shimm.w2cov(covf,mask)

fig = plt.figure(figsize=[10,6])
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)

ax1 = fig.add_subplot(gs[0,1])
im1 = ax1.imshow(pcovy)
ax1.set_title(r'$Cy_{ab}$ matrix \n')
fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.12)

ax2 = fig.add_subplot(gs[0,0])
im2 = ax2.imshow(pcovx)
ax2.set_title(r'$Cx_{ab}$ matrix \n')
fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.12)

ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(pcovf)
ax3.set_title(r'$Ci_{ab}$ matrix \n')
fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.12)


plt.tight_layout()


# %%
#===============================
#  Resolucion problema inverso
#===============================

import numpy as np
from scipy.optimize import nnls, lsq_linear
import matplotlib.pyplot as plt

mask =  np.array([[0,0,1,1,0,0],
                  [0,1,1,1,1,0],
                  [1,1,0,0,1,1],
                  [1,1,0,0,1,1],
                  [0,1,1,1,1,0],
                  [0,0,1,1,0,0]])

sampling = 1024
cn2 = 1
lamda = 500e-9
d = telescope_diameter / nsubx      # subaperture size
dx = d/2

cx = pcovx.flatten()
cy = pcovy.flatten()
cf = pcovf.flatten()

c = np.concat([cf,cx,cy])
# print(b)

tcovs1 = Shimm.TheoricalCov(mask, nsubx, wavelength, h=0e3)
tcovs2 = Shimm.TheoricalCov(mask, nsubx, wavelength, h=4e3)
tcovs3 = Shimm.TheoricalCov(mask, nsubx, wavelength, h=12e3)
tcovs4 = Shimm.TheoricalCov(mask, nsubx, wavelength, h=20e3)

_,wx1,tcovx1 = tcovs1["sx"]
_,wx3,tcovx2 = tcovs2["sx"]
_,wx3,tcovx3 = tcovs3["sx"]
_,wx4,tcovx4 = tcovs4["sx"]

_,wy1,tcovy1 = tcovs1["sy"]
_,wy3,tcovy2 = tcovs2["sy"]
_,wy3,tcovy3 = tcovs3["sy"]
_,wy4,tcovy4 = tcovs4["sy"]

wi1,tcovi1 = tcovs1["si"]
wi3,tcovi2 = tcovs2["si"]
wi3,tcovi3 = tcovs3["si"]
wi4,tcovi4 = tcovs4["si"]

W0  = np.concat([tcovi1.flatten(), tcovx1.flatten(), tcovy1.flatten()])
W4  = np.concat([tcovi2.flatten(), tcovx2.flatten(), tcovy2.flatten()])
W12 = np.concat([tcovi3.flatten(), tcovx3.flatten(), tcovy3.flatten()])
W20 = np.concat([tcovi4.flatten(), tcovx4.flatten(), tcovy4.flatten()])

W = np.column_stack([W0,W4,W12,W20])

# x, residuals, rank, s = scipy.linalg.lstsq(W, b)
y = [glob_cn*0.2, glob_cn*0.2,glob_cn*0.3,glob_cn*0.4]
sol = lsq_linear(
    W,
    c,
    method="bvls"
)
j = np.abs(sol.x)
total1 = np.sum(j)
print(total1)
total2 = np.sum(y)
print(total2)

print(fried_parameter_from_Cn_squared(total1,wavelength=wavelength))
print(fried_parameter_from_Cn_squared(total2,wavelength=wavelength))
print(j)
print(y)
plt.figure()
plt.plot(y,j, "-o")
plt.xlabel("Cn2 Real")
plt.ylabel("Cn2 Estimado")
plt.title("estimaciones vs valores reales")
plt.grid("both")
#
# fig = plt.figure(figsize=[15,6])
# gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], wspace=0.4)
# #
# ax1 = fig.add_subplot(gs[0,0])
# im1 = ax1.imshow(wy1)
# ax1.set_title(r'$Cx_{ab}$ matrix teorica')
# fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.12)
# #
# ax2 = fig.add_subplot(gs[0,1])
# im2 = ax2.imshow(wx1)
# ax2.set_title(r'$Cy_{ab}$ matrix teorica')
# fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.12)
# #
# ax3 = fig.add_subplot(gs[0, 2])
# im3 = ax3.imshow(wf2)
# ax3.set_title(r'$Ci_{ab}$ matrix teorica \n 4 km')
# fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.12)
# #
# ax4 = fig.add_subplot(gs[1,0])
# im4 = ax4.imshow(psfx1)
# ax4.set_title(r'$S_{x}$ matrix teorica')
# #
# ax5 = fig.add_subplot(gs[1,1])
# im5 = ax5.imshow(psfy1)
# ax5.set_title(r'$S_{y}$ matrix teorica')
# #
# ax6 = fig.add_subplot(gs[1, 2])
# im6 = ax6.imshow(psf2)
# ax6.set_title(r'$S_{i}$ matrix teorica \n 4 km')
# #
# #
# plt.tight_layout()
# rank = np.linalg.matrix_rank(W)
# cond = np.linalg.cond(W)
#
# print("rank(W) =", rank)
# print("cond(W) =", cond)
#

# %%

def empirical_spatial_covariance(coords, values, max_lag, bin_size):
    """
    Calcula la autocovarianza espacial empírica 2D.

    Parámetros:
        coords   : ndarray de forma (P, 2), coordenadas [x y] de cada subpupila
        values   : ndarray de forma (P,), desplazamientos (puede ser eje X o Y)
        max_lag  : float, máximo desplazamiento (lag) a considerar
        bin_size : float, tamaño de bin (en mismas unidades que coords)

    Retorna:
        C     : ndarray (2*M+1, 2*M+1), matriz de covarianzas empíricas
        count : ndarray (2*M+1, 2*M+1), cantidad de pares acumulados por bin
    """

    z = values - np.mean(values)  # desviación respecto a la media

    M = int(np.ceil(max_lag / bin_size))
    L = 2 * M + 1  # dimensiones de la matriz de bins

    C = np.zeros((L, L))
    count = np.zeros((L, L), dtype=int)

    P = len(z)
    for p in range(P - 1):
        for q in range(p + 1, P):
            dx = coords[q, 0] - coords[p, 0]
            dy = coords[q, 1] - coords[p, 1]

            ix = int(round(dx / bin_size)) + M
            iy = int(round(dy / bin_size)) + M

            if 0 <= ix < L and 0 <= iy < L:
                prod = z[p] * z[q]

                # acumulación normal
                C[iy, ix] += prod
                count[iy, ix] += 1

                # acumulación simétrica (-dx, -dy)
                C[L - iy - 1, L - ix - 1] += prod
                count[L - iy - 1, L - ix - 1] += 1

    # auto-pares (lag = 0)
    center = M
    C[center, center] += np.sum(z ** 2)
    count[center, center] += P

    # normalización
    with np.errstate(invalid='ignore'):  # suprime warning por división 0/0
        C = np.where(count > 0, C / count, np.nan)

    return C

# %%
