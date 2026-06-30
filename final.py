# %%
from hcipy import *
from tqdm import tqdm
from OpticalSystem import ShimmOptic
from atm import InfiniteVonKarman, LocalMultiLayerAtmosphere
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from matplotlib import animation

%matplotlib inline
newconf = Configuration()
newconf.update({'core':{"use_new_style_fields":True}})
# %%

telescope_diameter = 0.5          # diametro telescopio en metros
central_obscuration_ratio = 0.34   # obstrucción central
# central_obscuration_ratio = 0.   # obstrucción central
pixel_size = 3*5e-6               # Tamaño del pixel camara ZWO con bining * 4
telescope_fnumber = 6.8             # El de Seetrue es 6.8
fach = (25.4e-3)*(35/48)                    # El nuestro es 25.4
sensorSize = 500/3                   # Roi del sensor al arreglo de microlentes
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
# %%
data = pd.read_csv("./stae434_supplemental_files/shimm_data_latest_sup.csv")
# data.head()
cndata = data.loc[(data["cn2dh 0m [m^1/3]"] > 0 ) & (data["cn2dh 4000m [m^1/3]"] > 0) & (data["cn2dh 12000m [m^1/3]"] > 0) & (data["cn2dh 20000m [m^1/3]"] > 0), ["Seeing [arcsec]","cn2dh 0m [m^1/3]","cn2dh 4000m [m^1/3]","cn2dh 12000m [m^1/3]","cn2dh 20000m [m^1/3]"] ].to_numpy()
cn = cndata[:,1:]
# print(np.sum(cntest))
wav = 500e-9
glob_r0 = fried_parameter_from_Cn_squared(np.sum(cn[0,:]), wavelength=wav)
print(glob_r0)
cnsigma = cndata[:,1:]
print(cnsigma.shape)
# print(cntest[["cn2dh 4000m [m^1/3]","cn2dh 4000m error [m^1/3]"]])
# %%

%matplotlib inline
# wavelength = 652e-9                        #<-- Longitud de onda utilizada
# wavelength = 1290e-9                        #<-- Longitud de onda utilizada
wavelength = 500e-9                        #<-- Longitud de onda utilizada
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
%matplotlib qt

num_lenslets = 20                                       #<-- Numero de lenslets en Shack-Hartmann
lenslet_diameter = (0.5e-3)*(35/48)                            #<-- Diametro de cada sub-apertura
nsubx = 7
shwfs = Shimm.ShackHartman(lenslet_diameter,num_lenslets,focal_length,wavelength) #<-- Definicion de Shack-Hartmann
magnifier = Shimm.magnifier                             #<-- Magnificador: Es la relacion Focal_acromatico/focal_telescopio
# mag2 = Magnifier(35/45)

camera = Shimm.NoislessCamera()
image = camera.integrate(shwfs(magnifier(wf)),128e-3)
image_ref = camera.read_out()

shwfse = Shimm.estimator(shwfs,image_ref,0.9)               #<-- Estimador con algoritmo de centro de gravedad con umbral
slope_ref ,flux ,_ = shwfse.estimate([image_ref]) #<--intensidades por sub-apertura y ubicacion de centroides en metros

# camera = Shimm.IRCameraObs()

#=======================
# Shack-Hartmann con centroides detectados
#=======================

plt.figure()
# imshow_psf(image_ref,scale="linear",normalization="peak", cmap="gray")
imshow_field(image_ref,cmap="gray")
plt.plot(slope_ref[0,:],slope_ref[1,:],".",color="r")# type: ignore
plt.xlabel("X position [m]")
plt.ylabel("y position [m]")
plt.title("Shack-Hartmann Wavefront sensor")

# %%
F = 60
T = 1/F
glob_cn = fried_parameter_from_Cn_squared(np.sum(cn[0,:]), wavelength=wavelength)
print(glob_cn)

#============================
# Pantalla de fase Basada en OOPAO
#============================

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
layer1 = InfiniteVonKarman(input_grid=Shimm.pupil_grid,
                          height=0e3,
                          reference_wavelength=wavelength,
                          Cn_squared=cn[0,0],
                          direction=45,
                          fps=F,
                          n_extra_pixels=2,
                          device=device,
                          L0=25,
                          l0=1e-20,
                          speed=6,#<-- velocidad
                          D_tel=telescope_diameter,
                          N = int(Shimm.sensorSize),
                          )

layer2 = InfiniteVonKarman(input_grid=Shimm.pupil_grid,
                          height=4e3,
                          reference_wavelength=wavelength,
                          Cn_squared=cn[0,1],
                          direction=30,
                          fps=int(1/T),
                          n_extra_pixels=2,
                          L0=25,
                          l0=1e-20,
                          speed=5,             #<-- velocidad
                          device=device,
                          D_tel=telescope_diameter * Shimm.oversizing_factor,
                          N = int(Shimm.sensorSize))

# device = "cuda" if torch.cuda.is_available() else "cpu"
layer3 = InfiniteVonKarman(input_grid=Shimm.pupil_grid,
                          height=12e3,
                          Cn_squared=cn[0,2],
                          direction=60,
                          reference_wavelength=wavelength,
                          fps=int(1/T),
                          n_extra_pixels=2,
                          device = device,
                          L0=25,
                          l0=1e-20,
                          speed=5,#<-- velocidad
                          D_tel=telescope_diameter* Shimm.oversizing_factor,
                          N =int( Shimm.sensorSize))

layer4 = InfiniteVonKarman(input_grid=Shimm.pupil_grid,
                          height=20e3,
                          reference_wavelength=wavelength,
                          Cn_squared=cn[0,3],
                          direction=90,
                          fps=int(1/T),
                          n_extra_pixels=2,
                          L0=25,
                          l0=1e-20,
                          speed=5,
                          device=device,
                          D_tel=telescope_diameter* Shimm.oversizing_factor,
                          N =int( Shimm.sensorSize))

# =======================================================
# Propagar pantallas de fase mediante angular spectrum
# =======================================================

# atm = LocalMultiLayerAtmosphere([layer1],scintillation=True) #<--- atmosfera multicapa
atm = LocalMultiLayerAtmosphere([layer1,layer2,layer3,layer4],scintillation=True) #<--- atmosfera multicapa
#===========================
# Creacion de detector sin ruido
#===========================
# %%
# camera = Shimm.NoislessCamera()

#===========================
# Mascara con sub-aperturas activas sin vigneting
#===========================

mask = np.array([
                [0,0,1,1,0,0],
                [0,1,1,1,1,0],
                [1,1,0,0,1,1],
                [1,1,0,0,1,1],
                [0,1,1,1,1,0],
                [0,0,1,1,0,0],
                ])

#===========================
# segundos de medicion
#===========================

# %%
steps = F*30

#============================
#  Matrices de autocovarianza
#============================

matx = np.zeros([steps, 28])
maty = np.zeros([steps, 28])
matf = np.zeros([steps, 28])

count = 0
#=======================
# Adquicision de datos
#=======================

layer1.update_cn2(cn[count,0])
layer2.update_cn2(cn[count,1])
layer3.update_cn2(cn[count,2])
layer4.update_cn2(cn[count,3])
glob_cn = fried_parameter_from_Cn_squared(np.sum(cn[count,:]), wavelength=wavelength)

for t in tqdm(range(1, steps)):
       camera.integrate((shwfs(magnifier(atm(wf)))),1)     # captura con tiempo de expocision de 2 ms
       wfs_image = camera.read_out()                       # Generacion de imagen
       slopes, flux, _= shwfse.estimate([wfs_image], 0.0)  # type: ignore <--Estimacion
       slopes = (slopes - slope_ref) * Shimm.m2r           # type: ignore <--Conversion a rad

       matx[t,:] = slopes[0,:]
       maty[t,:] = slopes[1,:]
       matf[t,:] = flux

       atm.evolve()

       if t % F == 0:
            layer1.speed += np.random.randint(-2,2)
            layer1.direction += np.random.randint(-30,30)

            if layer1.speed == 0:
                layer1.speed += 3

            if layer1.speed > 8:
                layer1.speed -= 3

            layer2.direction += np.random.randint(-30,30)
            layer2.speed += np.random.randint(-3,3)

            if layer2.speed == 0:
                layer2.speed += 4

            if layer2.speed > 12:
                layer2.speed -= 3

            layer3.direction += np.random.randint(-30,30)
            layer3.speed += np.random.randint(-2,2)

            if layer3.speed == 0:
                layer3.speed += 3

            if layer3.speed > 8:
                layer3.speed -= 2

            layer4.direction += np.random.randint(-30,30)
            layer4.speed += np.random.randint(-2,2)

            if layer4.speed == 0:
                layer4.speed += 3

            if layer4.speed > 8:
                layer4.speed -= 2
            atm = LocalMultiLayerAtmosphere([layer1,layer2,layer3,layer4],scintillation=True) #<--- atmosfera multicapa

# %%
steps = F*30

#============================
#  Matrices de autocovarianza
#============================

matx1 = np.zeros([steps, len(slope_ref[0,:])])
maty1 = np.zeros([steps, len(slope_ref[0,:])])
matf1 = np.zeros([steps, len(slope_ref[0,:])])
print(matx1.shape)

count = 0
#=======================
# Adquicision de datos
#=======================
layer1.speed = 8
layer2.speed = 8
layer3.speed = 8
layer4.speed = 8
layer1.update_cn2(cn[count,0])
layer2.update_cn2(cn[count,1])
layer3.update_cn2(cn[count,2])
layer4.update_cn2(cn[count,3])
glob_cn = fried_parameter_from_Cn_squared(np.sum(cn[count,:]), wavelength=wavelength)
atm = LocalMultiLayerAtmosphere([layer1,layer2,layer3,layer4],scintillation=True) #<--- atmosfera multicapa

for t in tqdm(range(steps)):

    camera.integrate((shwfs(magnifier(atm(wf)))),2e-3)  # captura con tiempo de expocision de 2 ms
    wfs_image = camera.read_out()                       # Generacion de imagen
    slopes, flux, _= shwfse.estimate([wfs_image], 0.0)  # type: ignore <--Estimacion
    slopes = ((slopes) - (slope_ref)) * Shimm.m2r          # type: ignore <--Conversion a rad

    matx1[t,:] = slopes[0,:]
    maty1[t,:] = slopes[1,:]
    matf1[t,:] = flux

    atm.evolve()
    if t % F == 0:
        layer1.speed += np.random.randint(-1,1)
        layer1.direction += np.random.randint(-30,30)

        if layer1.speed == 0:
            layer1.speed += 3
            print(layer1.speed)
        if layer1.speed > 8:
            layer1.speed -= 3
            print(layer1.speed)

        layer2.direction += np.random.randint(-30,30)
        layer2.speed += np.random.randint(-1,1)

        if layer2.speed == 0:
            layer2.speed += 3
            print(layer2.speed)

        if layer2.speed > 8:
            layer2.speed -= 3
            print(layer2.speed)

        layer3.direction += np.random.randint(-30,30)
        layer3.speed += np.random.randint(-1,1)

        if layer3.speed == 0:
            layer3.speed += 3
            print(layer3.speed)

        if layer3.speed > 8:
            layer3.speed -= 2
            print(layer3.speed)

        layer4.direction += np.random.randint(-30,30)
        layer4.speed += np.random.randint(-1,1)

        if layer4.speed == 0:
            layer4.speed += 3
            print(layer4.speed)

        if layer4.speed > 8:
            layer4.speed -= 2
            print(layer4.speed)
        atm = LocalMultiLayerAtmosphere([layer1,layer2,layer3,layer4],scintillation=True) #<--- atmosfera multicapa

# %%
def SlopeDenoise(pcov):
    row_mean = pcov.mean(axis=1, keepdims=True)  # (nsubtot,1,2)
    col_mean = pcov.mean(axis=0, keepdims=True)  # (1,nsubtot,2)
    glob_mean = pcov.mean(axis=(0, 1), keepdims=True)  # (1,1,2)
    return pcov - row_mean - col_mean + glob_mean

# print(cns)
# %%
flu = np.mean(matf1, axis=0)
matff = (matf1 - flu) / flu

n = 10
print(steps)
varianza = steps/n
print(varianza)
nn = len(slope_ref[0,:])*len(slope_ref[0,:])*3
b0 = np.zeros([n,nn])
for i in tqdm(range(n)):

    inx = np.cov((matx1[int(i*varianza):int((i+1)*varianza),:]),rowvar = False).ravel()
    iny = np.cov((maty1[int(i*varianza):int((i+1)*varianza),:]),rowvar = False).ravel()
    ini = np.cov(matff[int(i*varianza):int((i+1)*varianza),:],rowvar = False).ravel()
    b0[i,:] = np.concat([ini,inx,iny])
# inx = np.cov(v1,rowvar = False).ravel()/ np.sqrt(n)
# iny = np.cov(v2,rowvar = False).ravel()/ np.sqrt(n)
# ini = np.cov(v3,rowvar = False).ravel()/ np.sqrt(n)
# b0 = np.concat([ini,inx,iny])
# %%

# b1 = np.zeros([n,nn])
# b1[0,:] = (W@cn[0,:])
# for i in range(n-1):
    # b1[i+1,:] = W@cn[cns[i],:]

# sigma = (b1/b0).mean(axis=0)

# sigma = np.cov(b0, rowvar=False, ddof=1)/n
sigma = b0.std(axis=0, ddof=1)/np.sqrt(n)
# sigma = 1/b0
plt.figure()
# plt.plot(sigma)
plt.plot(1/sigma)
# np.save("sigma1.npy", sigma)
# %%
flu = np.mean(matf1, axis=0)
matff1 = (matf1 - flu) / flu

pcovx = np.cov(matx1,rowvar = False)
pcovy = np.cov(maty1,rowvar = False)
pcovf = np.cov(matff1,rowvar = False)

pcovx2 = SlopeDenoise(pcovx)
pcovy2 = SlopeDenoise(pcovy)

plt.figure(figsize=[10,10])
plt.subplot(1,3,1)
plt.imshow(pcovx2)
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(pcovy2)
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(pcovf)
plt.colorbar()

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
# cf = pcovf[~np.eye(pcovf.shape[0], dtype=bool)]
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

# %%
W_np = W
c_np = b

y = cn[count,:]
E = np.diag(1/sigma)

sol = lsq_linear(
    E@W_np,
    E@c_np,
    tol=0,
    bounds=(0, np.inf),
    method="bvls"
)
j = sol.x
print("j original:")
print(j)
print(y)



# %%

total1 = np.sum(j)
total2 = np.sum(y)

from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="resultados estimacion")

table.add_column("parametro")
table.add_column("reales")
table.add_column("estimacion")
table.add_column("error relativo")

r0est = fried_parameter_from_Cn_squared(total1,wavelength=wavelength)
r0real =fried_parameter_from_Cn_squared(total2,wavelength=wavelength)

table.add_row("r0", f"{r0real:.3f} [m]", f"{r0est:.3f} [m]", f"{np.abs(r0real-r0est)/r0real:.3f} [-]")
table.add_row("h1: 0km " , f"{y[0]:.3e}" + r"[m^-3]", f"{j[0]:.3e}" + r"[m^-3]" ,f"{np.abs(y[0]-j[0])/y[0]:.3f} [-]")
table.add_row("h2: 4km " , f"{y[1]:.3e}" + r"[m^-3]", f"{j[1]:.3e}" + r"[m^-3]" ,f"{np.abs(y[1]-j[1])/y[1]:.3f} [-]")
table.add_row("h3: 12km ", f"{y[2]:.3e}" + r"[m^-3]", f"{j[2]:.3e}" + r"[m^-3]" ,f"{np.abs(y[2]-j[2])/y[2]:.3f} [-]")
table.add_row("h4: 20km ", f"{y[3]:.3e}" + r"[m^-3]", f"{j[3]:.3e}" + r"[m^-3]" ,f"{np.abs(y[3]-j[3])/y[3]:.3f} [-]")
table.add_row(
    "mse", f"{np.mean((j - y) ** 2):.3e} [-]"
)
console.print(table)

%matplotlib qt
plt.figure()
plt.plot(y,j, "g-o",linewidth=3)
plt.xlabel(r"$C_n^2$ Ground-Truth", fontsize=16)
plt.ylabel(r"$C_n^2$ Estimated", fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Estimation vs Ground-Truth values", fontsize=19)
condicion = np.linalg.cond(E@W )
print(f"Número de condición de la matriz teórica: {condicion}")
