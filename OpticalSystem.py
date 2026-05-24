from shwfs import CenteredSquareShackHartmannWavefrontSensorOptics, UncenteredSquareShackHartmannWavefrontSensorOptics, LocalShackHartmannWavefrontSensorEstimator
from hcipy import make_pupil_grid, make_obstructed_circular_aperture,evaluate_supersampled, imshow_field, Magnifier, NoiselessDetector, NoisyDetector
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from scipy import ndimage
import numpy as np
from dataclasses import dataclass

@dataclass
class ShimmOptic:
    D_t: float
    centralObstruction: float
    fach: float 
    pixelSize: float
    sensorSize: int
    SHcentered: bool = True
    telescope_fnumber: float = 10
    focal_length: float = 15.3e-3

    def __post_init__(self):
        self.beam_diameter: float = self.fach/self.telescope_fnumber
        self.npx = int(self.beam_diameter/self.pixelSize) 
        self.oversizing_factor =  self.sensorSize/self.npx          

        sensorDim = self.pixelSize * self.sensorSize               #<-- tamaño fisico del sensor
        pupil_grid_diameter = self.D_t * self.oversizing_factor    #<-- tamaño del sensor escalado a la apertura

        magnification =  sensorDim / pupil_grid_diameter           #<-- relacion tamaño del sensor y tamaño de apertura
        self.magnifier = Magnifier(magnification)


        self.pupil_grid = make_pupil_grid(self.sensorSize, pupil_grid_diameter)#type: ignore
        self.sensor_grid = make_pupil_grid(self.sensorSize, sensorDim)#type: ignore

        aperture_func = make_obstructed_circular_aperture(self.D_t,
                        self.centralObstruction, num_spiders=0)

        self.aperture = evaluate_supersampled(aperture_func, self.pupil_grid, 4)
        self.aperture = evaluate_supersampled(aperture_func, self.pupil_grid, 4)

        self.m2r = (magnification/(self.focal_length)) * (180/np.pi) * 3600  #<-- conversion a Arcsec


    def pupil_info(self, graph: bool = False):

        console = Console()
        table = Table(title="Caracteristicas de telescopio")

        table.add_column("Parametro")
        table.add_column("Valores")
        table.add_row("Tamaño de la pupila", f"{self.npx} [pixels]")
        table.add_row("Tamaño del sensor",f"{self.sensorSize} [pixels]" )
        table.add_row("Tamaño fisico de la pupila",f"{self.D_t} [m]" )
        table.add_row("Tamaño del haz en el sensor",f"{self.beam_diameter * 1000} [mm]" )
        console.print(table)

        if graph: 
            plt.figure()
            imshow_field(self.aperture, cmap="gray")
            plt.xlabel("x position(m)")
            plt.ylabel("y position(m)")
            plt.colorbar()


    def ShackHartman(self, lenslet_diameter: float , num_lenslets: int, focal_length: float, wavelength: float):

        sh_diameter = lenslet_diameter * num_lenslets
        f_number = focal_length/lenslet_diameter 
        lenslets_visibles = int(self.beam_diameter / lenslet_diameter)

        if self.SHcentered:
          shwfs =  CenteredSquareShackHartmannWavefrontSensorOptics(input_grid=self.sensor_grid,f_number=f_number,num_lenslets=num_lenslets,pupil_diameter=sh_diameter, lam=wavelength)
        else:
          shwfs = UncenteredSquareShackHartmannWavefrontSensorOptics(input_grid=self.sensor_grid,f_number=f_number,num_lenslets=num_lenslets,pupil_diameter=sh_diameter, lam=wavelength)

        console = Console()
        table = Table(title="Shack hartman caracteristicas")

        table.add_column("Parametro")
        table.add_column("Valores")

        table.add_row("Cantidad de subaperturas visibles", f"{lenslets_visibles}")
        table.add_row("f_number Shack Hartmann",f"{f_number}" )
        table.add_row("Tamaño fisico",f"{sh_diameter} [m]" )
        table.add_row("centrado",f"{self.SHcentered}" )
        console.print(table)
          
        return shwfs

    def estimator(self, shwfs, image):

        shwfse = LocalShackHartmannWavefrontSensorEstimator(shwfs.mla_grid,shwfs.micro_lens_array.mla_index)

        fluxes = ndimage.measurements.sum_labels(image, shwfse.mla_index, shwfse.estimation_subapertures) # type: ignore
        flux_limit = fluxes.max() * 0.9

        estimation_subapertures = shwfs.mla_grid.zeros(dtype='bool')  
          
        indices_validos = shwfse.estimation_subapertures[fluxes > flux_limit].astype(int)  
        estimation_subapertures[indices_validos] = True    # type: ignore

        shwfse = LocalShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index, estimation_subapertures)

        return shwfse

    def spatial_autocov_fft(self,T, M, min_pairs=1):
        """
        Autocovarianza espacial 'full' (2N-1 x 2M-1) usando FFT con corrección por máscara.
        T: Valores en las pocisiones (N x M), valores en apagadas pueden ser cualquier cosa (se ignoran con M)
        M: máscara binaria (N x M), 1=válido, 0=missing
        numero minimo de pares: espaciamiento en metros
        """
        
        T = np.asarray(T, dtype=float)
        mask = np.asarray(M, dtype=bool)

        N, P = T.shape

        if mask.sum() == 0:
            raise ValueError("La máscara no tiene datos válidos.")

        # Media solo sobre datos válidos
        mu = T[mask].mean()

        # Campo centrado y enmascarado
        Y = np.zeros_like(T, dtype=float)
        Y[mask] = T[mask] - mu

        # Tamaño para correlación lineal full
        H, W = 2 * N - 1, 2 * P - 1

        Ypad = np.zeros((H, W)); Ypad[:N, :P] = Y
        Mpad = np.zeros((H, W)); Mpad[:N, :P] = M

        FY = np.fft.fft2(Ypad)
        FM = np.fft.fft2(Mpad.astype(float))

        # num = np.fft.ifft2(FY * np.conj(FY)).real
        # den = np.fft.ifft2(FM * np.conj(FM)).real

        num = np.fft.ifft2(np.abs(FY)**2).real
        den = np.fft.ifft2(np.abs(FM)**2).real

        # Centrar el lag (0,0) al medio
        num = np.fft.fftshift(num)
        den = np.fft.fftshift(den)

        # Corregir errores numéricos tipo 2.999999999 o 1e-15
        count = np.rint(den).astype(int) 

        C = np.full_like(num, np.nan, dtype=float)

        valid = count >= min_pairs
        C[valid] = num[valid] / count[valid]

        return C, count

    def tip_tilt_sub(self, cov: np.ndarray, pupil_mask: np.ndarray):
        """
            slodar_refFuncs2D():
          - proyect cov to sub-apertures pairs 
          - tip/tilt subtraction 
          - Re-bined 
        """

        nsubx = pupil_mask.shape[0]
        nn = 2 * nsubx - 1

        # índices de sub-aperturas activas
        active = [(i, j) for j in range(nsubx) for i in range(nsubx) if pupil_mask[j, i] > 0]
        nsubtot = len(active)

        psf = np.zeros((nn, nn), dtype=np.float64)

        # pcov: matriz completa (nsubtot x nsubtot) para x e y intercalados
        pcov = np.zeros([nsubtot, nsubtot], dtype=np.float64)

        # 1) llenar pcov tomando valores desde cov[2*delta] (x) y cov[2*delta+1] (y)
        for a, (i1, j1) in enumerate(active):
            for b, (i2, j2) in enumerate(active):
                di = i2 - i1 + (nsubx - 1)
                dj = j2 - j1 + (nsubx - 1)
                pcov[a, b] = cov[di, dj]

        # 2) tip/tilt subtraction: C' = C - rowMean - colMean + globalMean
        row_mean = pcov.mean(axis=1, keepdims=True)   # (nsubtot,1,2)
        col_mean = pcov.mean(axis=0, keepdims=True)   # (1,nsubtot,2)
        glob_mean = pcov.mean(axis=(0, 1), keepdims=True)  # (1,1,2)
        pcov2 = pcov - row_mean - col_mean + glob_mean

        # 3) rebin a separaciones (nn x nn) 
        acc = np.zeros((nn, nn), dtype=np.float64)
        cnt = np.zeros((nn, nn), dtype=np.int64)

        for a, (i1, j1) in enumerate(active):
            for b, (i2, j2) in enumerate(active):
                di = i2 - i1 + (nsubx - 1)
                dj = j2 - j1 + (nsubx - 1)

                acc[di, dj] += pcov2[a, b]
                cnt[di, dj] += 1

        # cnt>0
        m = cnt > 0
        psf[m] = (acc[m] / cnt[m])
        return psf, pcov2, pcov 

    def NoislessCamera(self):
        return NoiselessDetector(self.sensor_grid )

    def ZWOCamera(self):
        return NoisyDetector(self.sensor_grid,read_noise=7)

    def IRCamera(self):
        return NoisyDetector(self.sensor_grid, dark_current_rate=640, read_noise=37)


