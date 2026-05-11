from shwfs import CenteredSquareShackHartmannWavefrontSensorOptics, UncenteredSquareShackHartmannWavefrontSensorOptics
from hcipy import make_pupil_grid, make_obstructed_circular_aperture,evaluate_supersampled, imshow_field, Magnifier
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
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

    def __post_init__(self):
        self.beam_diameter: float = self.fach/self.telescope_fnumber
        self.npx = int(self.beam_diameter/self.pixelSize) 
        self.oversizing_factor =  self.sensorSize/self.npx          

        sensorDim = self.pixelSize * self.sensorSize               #<-- tamaño fisico del sensor
        pupil_grid_diameter = self.D_t * self.oversizing_factor    #<-- tamaño del sensor escalado a la apertura

        magnification = sensorDim / pupil_grid_diameter       #<-- relacion tamaño del sensor y tamaño de apertura
        self.magnifier = Magnifier(magnification)

        pupil_grid = make_pupil_grid(self.sensorSize, pupil_grid_diameter)
        self.sensor_grid = make_pupil_grid(self.sensorSize, sensorDim)

        aperture_func = make_obstructed_circular_aperture(self.D_t,
                        self.centralObstruction, num_spiders=0)

        self.aperture = evaluate_supersampled(aperture_func, pupil_grid, 4)

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








