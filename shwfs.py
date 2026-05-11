from hcipy import make_rectangular_aperture,\
                  CartesianGrid,\
                  ShackHartmannWavefrontSensorOptics,\
                  WavefrontSensorOptics,OpticalElement,\
                  PhaseApodizer,\
                  AngularSpectrumPropagator,\
                  Field, OpticalSystem,\
                  SeparatedCoords
import numpy as np

class shPropagatorTest(WavefrontSensorOptics):
    def __init__(self, input_grid, micro_lens_array):
        # Make propagator
        sh_prop = AngularSpectrumPropagator(input_grid, micro_lens_array.focal_length)

        # Make optical system
        OpticalSystem.__init__(self, (micro_lens_array, sh_prop))
        self.mla_index = micro_lens_array.mla_index
        self.mla_grid = micro_lens_array.mla_grid
        self.micro_lens_array = micro_lens_array


class MicroLensArray(OpticalElement):
    '''A parabolic micro-lens array.

    Parameters
    ----------
    input_grid : Grid
        The grid on which the micro-lens array is evaluated.
    lenslet_grid : Grid
        A grid containing the lenslet positions.
    focal_length : scalar
        The focal length of the micro-lenses
    lenslet_shape : field generator
        The shape of a lenslet.
    '''
    def __init__(self, input_grid, lenslet_grid, focal_length,sh_length,lam, lenslet_shape=None):
        self.input_grid = input_grid
        self.focal_length = focal_length
        self.sh_length = sh_length
        self.mla_grid = lenslet_grid
        self.mla_opd = Field(np.zeros(self.input_grid.size), self.input_grid)  
        self.mla_index = Field(-np.ones(self.input_grid.size), self.input_grid)
        self.lam = lam
          
        for i, (x, y) in enumerate(self.mla_grid.as_('cartesian').points):  

            shifted_grid = self.input_grid.shifted((x, y))  
            grid_mask = make_rectangular_aperture(self.sh_length)(shifted_grid)
            mask = grid_mask != 0  
            self.mla_index[mask] = i

            # Fase paraxial como OPD  
            r2 = shifted_grid.x**2 + shifted_grid.y**2  
            k = 2*np.pi/self.lam  
            phi = -(k/(2*focal_length)) * r2  
              
            self.mla_opd[mask] = phi[mask]  
         
        self.mla_surface = PhaseApodizer(self.mla_opd)

    def forward(self, wavefront):
        return self.mla_surface.forward(wavefront)

    def backward(self, wavefront):
        return self.mla_surface.backward(wavefront)


class CenteredSquareShackHartmannWavefrontSensorOptics(ShackHartmannWavefrontSensorOptics):  
    def __init__(self, input_grid, f_number, num_lenslets, pupil_diameter,lam):  
        self.lam = lam
        lenslet_diameter = float(pupil_diameter) / num_lenslets  
        print(lenslet_diameter)
          
        # Centrar la grilla sumando medio diámetro de lenslet  
        x = np.arange(-pupil_diameter + lenslet_diameter/2,   
                      pupil_diameter - lenslet_diameter/2 + lenslet_diameter,   
                      lenslet_diameter)  

        self.mla_grid = CartesianGrid(SeparatedCoords((x, x)))  

        focal_length = f_number * lenslet_diameter   
          
        self.micro_lens_array = MicroLensArray(input_grid, self.mla_grid, focal_length,lenslet_diameter,self.lam)  
  
        shPropagatorTest.__init__(self, input_grid, self.micro_lens_array)


class UncenteredSquareShackHartmannWavefrontSensorOptics(ShackHartmannWavefrontSensorOptics):  
    def __init__(self, input_grid, f_number, num_lenslets, pupil_diameter,lam):  
        self.lam = lam
        lenslet_diameter = float(pupil_diameter) / num_lenslets  
        print(lenslet_diameter)
          
        x = np.arange(-pupil_diameter, pupil_diameter, lenslet_diameter)
        self.mla_grid = CartesianGrid(SeparatedCoords((x, x)))  

        focal_length = f_number * lenslet_diameter   
          
        self.micro_lens_array = MicroLensArray(input_grid, self.mla_grid, focal_length,lenslet_diameter,self.lam)  
  
        shPropagatorTest.__init__(self, input_grid, self.micro_lens_array)
