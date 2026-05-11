from hcipy import OpticalElement, fried_parameter_from_Cn_squared, Field, AngularSpectrumPropagator
import numpy as np
from atmosfera import InfiniteVonKarmanPhaseScreenGenerator


class LocalAtmosphere(OpticalElement):
    def __init__(self, input_grid, Cn_squared=None, L0=np.inf,
                  height=0., direction=0.):

        self.input_grid = input_grid

        self._Cn_squared = None
        self._outer_scale = None

        self.Cn_squared = Cn_squared
        self.L0 = L0

        self.direction = direction
        self.height = height

    def evolve(self):
        '''Evolve the atmospheric layer until time `t`.

        Parameters
        ----------
        t : scalar
            The time to which to evolve the atmospheric layer.
        '''
        raise NotImplementedError()


    @property
    def Cn_squared(self):
        return self._Cn_squared

    @Cn_squared.setter
    def Cn_squared(self, Cn_squared):
        raise NotImplementedError()

    @property
    def outer_scale(self):
        return self._outer_scale

    @outer_scale.setter
    def outer_scale(self, L0):
        raise NotImplementedError()

    @property
    def L0(self):
        return self.outer_scale

    @L0.setter
    def L0(self, L0):
        self.outer_scale = L0

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        if np.isscalar(velocity):
            self._velocity = np.array([velocity, 0.])
        else:
            self._velocity = np.array(velocity, dtype=float)

    @property
    def output_grid(self):
        return self.input_grid

    def phase_for(self, wavelength):
        raise NotImplementedError()

    def forward(self, wavefront):
        wf = wavefront.copy()
        wf.electric_field *= np.exp(1j * self.phase_for(wf.wavelength))
        return wf

    def backward(self, wavefront):
        wf = wavefront.copy()
        wf.electric_field *= np.exp(-1j * self.phase_for(wf.wavelength))
        return wf


class LocalMultiLayerAtmosphere(OpticalElement):
    '''A multi-layer atmospheric model.

    This :class:`OpticalElement` can model turbulence and scintillation effects
    due to atmospheric turbulence by propagating light through a series of
    infinitely-thin atmospheric phase screens at different altitudes. The distance
    between two phase screens can be propagated using Fresnel propagation, or using
    no :class:`Propagator`.

    Parameters
    ----------
    layers : list of AtmosphericLayer objects
        The series of atmospheric layers in this model.
    scintillation : bool
        If True, then the distance between two phase screens is propagated using
        a :class:`FresnelPropagator`. Otherwise, no propagator will be used.
    '''
    def __init__(self, layers, scintillation=False, scintilation=None):
        # Retain backwards compatibility.
        if scintilation is not None:
            import warnings
            warnings.warn('Please use the correct spelling for scintillation.')

            scintillation = scintilation

        self.layers = layers
        self._scintillation = scintillation
        self._dirty = True

        self.calculate_propagators()

    def calculate_propagators(self):
        '''Recalculates the list of optical elements used for a propagation.

        This function is called automatically by other functions, but a recalculation
        can be forced by calling it explicitly.
        '''
        heights = np.array([l.height for l in self.layers])
        layer_indices = np.argsort(-heights)

        sorted_heights = heights[layer_indices]
        delta_heights = sorted_heights[:-1] - sorted_heights[1:]
        grid = self.layers[0].input_grid

        if self.scintillation:
            propagators = [AngularSpectrumPropagator(grid, h) for h in delta_heights]

        self.elements = []
        for i, j in enumerate(layer_indices):
            self.elements.append(self.layers[j])
            if self.scintillation and i < len(propagators):
                self.elements.append(propagators[i])

        if self.scintillation and sorted_heights[-1] > 0:
            self.elements.append(AngularSpectrumPropagator(grid, sorted_heights[-1]))

        self._dirty = False

    def reset(self):
        for l in self.layers:
            l.reset()

    @property
    def layers(self):
        '''A list of :class:`AtmosphericLayer` objects.
        '''
        return self._layers

    @layers.setter
    def layers(self, layers):
        self._layers = layers
        self._dirty = True

    def phase_for(self, wavelength):
        '''Get the unwrapped phase in radians for the atmosphere.

        Parameters
        ----------
        wavelength : scalar
            The wavelength at which to calculate the phase screen.

        Returns
        -------
        Field
            The total unwrapped phase screen.
        '''
        if self.scintillation:
            raise ValueError('Cannot get the unwrapped phase for an atmosphere with scintillation.')

        unwrapped_phases = [layer.phase_for(wavelength) for layer in self.layers]

        return Field(np.sum(unwrapped_phases, axis=0), unwrapped_phases[-1].grid)

    @property
    def scintillation(self):
        '''Whether to include scintillation effects in the propagation.
        '''
        return self._scintillation

    @scintillation.setter
    def scintillation(self, scintillation):
        self._dirty = scintillation != self.scintillation
        self._scintillation = scintillation

    def evolve(self):
        '''Evolve all atmospheric layers to a time t.

        Parameters
        ----------
        t : scalar
            The time to which to evolve the atmospheric layers.
        '''
        for l in self.layers:
            l.evolve()

    @property
    def Cn_squared(self):  # noqa: N802
        '''The total Cn^2 value of the simulated atmosphere.
        '''
        return np.sum([l.Cn_squared for l in self.layers])

    @Cn_squared.setter
    def Cn_squared(self, Cn_squared):  # noqa: N802
        old_Cn_squared = self.Cn_squared
        for l in self.layers:
            l.Cn_squared = l.Cn_squared / old_Cn_squared * Cn_squared

    @property
    def outer_scale(self):
        '''The outer scale of all layers.
        '''
        return self.layers[0].outer_scale

    @outer_scale.setter
    def outer_scale(self, L0):
        for l in self.layers:
            l.outer_scale = L0

    def forward(self, wavefront):
        if self._dirty:
            self.calculate_propagators()

        wf = wavefront.copy()
        for el in self.elements:
            wf = el.forward(wf)
        return wf

    def backward(self, wavefront):
        if self._dirty:
            self.calculate_propagators()

        wf = wavefront.copy()
        for el in reversed(self.elements):
            wf = el.backward(wf)
        return wf



class InfiniteVonKarman(LocalAtmosphere):
    def __init__(self, input_grid, fps: int, Cn_squared=None,
                 L0=25, l0=0.01, speed=0., height=0.,
                 direction=0., seed=None, N=256, D_tel=0.5,
                 reference_wavelength=500e-9):

        self.fps = fps
        self.l0 = l0
        self.seed = seed
        self.N = N
        self.D_tel = D_tel
        self.speed = speed
        self.L0 = L0
        self.reference_wavelength = reference_wavelength

        if Cn_squared is None:
            raise ValueError("Cn_squared no puede ser None si quieres calcular r0.")

        self.r0 = fried_parameter_from_Cn_squared(
            Cn_squared,
            wavelength=reference_wavelength
        )

        super().__init__(
            input_grid=input_grid,
            Cn_squared=Cn_squared,
            L0=L0,
            height=height,
            direction=direction
        )

        self.reset()

    @property
    def Cn_squared(self):
        return self._Cn_squared

    @Cn_squared.setter
    def Cn_squared(self, Cn_squared):
        self._Cn_squared = Cn_squared

    @property
    def outer_scale(self):
        return self._outer_scale

    @outer_scale.setter
    def outer_scale(self, L0):
        self._outer_scale = L0

    def reset(self, make_independent_realization=False):
        seed = None if make_independent_realization else self.seed


        self.dyn = InfiniteVonKarmanPhaseScreenGenerator(
            N=self.N,
            D_tel=self.D_tel,
            r0=self.r0,
            L0=self.L0,
            l0=self.l0,
            wind_speed=self.speed,
            wind_dir_deg=self.direction,
            fps=self.fps,
            seed=seed
        )

    def phase_for(self, wavelength):
        phase = self.dyn.OPD

        # Si opd viene como matriz 2D y HCIPy espera Field plano:
        if isinstance(phase, np.ndarray):
            phase = Field(phase.ravel(), self.input_grid)

        return phase

    def evolve(self):
        self.dyn.evolve()