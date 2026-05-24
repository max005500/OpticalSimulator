from dataclasses import dataclass
import torch
import skimage.transform as sk
import numpy as np
from scipy import linalg
from scipy.special import kv, gamma
import torch.nn.functional as F
from aotools.turbulence import ft_sh_phase_screen
import numpy as np
from dataclasses import dataclass
import torch

def translationImageMatrix(image,shift):
    # translate the image with the corresponding shift value
    tf_shift = sk.SimilarityTransform(translation=shift)    
    return tf_shift

def globalTransformation(image,shiftMatrix,order=3):
        output  = sk.warp(image,(shiftMatrix).inverse,order=order)
        return output

def bsxfunMinus(a,b):      
    A =np.tile(a[...,None],len(b))
    B =np.tile(b[...,None],len(a))
    out = A-B.T
    return out

# Exact von Kármán covariance (as in AOTools)
def turb_phase_covariance(r, r0, L0):
    r = r + 1e-40
    A  = (L0 / r0)**(5.0/3.0)
    B1 = (2**(-5.0/6.0)) * gamma(11.0/6.0) / (np.pi**(8.0/3.0))
    B2 = ((24.0/5.0) * gamma(6.0/5.0))**(5.0/6.0)
    x = (2.0 * np.pi * r) / L0
    C = x**(5.0/6.0) * kv(5.0/6.0, x)
    C[np.isnan(C)] = ((2.0*np.pi*1e-40)/L0)**(5.0/6.0) * kv(5.0/6.0, (2.0*np.pi*1e-40)/L0)
    return A * B1 * B2 * C


@dataclass
class InfiniteVonKarmanPhaseScreenGenerator:
    N: int =128
    D_tel: float =8.0
    r0: float =0.15
    L0: float =25.0
    l0: float =0.01
    n_columns: float=2
    init_phase: any | None =None
    device: str ='cpu'
    seed: int|None = None,
    wind_dir_deg: float =0.0      # ← direction in degress x layer
    wind_speed: float =20        # wind speed in m/seg
    n_extra_pixel: int = 2
    pixel_size: float | None = None
    fps: int = 1000

    
    def __post_init__(self):

        if self.pixel_size == None:
            self.pixel_size = self.D_tel/self.N

        # Persistent RNG on correct device
        self._rng = torch.Generator(device=self.device)
        if self.seed is not None:
            self._rng.manual_seed(self.seed)

        # Resolution fit to crop area
        self.final_N = self.N

        if self.init_phase is not None:
            self.OPD = self.init_phase
        else:
            self.OPD = ft_sh_phase_screen(self.r0,self.N,self.D_tel/self.N,self.L0,self.l0, seed=self.seed) 

        vy = self.wind_speed * np.cos(np.deg2rad(self.wind_dir_deg))
        samplingTiem = 1/self.fps
        vx = self.wind_speed * np.sin(np.deg2rad(self.wind_dir_deg))
        self.ps_turb_x = samplingTiem * vx
        self.ps_turb_y = samplingTiem * vy

        ext_size = self.N + self.n_extra_pixel 
        # Outer ring of pixel for the phase screens update
        self.outerMask = np.ones(
                            [ext_size,ext_size],
                            dtype=np.bool)

        self.outerMask[1:-1, 1:-1] = False

        # inner pixels that contains the phase screens
        self.innerMask = np.ones(
                                [ext_size,ext_size]
                                 , dtype=np.bool)

        self.innerMask[self.outerMask] = False

        self.innerMask[
            1 + self.n_extra_pixel : -1 - self.n_extra_pixel,
            1 + self.n_extra_pixel : -1 - self.n_extra_pixel
            ] = False
        x = np.linspace(0, self.N+1, self.N + 2) * self.D_tel/(self.N-1)
        u, v = np.meshgrid(x, x)

        innerZ = u[self.innerMask != 0] + 1j*v[self.innerMask != 0]
        outerZ = u[self.outerMask != 0] + 1j*v[self.outerMask != 0]

        self.rho0 = np.abs(bsxfunMinus(innerZ, innerZ))
        self.rho1 = np.abs(bsxfunMinus(innerZ, outerZ))
        self.rho2 = np.abs(bsxfunMinus(outerZ, outerZ))

        # Build A/B matrices (NumPy)
        self._build_ab_matrices()

        # Convert A, B to torch tensors on self.device (ensure float32)
        self.A_mat = torch.from_numpy(self.A_mat).float().to(self.device)
        self.B_mat = torch.from_numpy(self.B_mat).float().to(self.device)

        self.mapShift = np.zeros([self.N+self.n_extra_pixel, self.N+self.n_extra_pixel])
        self.mapShift[self.outerMask]
        # Draw a new Gaussian vector on the correct device

        zsv = self.OPD[self.innerMask[1:-1,1:-1] ]
        zs = torch.from_numpy(zsv).float().to(self.device)

        # Compute the new row
        b = torch.randn(self.B_mat.shape[1], generator=self._rng, device=self.device)
        X = self.A_mat.matmul(zs) + self.B_mat.matmul(b) 

        # self.mapShift = torch.from_numpy(self.mapShift).float().to(self.device)
        X = X.cpu().numpy()
        self.mapShift[self.outerMask] = X
        self.mapShift[self.outerMask == False] = np.reshape(self.OPD, self.N*self.N)

        self.notDoneOnce = True

    def _build_ab_matrices(self):
        coords = np.column_stack(np.where(self.innerMask))
        Czz = turb_phase_covariance(self.rho0,self.r0,self.L0)
        print("czz")
        Czx = turb_phase_covariance(self.rho1,self.r0,self.L0)
        print("czx")
        Cxz = Czx.T
        Cxx = turb_phase_covariance(self.rho2,self.r0,self.L0)
        print("cxx")
        n = coords.shape[0]

        try:
            cf = linalg.cho_factor(Czz) 
            invCzz = linalg.cho_solve(cf, np.eye(n))
        except linalg.LinAlgError:
            invCzz = np.linalg.pinv(Czz)

        self.A_mat = Cxz.dot(invCzz)
        BBt = Cxx - self.A_mat.dot(Czx)
        U, W, _ = np.linalg.svd(BBt)
        self.B_mat = U.dot(np.diag(np.sqrt(W)))

    def add_row(self,stepInPixel):
        map_full = self.mapShift
        shiftMatrix = translationImageMatrix(map_full, [stepInPixel[0], stepInPixel[1]])  # units are in pixel of the M1
        tmp = globalTransformation(map_full, shiftMatrix)
        onePixelShiftedPhaseScreen = tmp[1:-1, 1:-1]

        z = onePixelShiftedPhaseScreen[self.innerMask[1:-1, 1:-1] != 0]
        Z = torch.from_numpy(z).float().to(self.device)

        b = torch.randn(self.B_mat.shape[1], generator=self._rng, device=self.device)
        X = self.A_mat.matmul(Z) + self.B_mat.matmul(b) 
        X = X.cpu().numpy()

        map_full[self.outerMask != 0] = X
        map_full[self.outerMask == 0] = np.reshape(
            onePixelShiftedPhaseScreen, self.N*self.N)

        return onePixelShiftedPhaseScreen 

    def evolve(self):

        if self.notDoneOnce:
            self.notDoneOnce = False
            self.ratio = np.zeros(2)
            self.ratio[0] = self.ps_turb_x/self.pixel_size
            self.ratio[1] = self.ps_turb_y/self.pixel_size
            self.buff = np.zeros(2)

        ratio = self.ratio
        tmpRatio = np.abs(ratio)
        tmpRatio[np.isinf(tmpRatio)] = 0
        nScreens = (tmpRatio)
        nScreens = nScreens.astype('int')
        stepInPixel = np.zeros(2)
        stepInSubPixel = np.zeros(2)


        for _ in range(nScreens.min()):
            stepInPixel[0] = 1
            stepInPixel[1] = 1
            stepInPixel = stepInPixel*np.sign(ratio)
            self.OPD = self.add_row(stepInPixel)

        for _ in range(nScreens.max()-nScreens.min()):
            stepInPixel[0] = 1
            stepInPixel[1] = 1
            stepInPixel = stepInPixel*np.sign(ratio)
            stepInPixel[np.where(nScreens == nScreens.min())] = 0
            self.OPD = self.add_row(stepInPixel)


        stepInSubPixel[0] = (np.abs(ratio[0]) % 1)*np.sign(ratio[0])
        stepInSubPixel[1] = (np.abs(ratio[1]) % 1)*np.sign(ratio[1])
        self.buff += stepInSubPixel


        if np.abs(self.buff[0]) >= 1 or np.abs(self.buff[1]) >= 1:
            stepInPixel[0] = 1*np.sign(self.buff[0])
            stepInPixel[1] = 1*np.sign(self.buff[1])
            stepInPixel[np.where(np.abs(self.buff) < 1)] = 0
            self.OPD = self.add_row(stepInPixel)
        
        self.buff[0] = (np.abs(self.buff[0]) % 1)*np.sign(self.buff[0])
        self.buff[1] = (np.abs(self.buff[1]) % 1)*np.sign(self.buff[1])

        shiftMatrix = translationImageMatrix(
           self.mapShift, [self.buff[0], self.buff[1]])  # units are in pixel of the M1

        self.OPD = globalTransformation(
        self.mapShift, shiftMatrix)[1:-1, 1:-1]
        return self.OPD
# === Von Kármán Phase Screen Generator ===
class KolmogorovGenerator:
    def __init__(self, N=128, D_tel=1, r0=0.1, L0=25, l0=0.01,pupil_mask=None, wavelength=None, batch_size=1, device='cuda', seed=None):
        self.device = device
        self.N = N
        self.D_tel = D_tel
        self.r0 = r0
        self.L0 = L0
        self.l0 = l0
        self.wavelength = wavelength
        self.batch_size = batch_size
        self.seed = seed
        self.pupil_mask = pupil_mask.to(device) if pupil_mask is not None else None
        self._build_grids()

    def _build_grids(self):
        N = self.N
        self.delta = self.D_tel / N
        self.D = N * self.delta

        del_f = 1.0 / (N * self.delta)
        fx = torch.linspace(-N / 2, N / 2 - 1, N, device=self.device) * del_f
        self.fx, self.fy = torch.meshgrid(fx, fx, indexing='xy')
        self.f = torch.sqrt(self.fx**2 + self.fy**2)

        coords = torch.linspace(-self.D / 2, self.D / 2 - self.delta, N, device=self.device)
        x, y = torch.meshgrid(coords, coords, indexing='xy')
        self.x = x.unsqueeze(0).expand(self.batch_size, -1, -1)
        self.y = y.unsqueeze(0).expand(self.batch_size, -1, -1)

    def update_parameters(self, r0=None, L0=None, l0=None, D_tel=None, batch_size=None, pupil_mask=None):
        if r0 is not None: self.r0 = r0
        if L0 is not None: self.L0 = L0
        if l0 is not None: self.l0 = l0
        if D_tel is not None: self.D_tel = D_tel
        if batch_size is not None: self.batch_size = batch_size
        if pupil_mask is not None: self.pupil_mask = pupil_mask.to(self.device)
        self._build_grids()

    def _apply_pupil_mask(self, phs_tensor):
        if self.pupil_mask is not None:
            return phs_tensor * self.pupil_mask
        return phs_tensor

    def _ifft2_batch(self, G, delta_f):
        N = G.shape[-1]
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(G, dim=(-2, -1)), dim=(-2, -1))) * (N * delta_f) ** 2

    def generate_random_phase(self):
        N = self.N
        r0, L0, l0 = self.r0, self.L0, self.l0
        del_f = 1.0 / (N * self.delta)


        PSD_phi = (0.023 * (r0 ** (-5. / 3.)) * (self.f ** (-11. / 6.)))
        PSD_phi[N//2, N//2] = 0
        PSD_phi = PSD_phi.expand(self.batch_size, -1, -1)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        cn = (torch.randn((self.batch_size, N, N), device=self.device) +
              1j * torch.randn((self.batch_size, N, N), device=self.device)) * torch.sqrt(PSD_phi) * del_f

        phs = self._ifft2_batch(cn, 1).real
        if self.wavelength is not None:
            phs = phs * (self.wavelength * 1e9) / (2 * np.pi)

        return self._apply_pupil_mask(phs.unsqueeze(1))

    def generate_subharmonic_phase(self):
        N = self.N
        phs_lo = torch.zeros((self.batch_size, N, N), dtype=torch.cfloat, device=self.device)

        for p in range(1, 4):
            del_f = 1.0 / (3 ** p * self.D)
            base = torch.tensor([-1, 0, 1], device=self.device) * del_f
            fx, fy = torch.meshgrid(base, base, indexing='xy')
            f = torch.sqrt(fx ** 2 + fy ** 2)

            fm = 5.92 / self.l0 / (2 * np.pi)
            f0 = 1.0 / self.L0

            PSD_phi = (0.023 * (self.r0 ** (-5. / 3.)) * (f ** (-11. / 6.)))
            PSD_phi[1, 1] = 0

            cn = (torch.randn((self.batch_size, 3, 3), device=self.device) +
                  1j * torch.randn((self.batch_size, 3, 3), device=self.device)) * torch.sqrt(PSD_phi)[None, :, :] * del_f

            for i in range(3):
                for j in range(3):
                    phase = 2 * np.pi * (fx[i, j] * self.x + fy[i, j] * self.y)
                    phs_lo += cn[:, i, j].unsqueeze(-1).unsqueeze(-1) * torch.exp(1j * phase)

        phs_lo = phs_lo.real - phs_lo.real.mean(dim=(1, 2), keepdim=True)

        if self.wavelength is not None:
            phs_lo = phs_lo * (self.wavelength * 1e9) / (2 * np.pi)

        return self._apply_pupil_mask(phs_lo.unsqueeze(1))

    def generate_total_phase(self):
        phs_hi = self.generate_random_phase()
        phs_lo = self.generate_subharmonic_phase()
        return (phs_hi + phs_lo)
