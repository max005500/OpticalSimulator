import time
from dataclasses import InitVar, dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as sk
import torch
import torch.nn.functional as F
from aotools.turbulence import ft_sh_phase_screen
from scipy import linalg
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv


def translationImageMatrix(shift):
    # translate the image with the corresponding shift value
    tf_shift = sk.SimilarityTransform(translation=shift)
    return tf_shift


def globalTransformation(image, shiftMatrix, order=3):
    output = sk.warp(image, (shiftMatrix).inverse, order=order)
    return output


# Exact von Kármán covariance (as in AOTools)
def turb_phase_covariance(r, r0, L0):
    r = r + 1e-40
    A = (L0 / r0) ** (5.0 / 3.0)
    B1 = (2 ** (-5.0 / 6.0)) * gamma(11.0 / 6.0) / (np.pi ** (8.0 / 3.0))
    B2 = ((24.0 / 5.0) * gamma(6.0 / 5.0)) ** (5.0 / 6.0)
    x = (2.0 * np.pi * r) / L0
    C = x ** (5.0 / 6.0) * kv(5.0 / 6.0, x)
    C[np.isnan(C)] = ((2.0 * np.pi * 1e-40) / L0) ** (5.0 / 6.0) * kv(
        5.0 / 6.0, (2.0 * np.pi * 1e-40) / L0
    )
    return A * B1 * B2 * C


@dataclass
class InfinitePhaseScreenGenerator:
    N: int = 128
    D_tel: float = 8.0
    r0: float = 0.15
    L0: float = 25.0
    l0: float = 0.01
    init_phase: np.ndarray | None = None
    device: str = "cpu"
    seed: int | None = None
    wind_dir_deg: float = 0.0
    wind_speed: float = 20  # wind speed in m/seg
    n_extra_pixel: int = 2
    pixel_size: float | None = None
    fps: int = 1000

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if name in {"wind_dir_deg", "wind_speed", "fps"}:
            attrs_ready = all(
                hasattr(self, attr) for attr in ("wind_dir_deg", "wind_speed", "fps")
            )

            if attrs_ready:
                self.update_wind()

    def __post_init__(self):

        if self.pixel_size == None:
            self.pixel_size = self.D_tel / self.N

        # Persistent RNG on correct device
        self._rng = torch.Generator(device=self.device)

        if self.seed:
            self._rng.manual_seed(self.seed)

        # Resolution fit to crop area
        self.final_N = self.N

        if self.init_phase is not None:
            self.OPD = self.init_phase
        else:
            self.OPD = ft_sh_phase_screen(
                self.r0, self.N, self.D_tel / self.N, self.L0, self.l0, seed=self.seed
            )

        vy = self.wind_speed * np.cos(np.deg2rad(self.wind_dir_deg))
        vx = self.wind_speed * np.sin(np.deg2rad(self.wind_dir_deg))

        samplingTiem = 1 / self.fps

        self.ps_turb_x = samplingTiem * vx
        self.ps_turb_y = samplingTiem * vy

        ext_size = self.N + self.n_extra_pixel

        # Outer ring of pixel for the phase screens update
        self.outerMask = np.ones([ext_size, ext_size], dtype=np.bool)
        pad = int(self.n_extra_pixel // 2)
        padD = int(pad + self.N)

        self.outerMask[pad:padD, pad:padD] = False

        # inner pixels that contains the phase screens
        self.innerMask = np.ones([ext_size, ext_size], dtype=np.bool)

        self.innerMask[self.outerMask] = False

        self.innerMask[
            1 + self.n_extra_pixel : -1 - self.n_extra_pixel,
            1 + self.n_extra_pixel : -1 - self.n_extra_pixel,
        ] = False

        # x = np.linspace(0, self.N + 1, self.N + 2) * self.D_tel / (self.N - 1)
        pixel_size = self.D_tel / (self.N - 1)
        x = np.arange(ext_size) * pixel_size
        u, v = np.meshgrid(x, x)

        inner_coords = np.column_stack([u[self.innerMask != 0], v[self.innerMask != 0]])
        outer_coords = np.column_stack([u[self.outerMask != 0], v[self.outerMask != 0]])

        self.rho0 = cdist(inner_coords, inner_coords)
        self.rho1 = cdist(inner_coords, outer_coords)
        self.rho2 = cdist(outer_coords, outer_coords)

        # Build A/B matrices (NumPy)
        self._build_ab_matrices()

        # Convert A, B to torch tensors on self.device (ensure float32)
        self.A_mat = torch.from_numpy(self.A_mat).float().to(self.device)
        self.B_mat = torch.from_numpy(self.B_mat).float().to(self.device)

        self.mapShift = np.zeros(
            [self.N + self.n_extra_pixel, self.N + self.n_extra_pixel]
        )
        self.mapShift[self.outerMask]
        # Draw a new Gaussian vector on the correct device

        zsv = self.OPD[self.innerMask[pad:padD, pad:padD]]
        zs = torch.from_numpy(zsv).float().to(self.device)

        # Compute the new row
        b = torch.randn(self.B_mat.shape[1], generator=self._rng, device=self.device)
        X = self.A_mat.matmul(zs) + self.B_mat.matmul(b)

        # self.mapShift = torch.from_numpy(self.mapShift).float().to(self.device)
        X = X.cpu().numpy()
        self.mapShift[self.outerMask] = X
        self.mapShift[self.outerMask == False] = np.reshape(self.OPD, self.N * self.N)
        # self.B_mat = self.A_mat

        self.notDoneOnce = True

    def _build_ab_matrices(self):
        self.Czz = turb_phase_covariance(self.rho0, self.r0, self.L0)
        print("czz")
        self.Czx = turb_phase_covariance(self.rho1, self.r0, self.L0)
        print("czx")
        self.Cxz = self.Czx.T
        self.Cxx = turb_phase_covariance(self.rho2, self.r0, self.L0)
        print("cxx")

        with torch.no_grad():
            Czz_torch = torch.from_numpy(self.Czz).to(self.device)
            ide = torch.from_numpy(np.identity(self.Czz.shape[0])).to(self.device)
            invCzz_torch = torch.linalg.lstsq(Czz_torch, ide).solution
            self.invCzz = invCzz_torch.detach().cpu().numpy()

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # invCzz = linalg.pinv(Czz)
        self.ab_matrix()

    def ab_matrix(self):
        self.A_mat = self.Cxz.dot(self.invCzz)

        BBt = self.Cxx - self.A_mat.dot(self.Czx)
        U, W, _ = np.linalg.svd(BBt)
        self.B_mat = U.dot(np.diag(np.sqrt(W)))

    def update_r0(self, r0):
        self.Czz = self.Czz * (self.r0 / r0) ** (5 / 3)
        self.Cxz = self.Cxz * (self.r0 / r0) ** (5 / 3)
        self.Czx = self.Czx * (self.r0 / r0) ** (5 / 3)
        self.Cxx = self.Cxx * (self.r0 / r0) ** (5 / 3)
        self.invCzz = self.invCzz / ((self.r0 / r0) ** (5 / 3))
        self.r0 = r0
        self.ab_matrix()
        self.B_mat = torch.from_numpy(self.B_mat).float().to(self.device)
        self.A_mat = torch.from_numpy(self.A_mat).float().to(self.device)

    def add_row(self, stepInPixel):
        pad = int(self.n_extra_pixel // 2)
        padD = int(pad + self.N)

        map_full = self.mapShift
        shiftMatrix = translationImageMatrix(
            [stepInPixel[0], stepInPixel[1]]
        )  # units are in pixel of the M1
        tmp = globalTransformation(map_full, shiftMatrix)
        onePixelShiftedPhaseScreen = tmp[pad:padD, pad:padD]

        z = onePixelShiftedPhaseScreen[self.innerMask[pad:padD, pad:padD] != 0]
        Z = torch.from_numpy(z).float().to(self.device)

        b = torch.randn(self.B_mat.shape[1], generator=self._rng, device=self.device)
        X = self.A_mat.matmul(Z) + self.B_mat.matmul(b)
        X = X.cpu().numpy()

        map_full[self.outerMask != 0] = X
        map_full[self.outerMask == 0] = np.reshape(
            onePixelShiftedPhaseScreen, self.N * self.N
        )

        return onePixelShiftedPhaseScreen

    def update_wind(self):
        vy = self.wind_speed * np.cos(np.deg2rad(self.wind_dir_deg))
        vx = self.wind_speed * np.sin(np.deg2rad(self.wind_dir_deg))

        samplingTiem = 1 / self.fps

        self.ps_turb_x = samplingTiem * vx
        self.ps_turb_y = samplingTiem * vy
        self.notDoneOnce = True

    def evolve(self):

        if self.notDoneOnce:
            self.notDoneOnce = False
            self.ratio = np.zeros(2)
            self.ratio[0] = self.ps_turb_x / self.pixel_size
            self.ratio[1] = self.ps_turb_y / self.pixel_size
            self.buff = np.zeros(2)

        ratio = self.ratio
        tmpRatio = np.abs(ratio)
        tmpRatio[np.isinf(tmpRatio)] = 0
        nScreens = tmpRatio
        nScreens = nScreens.astype("int")
        stepInPixel = np.zeros(2)
        stepInSubPixel = np.zeros(2)

        for _ in range(nScreens.min()):
            stepInPixel[0] = 1
            stepInPixel[1] = 1
            stepInPixel = stepInPixel * np.sign(ratio)
            self.OPD = self.add_row(stepInPixel)

        for _ in range(nScreens.max() - nScreens.min()):
            stepInPixel[0] = 1
            stepInPixel[1] = 1
            stepInPixel = stepInPixel * np.sign(ratio)
            stepInPixel[np.where(nScreens == nScreens.min())] = 0
            self.OPD = self.add_row(stepInPixel)

        stepInSubPixel[0] = (np.abs(ratio[0]) % 1) * np.sign(ratio[0])
        stepInSubPixel[1] = (np.abs(ratio[1]) % 1) * np.sign(ratio[1])
        self.buff += stepInSubPixel

        if np.abs(self.buff[0]) >= 1 or np.abs(self.buff[1]) >= 1:
            stepInPixel[0] = 1 * np.sign(self.buff[0])
            stepInPixel[1] = 1 * np.sign(self.buff[1])
            stepInPixel[np.where(np.abs(self.buff) < 1)] = 0
            self.OPD = self.add_row(stepInPixel)

        self.buff[0] = (np.abs(self.buff[0]) % 1) * np.sign(self.buff[0])
        self.buff[1] = (np.abs(self.buff[1]) % 1) * np.sign(self.buff[1])

        shiftMatrix = translationImageMatrix(
            [self.buff[0], self.buff[1]]
        )  # units are in pixel of the M1

        pad = int(self.n_extra_pixel // 2)
        padD = int(pad + self.N)
        self.OPD = globalTransformation(self.mapShift, shiftMatrix)[pad:padD, pad:padD]
        return self.OPD
