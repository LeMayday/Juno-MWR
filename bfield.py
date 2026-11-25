# modules
import numpy as np
from scipy.ndimage import map_coordinates
from JupiterMag.CoordConv import SIIItoMag

# local files
from coordinates import lat_lonE
from global_vars import RJ, RJ_polar_ratio, MJ

# default
from abc import ABC, abstractmethod
from typing import Literal


class UniformGridLinearInterpolator:
    # see https://docs.scipy.org/doc/scipy-1.14.1/tutorial/interpolate/ND_regular_grid.html
    def __init__(self, points, values):
        self.limits = np.array([[min(x), max(x)] for x in points])
        self.values = np.asarray(values, dtype=float)   # note this could be (..., 3) for a vector-valued function

    def __call__(self, xi):
        """
        xi here is an array-like (an array or a list) of points.

        Each "point" is an ndim-dimensional array_like, representing
        the coordinates of a point in ndim-dimensional space.
        """
        # transpose the xi array into the map_coordinates convention
        # which takes coordinates of a point along columns of a 2D array.
        xi = np.moveaxis(np.asarray(xi), -1, 0)

        if len(self.values.shape) == len(self.limits):  # if scalar function, add dummy axis
            self.values = np.expand_dims(self.values, -1)
        ns = self.values.shape[:-1]                     # dimensions of function domain
        n_values = self.values.shape[-1]                # dimensions of function value (1 if scalar, >1 if vector)
        out = np.zeros((*xi.shape[1:], n_values))       # output will be xi.shape[1:] for each value of the function
        for i in range(n_values):                       # loop through function scalar/vector components (for each dim...)
            # convert from data coordinates to pixel coordinates
            coords = [(n-1)*(val - lo) / (hi - lo) for val, n, (lo, hi) in zip(xi, ns, self.limits)]
            # interpolate
            out[..., i] = map_coordinates(self.values[..., i], coords, order=1, mode='constant', cval=np.nan)
        if out.shape[-1] == 1:   # scalar function
            return out[..., 0]
        return out

UGLI = UniformGridLinearInterpolator  # type alias


class BField(ABC):
    @abstractmethod
    def B(self, R: np.ndarray) -> np.ndarray:
        """
        R: vector of positions, with (x,y,z) along last axis

        Returns: vector of B components, similar shape as R, with (Bx,By,Bz) along last axis
        """
        pass
    @abstractmethod
    def Beq(self, R: np.ndarray) -> np.ndarray:
        """
        R: vector of positions, with (x,y,z) along last axis

        Returns: vector of B magnitudes, same shape as R
        """
        pass
    @abstractmethod
    def Bin(self, R: np.ndarray) -> np.ndarray:
        """
        R: vector of positions, with (x,y,z) along last axis

        Returns: vector of mapped B magnitudes at north ionosphere footprint, same shape as R
        """
        pass
    @abstractmethod
    def Bis(self, R: np.ndarray) -> np.ndarray:
        """
        R: vector of positions, with (x,y,z) along last axis

        Returns: vector of mapped B magnitudes at south ionosphere footprint, same shape as R
        """
        pass
    @abstractmethod
    def M(self, R: np.ndarray) -> np.ndarray:
        """
        R: vector of positions, with (x,y,z) along last axis

        Returns: vector of M-shell values, same shape as R
        """
        pass
    @abstractmethod
    def lonxy(self, R: np.ndarray) -> np.ndarray:
        """
        R: vector of positions, with (x,y,z) along last axis

        Returns: vector of magnetic longitudes as (x,y) coordinates in dipole frame, similar shape as R, with (x,y) along last axis
        """
        pass

class MeshInterpolator(BField):
    def __init__(self, mesh_file: str):
        # construct RegularGridInterpolators for each mesh value
        mesh_data = np.load(mesh_file)
        x_vec = mesh_data["x_vec"]
        y_vec = mesh_data["y_vec"]
        z_vec = mesh_data["z_vec"]
        points = (x_vec, y_vec, z_vec)
        self.__B     = UGLI(points=points, values=mesh_data["B_mesh"])
        self.__Beq   = UGLI(points=points, values=mesh_data["B_mesh_eq"])
        self.__M     = UGLI(points=points, values=mesh_data["Mshell"])
        self.__Bin   = UGLI(points=points, values=mesh_data["B_mesh_in"])
        self.__Bis   = UGLI(points=points, values=mesh_data["B_mesh_is"])
        self.__lonxy = UGLI(points=points, values=mesh_data["dipole_xy"])

    def B(self, R): return self.__B(R)

    def Beq(self, R): return self.__Beq(R)

    def Bin(self, R): return self.__Bin(R)

    def Bis(self, R): return self.__Bis(R)

    def M(self, R): return self.__M(R)

    def lonxy(self, xi):
        interp_lonxy = self.__lonxy(xi)     # interpolate xy since these are continuous
        _, interp_lon = lat_lonE(interp_lonxy[..., 0], interp_lonxy[..., 1], 0)
        return interp_lon


class DipoleField(BField):
    def __init__(self, config: Literal["NoTilt", "Jupiter"] = "NoTilt", m_dipole: float = None):
        """
        Tilt dipole field for standard configurations
        """
        self.config = config
        if config == "NoTilt":
            if m_dipole is None:
                raise ValueError("NoTilt dipole must specify m_dipole")
            self.rot_th, self.rot_phi = 0, 0
            self.m_dipole = m_dipole
        elif config == "Jupiter":
            self.rot_th, self.rot_phi = 10.25, 163.62   # see libjupitermag/src/trace.cc lines 30-31
            self.m_dipole = MJ * 1E5                    # [nT RJ^3]

    def __transform(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config == "NoTilt":
            return X, Y, Z
        Xm, Ym, Zm = SIIItoMag(X, Y, Z, self.rot_th, self.rot_phi)
        return Xm, Ym, Zm

    def B(self, R):
        X, Y, Z = self.__transform(R[..., 0], R[..., 1], R[..., 2])
        Bx, By, Bz = DipoleField.B_dipole(X, Y, Z, self.m_dipole)
        return np.stack((Bx, By, Bz), axis=-1)

    def Beq(self, R):
        X, Y, Z = self.__transform(R[..., 0], R[..., 1], R[..., 2])
        return DipoleField.Beq_dipole(X, Y, Z, self.m_dipole)

    def Bin(self, R):
        X, Y, Z = self.__transform(R[..., 0], R[..., 1], R[..., 2])
        return DipoleField.Bion_dipole(X, Y, Z, self.m_dipole)

    def Bis(self, R):
        return self.Bin(R)

    def M(self, R):
        X, Y, Z = self.__transform(R[..., 0], R[..., 1], R[..., 2])
        return DipoleField.__L_dipole(X, Y, Z)

    def lonxy(self, R):
        X, Y, _ = self.__transform(R[..., 0], R[..., 1], R[..., 2])
        _, lon = lat_lonE(X, Y, 0)
        return lon

    @staticmethod
    def B_dipole(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho2 = X**2 + Y**2
        r2 = rho2 + Z**2

        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(r2)
            r5 = r2**2 * r
            Bx = 3 * m * X * Z / r5
            By = 3 * m * Y * Z / r5
            Bz = m * (3 * Z**2 - r2) / r5

        mask = X**2 + Y**2 + (Z / RJ_polar_ratio)**2 < 1
        Bx[mask] = np.nan
        By[mask] = np.nan
        Bz[mask] = np.nan
        return Bx, By, Bz

    @staticmethod
    def __L_dipole(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        rho2 = X**2 + Y**2
        r3 = (rho2 + Z**2)**(3/2)
        return r3 / rho2

    @staticmethod
    def Beq_dipole(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, m: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            L = DipoleField.__L_dipole(X, Y, Z)
            Beq = m / L**3
        return Beq

    @staticmethod
    def Bion_dipole(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, m: float) -> np.ndarray:
        # calculates footprint assuming intersection with ellipsoid of height 500 km above Jupiter
        a = 1; c = RJ_polar_ratio; h = 500 / RJ
        b = (c+h)**2 - (a+h)**2
        with np.errstate(divide="ignore", invalid="ignore"):
            L = DipoleField.__L_dipole(X, Y, Z)
            sin2th = (-b + np.sqrt(b**2 + 4 * L**2 * (c+h)**2)) / (2 * L**2)
            Bion = m / L**3 * np.sqrt(4 - 3 * sin2th) / sin2th**3
        return Bion


def alpha(r_b: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    r_b: array of boresight vectors

    B: array of B vectors at positions
    """
    r_b_norm = r_b / np.linalg.norm(r_b, axis=-1, keepdims=True)
    b = B / np.linalg.norm(B, axis=-1, keepdims=True)
    return np.acos(np.einsum('ijk,ik->ij', b, r_b_norm))    # num samples x num LOS pts


def alpha_eq(alpha: np.ndarray, Beq: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    alpha: array of local pitch angles

    Beq: array of B magnitudes at magnetic equator mapped from positions

    B: array of B magnitudes at positions
    """
    # sin^2(alpha) / B = sin^2(alpha_eq) / Beq => sin(alpha_eq) = sin(alpha) * sqrt(Beq / B)
    inner = np.sin(alpha) * np.sqrt(Beq / B)
    clip_mask = np.abs(inner) < 1.1
    inner[clip_mask] = np.clip(inner[clip_mask], a_min=-1, a_max=1)
    return np.asin(inner)
