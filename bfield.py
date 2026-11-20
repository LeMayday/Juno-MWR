import numpy as np
from scipy.ndimage import map_coordinates
from coordinates import lat_lonE
from global_vars import RJ_polar_ratio, MJ


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


class MeshInterpolator:
    def __init__(self, mesh_file: str):
        # construct RegularGridInterpolators for each mesh value
        mesh_data = np.load(mesh_file)
        x_vec = mesh_data["x_vec"]
        y_vec = mesh_data["y_vec"]
        z_vec = mesh_data["z_vec"]
        points = (x_vec, y_vec, z_vec)
        self.B     = UGLI(points=points, values=mesh_data["B_mesh"])
        self.Beq   = UGLI(points=points, values=mesh_data["B_mesh_eq"])
        self.M     = UGLI(points=points, values=mesh_data["Mshell"])
        self.Bin   = UGLI(points=points, values=mesh_data["B_mesh_in"])
        self.Bis   = UGLI(points=points, values=mesh_data["B_mesh_is"])
        self.__lonxy = UGLI(points=points, values=mesh_data["dipole_xy"])

    def lonxy(self, xi):
        """
        Interpolate magnetic longitudes.
        """
        interp_lonxy = self.__lonxy(xi)     # interpolate xy since these are continuous
        _, interp_lon = lat_lonE(interp_lonxy[..., 0], interp_lonxy[..., 1], 0)
        return interp_lon


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
