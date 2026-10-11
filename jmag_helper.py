# modules
import numpy as np
import JupiterMag as jm

# local files
from synchrotron_map import RJ

# default
import argparse
import concurrent.futures
import os

MAX_WORKERS = max(1, os.cpu_count() - 2)
TWO_D_NDArray = np.ndarray[tuple[int, int], np.dtype[np.float32]]
RJ_polar_ratio = 66854 / RJ

def init_jm_config():
    jm.Con2020.Config(equation_type='analytic')
    jm.Internal.Config(Model="jrm33", CartesianIn=True, CartesianOut=True)


def B(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    assert X.shape == Y.shape == Z.shape
    XYZ = np.stack([X, Y, Z], axis=-1)
    # B-field model only operates on inputs as 1D vectors. This ensures contiguous memory layout
    XYZ_flat = np.ascontiguousarray(XYZ.reshape(-1, 3), dtype=np.float32)
    Bx_int, By_int, Bz_int = jm.Internal.Field(XYZ_flat[:, 0], XYZ_flat[:, 1], XYZ_flat[:, 2])
    Bx_ext, By_ext, Bz_ext = jm.Con2020.Field(XYZ_flat[:, 0], XYZ_flat[:, 1], XYZ_flat[:, 2])
    B_list = np.stack((Bx_int + Bx_ext, By_int + By_ext, Bz_int + Bz_ext), axis=-1).astype(np.float32)
    return B_list.reshape((*X.shape, 3))    # reshape back to match input


def batch_equator(R: np.ndarray) -> np.ndarray:
    # r represents positions, with x,y,z, along last axis
    # B-field model only operates on inputs as 1D vectors. This ensures contiguous memory layout
    R_flat = np.ascontiguousarray(R.reshape(-1, 3), dtype=np.float32)
    Teq = jm.TraceField(R_flat[:, 0], R_flat[:, 1], R_flat[:, 2], Verbose=False, IntModel='jrm33', ExtModel='Con2020', MaxStep=0.1).equator

    out_flat = np.empty((R_flat.shape[0], 5), dtype=np.float32)     # out is size of R with x3, y3, z3, M, lon fields
    out_flat[:, 0] = Teq.x3
    out_flat[:, 1] = Teq.y3
    out_flat[:, 2] = Teq.z3
    out_flat[:, 3] = Teq.mshell
    out_flat[:, 4] = np.mod(Teq.mlone, 2*np.pi)     # mlone returns lon in radians from [-pi, pi]. (lon + 2pi) % 2pi
    return out_flat.reshape((*R.shape[:-1], 5))     # reshape back to original R shape

def tracefield_chunked(X, Y, Z, n_splits=3):
    """
    Splits 3D coordinate grids (X, Y, Z) into n_splits along each axis (default 3x3x3=27),
    evaluates TraceField on each chunk, and concatenates the resulting .equator fields.
    """
    XYZ = np.stack([X, Y, Z], axis=-1)

    flat_chunks = [
        chunk
        for x_chunk in np.array_split(XYZ, n_splits, axis=0)       # Spatial X
        for y_chunk in np.array_split(x_chunk, n_splits, axis=1)   # Spatial Y
        for chunk in np.array_split(y_chunk, n_splits, axis=2)     # Spatial Z
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_jm_config) as executor:
        flat_results = list(executor.map(batch_equator, flat_chunks))

    # Create an explicit object array to hold chunks of varying shapes
    obj_grid = np.empty(len(flat_results), dtype=object)
    obj_grid[:] = flat_results
    obj_grid_3d = obj_grid.reshape(n_splits, n_splits, n_splits)

    # np.block handles varying spatial chunk shapes seamlessly
    return np.block(obj_grid_3d)


def generate_Bfield_mesh(M_max: float, N: float = 120):
    x_vec = np.linspace(-M_max, M_max, N+1)
    y_vec = np.linspace(-M_max, M_max, N+1)
    z_vec = np.linspace(-M_max*2/3, M_max*2/3, N*2//3 + 1)
    X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')

    B_mesh = B(X, Y, Z)     # these positions are SIII!

    num_bytes = (N+1) * (N+1) * (N*2//3 + 1) * 1000 * 8     # total size of X,Y,Z x 1000 pts per trace x 8 bytes per float64
    n_splits = int(np.ceil((num_bytes / 2E9)**(1/3)))       # aim for 2 GB: num_bytes / n_splits**3 ~= 2 GB
    eq_data = tracefield_chunked(X, Y, Z, n_splits)         # these positions are SIII! --- eq_data is domain x (x3, y3, z3, M, lon) along last axis, lon in radians
    B_mesh_eq = B(eq_data[..., 0], eq_data[..., 1], eq_data[..., 2])
    Mshell = eq_data[..., -2]
    dipole_lon = eq_data[..., -1]
    dipole_xy = np.stack((np.cos(dipole_lon), np.sin(dipole_lon)), axis=-1)     # dipole_lon is dicontinuous and not interpolatable
    points = np.stack(x_vec, y_vec, z_vec, axis=0)

    jupiter_mask = X**2 + Y**2 + (Z / RJ_polar_ratio)**2 <= 1
    B_mesh[jupiter_mask, :] = np.nan
    B_mesh_eq[jupiter_mask, :] = np.nan
    Mshell[jupiter_mask] = np.nan
    dipole_xy[jupiter_mask, :] = np.nan

    np.savez("B-mesh_Mmax-{M_max}_N-{N}.npz", points=points, B_mesh=B_mesh, B_mesh_eq=B_mesh_eq, Mshell=Mshell, dipole_xy=dipole_xy)  # all are float32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Mmax", required=True, type=float, help="Max RJ distance of mesh")
    parser.add_argument('-N', "--N", required=False, default=120, type=int, help="Number of mesh intervals in x and y dimensions")
    args = parser.parse_args()
    generate_Bfield_mesh(args.Mmax, args.N)


if __name__ == "__main__":
    init_jm_config()
    main()
