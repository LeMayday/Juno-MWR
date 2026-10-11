# modules
import numpy as np
import JupiterMag as jm

# local files
from synchrotron_map import RJ
from mshell_map import THREE_D_NDArray, FOUR_D_NDArray

# default
import argparse
import concurrent.futures
import os
import sys

MAX_WORKERS = max(1, os.cpu_count() - 2)
RJ_polar_ratio = 66854 / RJ

def init_jm_config():
    jm.Con2020.Config(equation_type='analytic')
    jm.Internal.Config(Model="jrm33", CartesianIn=True, CartesianOut=True)


def B(X: THREE_D_NDArray, Y: THREE_D_NDArray, Z: THREE_D_NDArray) -> FOUR_D_NDArray:
    """
    Compute JRM33 + Con2020 B-field
    """
    assert X.shape == Y.shape == Z.shape
    XYZ = np.stack([X, Y, Z], axis=-1)
    # B-field model only operates on inputs as 1D vectors. This ensures contiguous memory layout
    XYZ_flat = np.ascontiguousarray(XYZ.reshape(-1, 3), dtype=np.float32)
    Bx_int, By_int, Bz_int = jm.Internal.Field(XYZ_flat[:, 0], XYZ_flat[:, 1], XYZ_flat[:, 2])
    Bx_ext, By_ext, Bz_ext = jm.Con2020.Field(XYZ_flat[:, 0], XYZ_flat[:, 1], XYZ_flat[:, 2])
    B_list = np.stack((Bx_int + Bx_ext, By_int + By_ext, Bz_int + Bz_ext), axis=-1).astype(np.float32)
    return B_list.reshape((*X.shape, 3))    # reshape back to match input


def batch_equator(R: FOUR_D_NDArray) -> FOUR_D_NDArray:
    """
    Compute equator positions, Mshell, and longitude for given batch of positions
    (with x,y,z along last axis).
    """
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


def chunk_and_flatten_3D(X: THREE_D_NDArray, Y: THREE_D_NDArray, Z: THREE_D_NDArray, n_splits: int) -> list[FOUR_D_NDArray]:
    """
    Break 3D domain into chunks (n_splits along each dimension) and return 1D of chunks (last dimension contiguous).
    """
    XYZ = np.stack([X, Y, Z], axis=-1)
    flat_chunks = [
        chunk
        for x_chunk in np.array_split(XYZ, n_splits, axis=0)       # Spatial X
        for y_chunk in np.array_split(x_chunk, n_splits, axis=1)   # Spatial Y
        for chunk in np.array_split(y_chunk, n_splits, axis=2)     # Spatial Z
    ]
    return flat_chunks


def recombine_chunked_data(flat_results: list[FOUR_D_NDArray], n_splits: int) -> FOUR_D_NDArray:
    """
    Recombine chunked data into original domain
    """
    list_of_xs = []
    for i in range(n_splits):
        idx_x = i * n_splits**2
        list_of_ys = []
        for j in range(n_splits):
            idx_j = j * n_splits
            idx_start = idx_x + idx_j
            list_of_ys.append(np.concatenate(flat_results[idx_start : idx_start + n_splits], axis=2))   # combine across z
        list_of_xs.append(np.concatenate(list_of_ys, axis=1))                                           # combine across y
    return np.concatenate(list_of_xs, axis=0)                                                           # combine across x


def tracefield_chunked(X: THREE_D_NDArray, Y: THREE_D_NDArray, Z: THREE_D_NDArray, n_splits: int) -> FOUR_D_NDArray:
    """
    Splits 3D coordinate grids (X, Y, Z) into n_splits along each axis, evaluates
    TraceField on each chunk, and concatenates the resulting .equator fields.
    """
    flat_chunks = chunk_and_flatten_3D(X, Y, Z, n_splits)

    # keep this code in case multithreading doesn't work
    # flat_results = []
    # for i, chunk in enumerate(flat_chunks):
    #     flat_results.append(batch_equator(chunk))

    with concurrent.futures.ProcessPoolExecutor(max_workers=12, initializer=init_jm_config) as executor:
        flat_results = list(executor.map(batch_equator, flat_chunks))

    return recombine_chunked_data(flat_results, n_splits)


def generate_Bfield_mesh(M_max: float, N: float = 120):
    x_vec = np.linspace(-M_max, M_max, N+1)
    y_vec = np.linspace(-M_max, M_max, N+1)
    z_vec = np.linspace(-M_max*2/3, M_max*2/3, N*2//3 + 1)
    X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')

    B_mesh = B(X, Y, Z)     # these positions are SIII!

    num_bytes = (N+1) * (N+1) * (N*2//3 + 1) * 1000 * 8     # total size of X,Y,Z x 1000 pts per trace x 8 bytes per float64
    n_splits = int(np.ceil((num_bytes / 2E9)**(1/3)))       # aim for 2 GB: num_bytes / n_splits**3 ~= 2 GB
    n_splits *= 2                                           # be more conservative since extra memory could be used by TraceField object
    eq_data = tracefield_chunked(X, Y, Z, n_splits)         # these positions are SIII! --- eq_data is domain x (x3, y3, z3, M, lon) along last axis, lon in radians
    B_mesh_eq = B(eq_data[..., 0], eq_data[..., 1], eq_data[..., 2])
    Mshell = eq_data[..., -2]
    dipole_lon = eq_data[..., -1]
    dipole_xy = np.stack((np.cos(dipole_lon), np.sin(dipole_lon)), axis=-1)     # dipole_lon is dicontinuous and not interpolatable

    jupiter_mask = X**2 + Y**2 + (Z / RJ_polar_ratio)**2 <= 1
    B_mesh[jupiter_mask, :] = np.nan
    B_mesh_eq[jupiter_mask, :] = np.nan
    Mshell[jupiter_mask] = np.nan
    dipole_xy[jupiter_mask, :] = np.nan

    np.savez(f"B-mesh_Mmax-{M_max}_N-{N}.npz", x_vec=x_vec, y_vec=y_vec, z_vec=z_vec,
             B_mesh=B_mesh, B_mesh_eq=B_mesh_eq, Mshell=Mshell, dipole_xy=dipole_xy)  # all are float32


def test_chunking_and_recombining():
    N = 12
    x_vec = np.linspace(-12, 12, N+1)
    y_vec = np.linspace(-10, 10, N+1)
    z_vec = np.linspace(-12*2/3, 12*2/3, N*2//3 + 1)
    X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')
    n_splits = 2
    XYZ_test = recombine_chunked_data(chunk_and_flatten_3D(X, Y, Z, n_splits), n_splits)
    XYZ = np.stack((X, Y, Z), axis=-1)
    assert np.count_nonzero(XYZ != XYZ_test) == 0
    print("chunking_and_recombining test passed!")


def test_Bfield_trace_chunking():
    # Note: chunking JMag produces a different output than not chunking. Google AI claims this is due to
    # JMag picking numerical integration steps based on what values are present in the given vector of positions.
    # I haven't been able to verify this, but some of the discrepancies probably won't matter? (e.g. at a point,
    # one method finds Mshell 160, one finds 0. This is too high to matter anyway)
    N = 12
    x_vec = np.linspace(-12, 12, N+1)
    y_vec = np.linspace(-10, 10, N+1)
    z_vec = np.linspace(-12*2/3, 12*2/3, N*2//3 + 1)
    X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')
    n_splits = 2
    eq_data = tracefield_chunked(X, Y, Z, n_splits)
    Mshell_chunked = eq_data[..., -2]
    Mshell_chunked[Mshell_chunked < 0.5] = np.nan
    Mshell_chunked = np.nan_to_num(Mshell_chunked)
    R = np.stack((X, Y, Z), axis=-1)
    Mshell_not_chunked = batch_equator(R)[..., -2]
    Mshell_not_chunked[Mshell_not_chunked < 0.5] = np.nan
    Mshell_not_chunked = np.nan_to_num(Mshell_not_chunked)
    assert np.count_nonzero(~np.isclose(Mshell_chunked, Mshell_not_chunked)) == 0
    print("Bfield_trace_chunking test passed!")


def main():
    if len(sys.argv) == 1:
        test_chunking_and_recombining()
        test_Bfield_trace_chunking()
    parser = argparse.ArgumentParser()
    parser.add_argument("--Mmax", required=True, type=float, help="Max RJ distance of mesh")
    parser.add_argument('-N', "--N", required=False, default=120, type=int, help="Number of mesh intervals in x and y dimensions")
    args = parser.parse_args()
    generate_Bfield_mesh(args.Mmax, args.N)


if __name__ == "__main__":
    init_jm_config()
    main()
