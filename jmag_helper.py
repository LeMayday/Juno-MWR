# modules
import numpy as np
import JupiterMag as jm

# local files
from synchrotron_map import RJ

# default
import argparse

TWO_D_NDArray = np.ndarray[tuple[int, int], np.dtype[np.float32]]
RJ_polar_ratio = 66854 / RJ

def init_jm_config():
    jm.Con2020.Config(equation_type='analytic')
    jm.Internal.Config(Model="jrm33", CartesianIn=True, CartesianOut=True)


def B(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Bx_int, By_int, Bz_int = jm.Internal.Field(X, Y, Z)
    Bx_ext, By_ext, Bz_ext = jm.Con2020.Field(X, Y, Z)
    return np.stack((Bx_int + Bx_ext, By_int + By_ext, Bz_int + Bz_ext), axis=-1).astype(np.float32)


def generate_Bfield_mesh(M_max: float, N: float = 120):
    x_vec = np.linspace(-M_max, M_max, N+1)
    y_vec = np.linspace(-M_max, M_max, N+1)
    z_vec = np.linspace(-M_max*2/3, M_max*2/3, N*2//3 + 1)
    X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')

    B_mesh = B(X, Y, Z)     # these positions are SIII!

    TFeq = jm.TraceField(X, Y, Z, Verbose=False, IntModel='jrm33', ExtModel='Con2020', MaxStep=0.01).equator    # these positions are SIII!
    B_mesh_eq = B(TFeq.x3, TFeq.y3, TFeq.z3)
    Mshell = np.array(TFeq.mshell, dtype=np.float32)
    dipole_lon = np.rad2deg(np.array(TFeq.mlone, dtype=np.float32)) + 180   # longitudes in degrees!
    points = np.stack(x_vec, y_vec, z_vec, axis=0)

    jupiter_mask = X**2 + Y**2 + (Z / RJ_polar_ratio)**2 <= 1
    B_mesh[jupiter_mask, :] = np.nan
    B_mesh_eq[jupiter_mask, :] = np.nan
    Mshell[jupiter_mask, :] = np.nan
    dipole_lon[jupiter_mask, :] = np.nan

    np.savez("B-mesh_Mmax-{M_max}_N-{N}.npz", points=points, B_mesh=B_mesh, B_mesh_eq=B_mesh_eq, Mshell=Mshell, dipole_lon=dipole_lon)  # all are float32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Mmax", required=True, type=float, help="Max RJ distance of mesh")
    parser.add_argument('-N', "--N", required=False, default=120, type=int, help="Number of mesh intervals in x and y dimensions")
    args = parser.parse_args()
    generate_Bfield_mesh(args.Mmax, args.N)


if __name__ == "__main__":
    init_jm_config()
    main()
