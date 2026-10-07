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


def B(X, Y, Z):
    Bx_int, By_int, Bz_int = jm.Internal.Field(X, Y, Z)
    Bx_ext, By_ext, Bz_ext = jm.Con2020.Field(X, Y, Z)
    return np.stack(Bx_int + Bx_ext, By_int + By_ext, Bz_int + Bz_ext, axis=-1).astype(np.float32)


def trace_batch_mshell(r: TWO_D_NDArray):
    T = jm.TraceField(*r.T, IntModel='jrm33', ExtModel='Con2020')
    mshells = np.array(T.equator.mshell, dtype=np.float32)
    return mshells


def find_lats_M(phi_vec, M, tol=1e-3):
    theta = np.arcsin(np.sqrt(1/M))                                 # r/R = 1 = M cos^2(lat)
    theta_vec = np.full_like(phi_vec, theta)

    n = 0
    prev_res = np.zeros_like(phi_vec)
    damping = np.ones_like(phi_vec)
    while True:
        x0 = np.cos(phi_vec) * np.sin(theta_vec)
        y0 = np.sin(phi_vec) * np.sin(theta_vec)
        z0 = np.cos(theta_vec)
        M_trace = jm.TraceField(x0, y0, z0, Verbose=True, IntModel='jrm33', ExtModel='Con2020').equator.mshell
        res = M_trace - M
        if np.max(np.abs(res)) < tol:
            print(f"Newton-Raphson completed in {n} iterations.")
            break
        meets_tol = np.abs(res) < tol
        overshoot = (prev_res * res) < 0
        damping[overshoot] *= 0.5                                   # fixes oscillations near magnetic great red spot
        step = res * np.tan(theta_vec) / (2 * M) * damping          # dM/dtheta
        theta_vec[~meets_tol] += np.clip(step[~meets_tol], -0.1, 0.1)
        prev_res = np.copy(res)
        n += 1

    return theta_vec


def pre_compute_mshell_traces(M: float, ntraces: int = 100) -> jm.TraceField:
    jm.Con2020.Config(equation_type='analytic')
    phi = np.linspace(0, 2*np.pi, ntraces, endpoint=False)
    theta = find_lats_M(phi, M)
    x0 = np.cos(phi) * np.sin(theta)
    y0 = np.sin(phi) * np.sin(theta)
    z0 = np.cos(theta)
    # see https://github.com/mattkjames7/JupiterMag/blob/a3fc24f20e0860296a11a55ee14f0e5f5e8fc577/JupiterMag/TraceField.py#L16 for args
    return jm.TraceField(x0, y0, z0, Verbose=False, IntModel='jrm33', ExtModel='Con2020', MaxStep=0.1)


def generate_Bfield_mesh(M_max: float, N: float = 120):
    x_vec = np.linspace(-M_max, M_max, N+1)
    y_vec = np.linspace(-M_max, M_max, N+1)
    z_vec = np.linspace(-M_max*2/3, M_max*2/3, N*2//3 + 1)
    X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij', dtype=np.float32)

    B_mesh = B(X, Y, Z)

    TFeq = jm.TraceField(X, Y, Z, Verbose=False, IntModel='jrm33', ExtModel='Con2020', MaxStep=0.01).equator
    B_mesh_eq = B(TFeq.x3, TFeq.y3, TFeq.z3)
    Mshell = np.array(TFeq.mshell, dtype=np.float32)
    dipole_lon = np.rad2deg(np.array(TFeq.mlone, dtype=np.float32)) + 180
    r_mesh = np.stack(X, Y, Z, axis=-1)

    jupiter_mask = X**2 + Y**2 + (Z / RJ_polar_ratio)**2 <= 1
    B_mesh[jupiter_mask, :] = np.nan
    B_mesh_eq[jupiter_mask, :] = np.nan
    Mshell[jupiter_mask, :] = np.nan
    dipole_lon[jupiter_mask, :] = np.nan

    np.savez("B-mesh_Mmax-{M_max}_N-{N}.npz", r_mesh=r_mesh, B_mesh=B_mesh, B_mesh_eq=B_mesh_eq, Mshell=Mshell, dipole_lon=dipole_lon)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Mmax", required=True, type=float, help="Max RJ distance of mesh")
    parser.add_argument('-N', "--N", required=False, default=120, type=int, help="Number of mesh intervals in x and y dimensions")
    args = parser.parse_args()
    generate_Bfield_mesh(args.Mmax, args.N)


if __name__ == "__main__":
    init_jm_config()
    main()
