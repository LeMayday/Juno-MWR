# modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import healpy as hp
import JupiterMag as jm
from JupiterMag import TraceField

# local files
from plot_data import make_subplots
from PDS_helper import load_PJ_data, NoProductsError, FileDownloadError, DownloadShortCircuitError
from coordinates import lat_lonW, lat_lonE
from synchrotron_map import parse_PJs, COLS_GRDR_MAP, RJ

# default
import argparse

TWO_D_NDArray = np.ndarray[tuple[int, int], np.dtype[np.float32]]
THREE_D_NDArray = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]


def B(X, Y, Z):
    jm.Con2020.Config(equation_type='analytic')
    jm.Internal.Config(Model="jrm33", CartesianIn=True, CartesianOut=True)
    Bx, By, Bz = jm.Internal.Field(X, Y, Z) + jm.Con2020.Field(X, Y, Z)
    return Bx, By, Bz


def find_lats_M(phi_vec, M, tol=1e-4):
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
        overshoot = (prev_res * res) < 0
        damping[overshoot] *= 0.5                                   # fixes oscillations near magnetic great red spot
        step = res * np.tan(theta_vec) / (2 * M) * damping          # dM/dtheta
        theta_vec += np.clip(step, -0.1, 0.1)
        prev_res = np.copy(res)
        n += 1
    
    return theta_vec


def pre_compute_mshell_traces(M: float, ntraces: int = 100) -> TraceField:
    jm.Con2020.Config(equation_type='analytic')
    phi = np.linspace(0, 2*np.pi, ntraces, endpoint=False)
    theta = find_lats_M(phi, M)
    x0 = np.cos(phi) * np.sin(theta)
    y0 = np.sin(phi) * np.sin(theta)
    z0 = np.cos(theta)
    # see https://github.com/mattkjames7/JupiterMag/blob/a3fc24f20e0860296a11a55ee14f0e5f5e8fc577/JupiterMag/TraceField.py#L16 for args
    return TraceField(x0, y0, z0, Verbose=False, IntModel='jrm33', ExtModel='Con2020', MaxStep=0.1)


def intersect_w_alphaeq_lon(T: TraceField, r_sc: TWO_D_NDArray, r_b: TWO_D_NDArray):
    # return positions, B fields, and pitch angles at intersections
    # r_sc is vector of normalized S/C pos vectors in SIII
    # r_b is vector of normalized boresight vectors in SIII
    r_mesh = np.stack((T.x, T.y, T.z), axis=-1).astype(np.float32)          # ntraces x 1000 pts per trace x 3
    max_trace = np.max(np.sum(~np.isnan(r_mesh), axis=1))                   # the max non nan entries in any column
    r_mesh = r_mesh[:, :max_trace, :]
    r_mesh_collapsed = np.reshape(r_mesh, (-1, 3))                          # num mesh pts x 3
    r_mesh_sc = r_mesh_collapsed[:, None, :] - r_sc                         # num mesh pts x num samples x 3

    # get points where ((x,y,z) - r_sc) dot r_b is maximized
    r_mesh_sc_norm = r_mesh_sc / np.linalg.norm(r_mesh_sc, axis=-1, keepdims=True)
    los_mask = np.nanargmax(np.einsum('ijk,jk->ij', r_mesh_sc_norm, r_b), axis=0)   # num samples -- take max over mesh pts

    # need B field to take dot product to get alpha
    # need equatorial B field to get alpha_eq using B, Beq, alpha
    # need longitude
    B_mesh = np.stack((T.Bx, T.By, T.Bz), axis=-1).astype(np.float32)
    B_mesh = B_mesh[:, :max_trace, :]
    B_mesh_collapsed = np.reshape(B_mesh, (-1, 3))                          # num mesh pts x 3
    B_data = B_mesh_collapsed[los_mask, :]                                  # num samples x 3
    b_data = B_data / np.linalg.norm(B_data, axis=-1, keepdims=True)
    alpha_data = np.acos(np.einsum('ij,ij->i', b_data, r_b))                # num samples
    
    Xeq, Yeq, Zeq = T.equator.x3, T.equator.y3, T.equator.z3                # ntraces
    Bx_eq, By_eq, Bz_eq = B(Xeq, Yeq, Zeq)
    B_eq_vec = np.stack((Bx_eq, By_eq, Bz_eq), axis=-1).astype(np.float32)  # ntraces x 3
    # need to convert from collapsed indices in los_mask to ntraces
    trace_mask = los_mask // max_trace
    B_eq_data = B_eq_vec[trace_mask, :]                                     # num samples x 3
    # sin(alpha) / B^2 = sin(alpha_eq) / Beq^2
    alpha_eq_data = np.asin(np.sin(alpha_data) * np.einsum('ij,ij->i', B_data, B_data) / np.einsum('ij,ij->i', B_eq_data, B_eq_data))   # num samples

    lon_m = T.equator.mlone[trace_mask]                                     # num samples, lon in degrees!
    return alpha_eq_data, lon_m


def compile_data(pjs: list[int], dt: int, chs: np.ndarray, nside: int = 128) -> np.ndarray:
    # healpix has pixels ordered by index, so (lat, lon) -> (npix)
    # see https://lambda.gsfc.nasa.gov/toolbox/pixelcoords.html for nside -> npix
    # create numpy array that is (#chs, #pix, #pjs) so i can take median over pjs
    out = np.empty((len(chs), hp.nside2npix(nside), len(pjs)))
    out[:] = np.nan     # initialize as NaNs
    M3_T = pre_compute_mshell_traces(3)
    for i, pj in enumerate(pjs):
        try:
            IRDR_data_pj, GRDR_data_pj = load_PJ_data(pj, dt, chs, keep_cols_GRDR=COLS_GRDR_MAP)
        except (NoProductsError, FileDownloadError, DownloadShortCircuitError) as err:
            continue
        # grab relevant columns
        Jn_SIII = GRDR_data_pj[['S3RH_x_JcJn', 'S3RH_y_JcJn', 'S3RH_z_JcJn']].to_numpy() / RJ
        boresight_SIII_1 = GRDR_data_pj[['S3RH_x_B1', 'S3RH_y_B1', 'S3RH_z_B1']].to_numpy()  # normalized
        boresight_SIII_2 = GRDR_data_pj[['S3RH_x_B2', 'S3RH_y_B2', 'S3RH_z_B2']].to_numpy()  # normalized

        T_sc = TraceField(*Jn_SIII.T, IntModel='jrm33', ExtModel='Con2020')
        w_in_mshell_mask = T_sc.equator.mshell < 3 * 0.95

        find_intersections(M3_T, Jn_SIII, boresight_SIII_1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", required=True, type=float, help="Delta time around each perijove in minutes")
    parser.add_argument("--ch", required=False, type=str, default="1,2,3,4,5,6", help="List of channels separated by comma")
    parser.add_argument("--PJs", required=True, type=str, help="Perijove range (e.g. 1,2,5,6 or 1-7 or 1,3-6)")
    args = parser.parse_args()
    chs = np.array([int(ch) for ch in args.ch.split(',')])
    for ch in chs: assert ch in range(1, 7), "Valid channel numbers are 1-6"
    pjs = parse_PJs(args.PJs)
    for pj in pjs: assert pj in range(1, 78), "Valid perijoves numbers are 1-77"
    compile_data(pjs, args.dt, chs)


if __name__ == "__main__":
    main()
