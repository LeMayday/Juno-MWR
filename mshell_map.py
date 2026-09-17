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


def pre_compute_mshell_traces(M, ntraces=100) -> TraceField:
    jm.Con2020.Config(equation_type='analytic')
    phi = np.linspace(0, 2*np.pi, ntraces, endpoint=False)
    theta = find_lats_M(phi, M)
    x0 = np.cos(phi) * np.sin(theta)
    y0 = np.sin(phi) * np.sin(theta)
    z0 = np.cos(theta)
    # see https://github.com/mattkjames7/JupiterMag/blob/a3fc24f20e0860296a11a55ee14f0e5f5e8fc577/JupiterMag/TraceField.py#L16 for args
    return TraceField(x0, y0, z0, Verbose=False, IntModel='jrm33', ExtModel='Con2020', MaxStep=0.1)


def find_intersections(T: TraceField, r_sc: np.ndarray, r_b: np.ndarray):
    # return positions, B fields, and pitch angles at intersections
    # r_sc is vector of normalized S/C pos vectors in SIII
    # r_los is vector of normalized boresight vectors in SIII

    r_b = r_b.astype(np.float32)
    r_sc = r_sc.astype(np.float32)
    r_los = (r_b - r_sc)                                                # num samples x 3
    r_los = r_los / np.linalg.norm(r_los, axis=-1, keepdims=True)
    r_M = np.stack((T.x, T.y, T.z), axis=-1).astype(np.float32)         # ntraces x 1000 pts per trace x 3
    max_trace = np.max(np.sum(~np.isnan(r_M), axis=1))
    r_M = r_M[:, :max_trace, :]
    r_M_collapsed = np.reshape(r_M, (-1, 3))                            # num mesh pts x 3
    r_M_sc = r_M_collapsed[:, None, :] - r_sc                           # num mesh pts x num samples x 3
    parallel_comp = np.einsum('ijk,jk->ij', r_M_sc, r_los)
    perp_vecs = r_M_sc - parallel_comp[..., None] * r_los
    perp_dist_sq = np.einsum('ijk,ijk->ji', perp_vecs, perp_vecs)       # num samples x num mesh pts
    closest_mesh = np.nanargmin(perp_dist_sq, axis=-1)
    closest_mesh_vecs = r_M_collapsed[closest_mesh, :]                  # num samples x 3
    _, SIII_longitudes = lat_lonW(*closest_mesh_vecs.T)

    B_M = np.stack((T.Bx, T.By, T.Bz), axis=-1).astype(np.float32)
    B_M = B_M[:, :max_trace, :]
    B_M_collapsed = np.reshape(r_M, (-1, 3))                            # num mesh pts x 3
    B_vecs = B_M_collapsed[closest_mesh, :]                             # num samples x 3
    pitch_angles = np.acos(np.einsum('ij,ij->i', r_los, B_vecs))

    

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
