# modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from JupiterMag import TraceField

# local files
from plot_data import make_subplots
from PDS_helper import load_PJ_data, NoProductsError, FileDownloadError, DownloadShortCircuitError, BadPJError
from synchrotron_map import parse_PJs, stack_data, RJ
from coordinates import lat_lonW
from jmag_helper import B, pre_compute_mshell_traces, init_Con2020_config, trace_batch_mshell

# default
import argparse
import pickle
import concurrent.futures
import os

MAX_WORKERS = max(1, os.cpu_count() - 2)
TWO_D_NDArray = np.ndarray[tuple[int, int], np.dtype[np.float32]]
THREE_D_NDArray = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
COLS_GRDR = ['t_ephem_time', 't_utc_doy',
             'PC_lon_JsB1', 'PC_lon_JsB2',
             'S3RH_x_B1', 'S3RH_y_B1', 'S3RH_z_B1', 'S3RH_x_B2', 'S3RH_y_B2', 'S3RH_z_B2',
             'range_JnJc', 'S3RH_x_JcJn', 'S3RH_y_JcJn', 'S3RH_z_JcJn']


def batch_data(data: np.ndarray) -> list[np.ndarray]:
    size = data.shape[0]
    batch_size = max(500, int(np.ceil(size / MAX_WORKERS)))                 # ceiling division to prevent missing remainder, with 500 as smallest size
    return [data[i:i + batch_size] for i in range(0, size, batch_size)]


def intersect_w_alphaeq_lon(T: TraceField, r_sc: TWO_D_NDArray, r_b: TWO_D_NDArray, pj: int):
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
    # sin^2(alpha) / B = sin^2(alpha_eq) / Beq
    alpha_eq_data = np.asin(np.sin(alpha_data) * np.sqrt(np.linalg.norm(B_eq_data, axis=-1) / np.linalg.norm(B_data, axis=-1)))   # num samples

    lon_m = np.rad2deg(T.equator.mlone[trace_mask]) + 180                   # num samples, lon in degrees!
    plot_points(r_mesh_collapsed, r_sc, los_mask)
    return alpha_eq_data, lon_m


def plot_points(r_mesh: TWO_D_NDArray, r_sc: TWO_D_NDArray, los_mask: np.ndarray, pj: int):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # plot Jupiter
    RJ_polar_ratio = 66,854 / RJ
    th = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    x = np.outer(np.cos(phi), np.sin(th))
    y = np.outer(np.sin(phi), np.sin(th))
    z = np.outer(np.ones(np.size(phi)), np.cos(th)) / RJ_polar_ratio**2
    ax1.plot_surface(x, y, z, edgecolor='None')
    ax2.plot_surface(x, y, z, edgecolor='None')

    # plot viewed points
    ax1.scatter(*(r_mesh[los_mask, :].T), color='red', marker='.', s=50)
    # plot M-shell mesh
    ax2.scatter(*r_mesh.T, color='blue', marker='.')

    # plot Juno trajectory
    ax1.plot(*r_sc.T, color='black')
    ax2.plot(*r_sc.T, color='black')

    with open("pickle/interactive_plot_pj{pj}.pickle", "wb") as f:
        pickle.dump(fig, f)


def compile_data(pjs: list[int], dt: int, chs: np.ndarray, M: float, ntraces: int) -> np.ndarray:
    # create numpy array that is (#chs, #alpha, #lon, #pjs) so i can take median over pjs
    out = np.empty((len(chs), ntraces, 90, len(pjs)))
    out[:] = np.nan     # initialize as NaNs
    M_trace = pre_compute_mshell_traces(M, ntraces)
    max_lat = np.min([M_trace.ionosphere.latn, M_trace.surface.latn])
    min_lat = np.max([M_trace.ionosphere.lats, M_trace.surface.lats])
    skipped_PJs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_Con2020_config) as executor:
        for pj in pjs:
            print(f"Loading PJ {pj}")
            try:
                IRDR_data_pj, GRDR_data_pj = load_PJ_data(pj, dt, chs, keep_cols_GRDR=COLS_GRDR)
            except (NoProductsError, FileDownloadError, DownloadShortCircuitError, BadPJError) as err:
                print(f"Skipping PJ: {pj}")
                skipped_PJs.append(pj)
                continue
            # grab relevant columns
            Jn_SIII = GRDR_data_pj[['S3RH_x_JcJn', 'S3RH_y_JcJn', 'S3RH_z_JcJn']].to_numpy(dtype=np.float32) / RJ   # normalized to Jupiter radius
            Jn_SIII_norm = Jn_SIII / np.linalg.norm(Jn_SIII, axis=-1, keepdims=True)
            lat, _ = lat_lonW(*Jn_SIII_norm.T)
            lat_mask = np.logical_and(lat > min_lat, lat < max_lat)
            boresight_SIII_1 = GRDR_data_pj[['S3RH_x_B1', 'S3RH_y_B1', 'S3RH_z_B1']].to_numpy(dtype=np.float32)     # normalized
            boresight_SIII_2 = GRDR_data_pj[['S3RH_x_B2', 'S3RH_y_B2', 'S3RH_z_B2']].to_numpy(dtype=np.float32)     # normalized

            print("Filtering")
            # filter out views of Jupiter --- 12 deg/s, so ~1-2 sec for whole beam width to be off Jupiter -> 10-20 extra samples
            mshell_batches = list(executor.map(trace_batch_mshell, batch_data(Jn_SIII)))
            Jn_mshell = np.concatenate(mshell_batches)
            in_mshell_mask = np.logical_and(Jn_mshell > 1.01, Jn_mshell < M * 0.95)
            pos_mask = np.logical_and(lat_mask, in_mshell_mask)

            n_extra = 15                                                                                        # 15 extra samples
            jupiter_mask_ch1 = ~np.isnan(GRDR_data_pj["PC_lon_JsB1"].to_numpy())                                # masks for where antenna beam is looking at Jupiter
            jupiter_mask_ch1 = np.convolve(jupiter_mask_ch1, np.ones(2*n_extra + 1).astype(bool), 'same')       # expand mask to include beamwidth
            mask1 = np.logical_and(~jupiter_mask_ch1, pos_mask)
            Jn_SIII_ch1 = Jn_SIII[mask1, :]                                                                     # mask positions
            boresight_SIII_1 = boresight_SIII_1[mask1, :]                                                       # mask boresights

            jupiter_mask_ch2 = ~np.isnan(GRDR_data_pj["PC_lon_JsB2"].to_numpy())
            jupiter_mask_ch2 = np.convolve(jupiter_mask_ch2, np.ones(2*n_extra + 1).astype(bool), 'same')
            mask2 = np.logical_and(~jupiter_mask_ch2, pos_mask)
            Jn_SIII_ch2 = Jn_SIII[mask2, :]                                                                     # mask positions
            boresight_SIII_2 = boresight_SIII_2[mask2, :]                                                       # mask boresights

            print("Calculating intersections")
            if 1 in chs:
                alphas1, lons1 = intersect_w_alphaeq_lon(M_trace, Jn_SIII_ch1, boresight_SIII_1, pj)
            if (len(chs) == 1 and chs[0] != 1) or len(chs) > 1:
                alphas2, lons2 = intersect_w_alphaeq_lon(M_trace, Jn_SIII_ch2, boresight_SIII_2, pj)
            print("Binning")
            for j, ch in enumerate(chs):
                T_a = IRDR_data_pj[f"Ch{ch}"]   # antenna temperature
                if ch == 1:
                    alphas = alphas1; lons = lons1
                    T_a = T_a[mask1]
                else:
                    alphas = alphas2; lons = lons2
                    T_a = T_a[mask2]
                assert T_a.shape == alphas.shape == lons.shape, "T_a, alphas, and lons must have same shape!"

                binned_medians = bin_data(T_a, lons, np.rad2deg(alphas), ntraces)
                out[j, :, :, pj-1] = binned_medians     # everything else should still be NaN
    print(f"Data compilation finished. Skipped PJs: {skipped_PJs}")
    return out


def bin_data(T_a: np.ndarray, lons: np.ndarray, alphas: np.ndarray, ntraces: int) -> np.ndarray:
    # longitudes in degrees!
    lon_bins = np.linspace(0, 360, ntraces+1, endpoint=True)
    alpha_bins = np.linspace(0, 90, 91, endpoint=True)
    med, _, _, _ = binned_statistic_2d(x=lons, y=alphas, values=T_a, statistic="median", bins=[lon_bins, alpha_bins])
    return med


def plot_data(data: np.ndarray, chs: list, params_str: str):
    fig = plt.figure(figsize=(18,8))
    axes = make_subplots(fig, data.shape[0])
    for i, ax in enumerate(axes):
        im = ax.imshow(data[i, :].T, origin='lower', cmap='gist_ncar', aspect='auto', extent=[0, 360, 0, 90])
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Ch{chs[i]}")
        ax.set_xlabel("JRM33 Dipole Longitude")
        ax.set_ylabel("$\\alpha_{eq}$")
    fig.tight_layout()
    fig.savefig(f"MWR_alpha_longitude_distribution_{params_str}.png", dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", required=True, type=float, help="Delta time around each perijove in minutes")
    parser.add_argument("--ch", required=False, type=str, default="1,2,3,4,5,6", help="List of channels separated by comma")
    parser.add_argument("--PJs", required=True, type=str, help="Perijove range (e.g. 1,2,5,6 or 1-7 or 1,3-6)")
    parser.add_argument("--M", required=True, type=float, help="M-shell")
    args = parser.parse_args()
    chs = np.array([int(ch) for ch in args.ch.split(',')])
    for ch in chs: assert ch in range(1, 7), "Valid channel numbers are 1-6"
    pjs = parse_PJs(args.PJs)
    for pj in pjs: assert pj in range(1, 78), "Valid perijoves numbers are 1-77"

    time_series_data = compile_data(pjs, args.dt, chs, M=args.M, ntraces=100)
    stacked_data = stack_data(time_series_data)
    plot_data(stacked_data, chs, f"PJs{args.PJs}_CHs{args.ch}_M{args.M}_dt{args.dt}")


if __name__ == "__main__":
    init_Con2020_config()
    main()
