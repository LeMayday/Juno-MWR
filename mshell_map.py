# modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from JupiterMag import TraceField
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage import map_coordinates

# local files
from plot_data import make_subplots
from PDS_helper import load_PJ_data, NoProductsError, FileDownloadError, DownloadShortCircuitError, BadPJError
from synchrotron_map import parse_PJs, stack_data, RJ
from coordinates import lat_lonW
from jmag_helper import B, pre_compute_mshell_traces, init_jm_config, trace_batch_mshell

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


class UniformGridLinearInterpolator:
    # see https://docs.scipy.org/doc/scipy-1.14.1/tutorial/interpolate/ND_regular_grid.html
    def __init__(self, points, values):
        self.limits = np.array([[min(x), max(x)] for x in points])
        self.values = np.asarray(values, dtype=float)   # note this could be (..., 3) for a vector-valued function

    def __call__(self, xi):
        """
        `xi` here is an array-like (an array or a list) of points.

        Each "point" is an ndim-dimensional array_like, representing
        the coordinates of a point in ndim-dimensional space.
        """
        # transpose the xi array into the ``map_coordinates`` convention
        # which takes coordinates of a point along columns of a 2D array.
        xi = np.moveaxis(np.asarray(xi), -1, 0)

        if len(self.values.shape) == len(self.limits):  # if scalar function, add dummy axis
            self.values = np.expand_dims(self.values, -1)
        ns = self.values.shape[:-1]                     # dimensions of function domain
        n_values = self.values.shape[-1]                # dimensions of function value (1 if scalar, >1 if vector)
        out = np.zeros((*coords.shape[1:], n_values))   # output will be coords.shape[1:] for each value of the function
        for i in range(n_values):                       # loop through function scalar/vector components (for each dim...)
            # convert from data coordinates to pixel coordinates
            coords = [(n-1)*(val - lo) / (hi - lo) for val, n, (lo, hi) in zip(xi, ns, self.limits)]
            # interpolate
            out[:, i] = map_coordinates(self.values, coords, order=1, mode='constant', cval=np.nan)
        if out.shape[-1] == 1:   # scalar function
            return out[:, 0]
        return out

UGLI = UniformGridLinearInterpolator  # type alias


def batch_data(data: np.ndarray) -> list[np.ndarray]:
    size = data.shape[0]
    batch_size = max(500, int(np.ceil(size / MAX_WORKERS)))                 # ceiling division to prevent missing remainder, with 500 as smallest size
    return [data[i:i + batch_size] for i in range(0, size, batch_size)]


def intersect_alphas_lons(r_sc: TWO_D_NDArray, r_b: TWO_D_NDArray, s: np.ndarray,
                          UGLI_M: UGLI, UGLI_B: UGLI, UGLI_Beq: UGLI, UGLI_lon: UGLI,
                          M: float, dM: float):
    # r_sc is vector of normalized S/C pos vectors in SIII
    # r_b is vector of normalized boresight vectors in SIII
    # s_los is distance along line of sight
    s_los = r_b[:, None, :] * s[None, :, None]          # num samples x num LOS pts x 3
    r_los = r_sc[:, None, :] + s_los                    # add spacecraft positions to LOS points
    M_los = UGLI_M(r_los)                               # num samples x num LOS pts
    mask_M = np.logical_and(M_los > M - dM, M_los < M + dM)     # where the LOS is looking at the desired M-shell
    r_los_M = r_los[mask_M, :]                          # valid mask positions x 3
    B_los = UGLI_B(r_los)                               # num samples x num LOS pts x 3
    b_los = B_los / np.linalg.norm(B_los, axis=-1, keepdims=True)
    alpha_los = np.acos(np.einsum('ijk,ik->ij', b_los, r_b))    # num samples x num LOS pts
    Beq_los = UGLI_Beq(r_los)
    # sin^2(alpha) / B = sin^2(alpha_eq) / Beq => sin(alpha_eq) = sin(alpha) * sqrt(Beq / B)
    alpha_eq_los = np.asin(np.sin(alpha_los) * np.sqrt(np.linalg.norm(Beq_los, axis=-1) / np.linalg.norm(B_los, axis=-1)))
    lon_los = UGLI_lon(r_los)

    # filter regions outside Mshell
    alpha_los[~mask_M] = np.nan
    alpha_eq_los[~mask_M] = np.nan
    lon_los[~mask_M] = np.nan
    # average over valid M-shell region in row (sample)
    alpha_data = np.nanmean(alpha_los, axis=-1)
    alpha_eq_data = np.nanmean(alpha_eq_los, axis=-1)
    lon_data = np.nanmean(lon_los, axis=-1)
    Mshell_intersect_pts = r_los[mask_M, :]             # physical points where Mshell was intersected (for plotting)
    return np.rad2deg(alpha_data), np.rad2deg(alpha_eq_data), lon_data, Mshell_intersect_pts    # return everything in degrees


def intersect_w_alphaeq_lon(T: TraceField, r_sc: TWO_D_NDArray, r_b: TWO_D_NDArray, M: float, pj: int):
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
    B_eq_vec = B(Xeq, Yeq, Zeq)                                             # ntraces x 3
    # need to convert from collapsed indices in los_mask to ntraces
    trace_mask = los_mask // max_trace
    B_eq_data = B_eq_vec[trace_mask, :]                                     # num samples x 3
    # sin^2(alpha) / B = sin^2(alpha_eq) / Beq
    alpha_eq_data = np.asin(np.sin(alpha_data) * np.sqrt(np.linalg.norm(B_eq_data, axis=-1) / np.linalg.norm(B_data, axis=-1)))   # num samples

    lon_m = np.rad2deg(T.equator.mlone[trace_mask]) + 180                   # num samples, lon in degrees!
    plot_points(r_mesh_collapsed, r_sc, los_mask, M, pj)
    return alpha_eq_data, lon_m


def plot_points(r_mesh: TWO_D_NDArray, r_sc: TWO_D_NDArray, los_mask: np.ndarray, M: float, pj: int):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # plot Jupiter
    RJ_polar_ratio = 66854 / RJ
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

    with open(f"pickle/interactive_plot_M{M}_pj{pj}.pickle", "wb") as f:
        pickle.dump(fig, f)


def compile_data(pjs: list[int], dt: int, chs: np.ndarray, M: float, ntraces: int) -> np.ndarray:
    # create numpy array that is (#chs, #alpha, #lon, #pjs) so i can take median over pjs
    out = np.empty((len(chs), ntraces, 90, len(pjs)))
    out[:] = np.nan     # initialize as NaNs
    M_trace = pre_compute_mshell_traces(M, ntraces)
    max_lat = 80#np.min([M_trace.ionosphere.latn, M_trace.surface.latn])
    min_lat = -80#np.max([M_trace.ionosphere.lats, M_trace.surface.lats])
    skipped_PJs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_jm_config) as executor:
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
            in_mshell_mask = np.logical_and(Jn_mshell > 1.01, Jn_mshell < M * 0.98)
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
                alphas1, lons1 = intersect_w_alphaeq_lon(M_trace, Jn_SIII_ch1, boresight_SIII_1, M, pj)
            if (len(chs) == 1 and chs[0] != 1) or len(chs) > 1:
                alphas2, lons2 = intersect_w_alphaeq_lon(M_trace, Jn_SIII_ch2, boresight_SIII_2, M, pj)
            print("Binning")
            for ch in chs:
                T_a = IRDR_data_pj[f"Ch{ch}"]   # antenna temperature
                if ch == 1:
                    alphas = alphas1; lons = lons1
                    T_a = T_a[mask1]
                else:
                    alphas = alphas2; lons = lons2
                    T_a = T_a[mask2]
                assert T_a.shape == alphas.shape == lons.shape, "T_a, alphas, and lons must have same shape!"

                binned_medians = bin_data(T_a, lons, np.rad2deg(alphas), ntraces)
                out[ch-1, :, :, pj-1] = binned_medians      # everything else should still be NaN
    print(f"Data compilation finished. Skipped PJs: {skipped_PJs}")
    return out

def mask_M(Jn_SIII: TWO_D_NDArray, UGLI_M: UGLI, M: float) -> np.ndarray:
    # inputs are Juno positions in SIII / RJ, an interpolator for Mshell on the mesh, and the desired M shell
    # Jn_SIII is # num samples x 3
    Jn_Ms = UGLI_M(Jn_SIII)
    return np.logical_and(Jn_Ms > 1.01, Jn_Ms < M * 0.98)


def mask_jupiter(jupiter_intersections: np.ndarray) -> np.ndarray:
    # filter out views of Jupiter --- 12 deg/s, so ~1-2 sec for whole beam width to be off Jupiter -> 10-20 extra samples
    n_extra = 15                                                                            # 15 extra samples
    jupiter_mask = ~np.isnan(jupiter_intersections)                                         # masks for where antenna beam is looking at Jupiter
    jupiter_mask = np.convolve(jupiter_mask, np.ones(2*n_extra + 1).astype(bool), 'same')   # expand mask to include beamwidth
    return jupiter_mask


def bin_data(T_a: np.ndarray, lons: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    # longitudes in degrees!
    lon_bins = np.linspace(0, 360, 181, endpoint=True)
    alpha_bins = np.linspace(0, 90, 91, endpoint=True)
    med, _, _, _ = binned_statistic_2d(x=lons, y=alphas, values=T_a, statistic="median", bins=[lon_bins, alpha_bins])
    return med


def plot_data(data: np.ndarray, chs: list, params_str: str, vmax: list = None):
    fig = plt.figure(figsize=(18,8))
    axes = make_subplots(fig, data.shape[0])
    for i, ax in enumerate(axes):
        if vmax is not None:
            im = ax.imshow(data[i, :].T, origin='lower', cmap='gist_ncar', vmin=0, vmax=vmax[i], aspect='auto', extent=[0, 360, 0, 90])
        else:
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
    parser.add_argument("--M", required=True, type=str, help="M-shell")
    parser.add_argument('-f', "--mesh-file", required=True, type=str, help="npz file containing mesh information")
    args = parser.parse_args()

    # input validation
    chs = np.array([int(ch) for ch in args.ch.split(',')])
    for ch in chs: assert ch in range(1, 7), "Valid channel numbers are 1-6"
    pjs = parse_PJs(args.PJs)
    for pj in pjs: assert pj in range(1, 78), "Valid perijoves numbers are 1-77"
    Ms = np.array([float(M) for M in args.M.split(',')])

    # construct RegularGridInterpolators for each mesh value
    mesh_data = np.load(args.mesh_file)
    points = mesh_data["points"]
    UGLI_B =   UGLI(points=points, values=mesh_data["B_mesh"])
    UGLI_Beq = UGLI(points=points, values=mesh_data["B_mesh_eq"])
    UGLI_M =   UGLI(points=points, values=mesh_data["Mshell"])
    UGLI_lon = UGLI(points=points, values=mesh_data["dipole_lon"])

    for M in Ms:
        # create numpy array that is (#chs, #alpha, #lon, #pjs) so i can take median over pjs
        series_Ta = np.full((len(chs), 90, 180, len(pjs)), np.nan)      # longitudes as cols for plotting; initialize as NaNs
        series_alpha = np.full_like(series_Ta, np.nan)
        skipped_PJs = []
        for pj in pjs:
            print(f"Loading PJ {pj}")
            try:
                IRDR_data_pj, GRDR_data_pj = load_PJ_data(pj, args.dt, chs, keep_cols_GRDR=COLS_GRDR)
            except (NoProductsError, FileDownloadError, DownloadShortCircuitError, BadPJError) as err:
                print(f"Skipping PJ: {pj}")
                skipped_PJs.append(pj)
                continue
            # grab relevant columns
            Jn_SIII = GRDR_data_pj[['S3RH_x_JcJn', 'S3RH_y_JcJn', 'S3RH_z_JcJn']].to_numpy(dtype=np.float32) / RJ   # normalized to Jupiter radius
            boresight_SIII_1 = GRDR_data_pj[['S3RH_x_B1', 'S3RH_y_B1', 'S3RH_z_B1']].to_numpy(dtype=np.float32)     # normalized
            boresight_SIII_2 = GRDR_data_pj[['S3RH_x_B2', 'S3RH_y_B2', 'S3RH_z_B2']].to_numpy(dtype=np.float32)     # normalized

            print("Filtering")
            pos_mask = mask_M(Jn_SIII, UGLI_M, M, 0.05)
            jupiter_mask_ch1 = mask_jupiter(GRDR_data_pj["PC_lon_JsB1"].to_numpy())
            jupiter_mask_ch2 = mask_jupiter(GRDR_data_pj["PC_lon_JsB2"].to_numpy())
            mask1 = np.logical_and(~jupiter_mask_ch1, pos_mask)
            mask2 = np.logical_and(~jupiter_mask_ch2, pos_mask)
            Jn_SIII_ch1 = Jn_SIII[mask1, :]                         # mask positions
            boresight_SIII_1 = boresight_SIII_1[mask1, :]           # mask boresights
            Jn_SIII_ch2 = Jn_SIII[mask2, :]                         # mask positions
            boresight_SIII_2 = boresight_SIII_2[mask2, :]           # mask boresights

            print("Calculating intersections")
            s = np.linspace(0, 7, 100)                              # LOS distance
            if 1 in chs:
                alphas1, alphas_eq1, lons1, pos1 = intersect_alphas_lons(r_sc=Jn_SIII_ch1, r_b=boresight_SIII_1, s=s,
                                                                         UGLI_M=UGLI_M, UGLI_B=UGLI_B, UGLI_Beq=UGLI_Beq, UGLI_lon=UGLI_lon,
                                                                         M=M, dM=0.05)
            if (len(chs) == 1 and chs[0] != 1) or len(chs) > 1:
                alphas2, alphas_eq2, lons2, pos2 = intersect_alphas_lons(r_sc=Jn_SIII_ch2, r_b=boresight_SIII_2, s=s,
                                                                         UGLI_M=UGLI_M, UGLI_B=UGLI_B, UGLI_Beq=UGLI_Beq, UGLI_lon=UGLI_lon,
                                                                         M=M, dM=0.05)
            print("Binning")
            for ch in chs:
                Ta = IRDR_data_pj[f"Ch{ch}"]    # antenna temperature
                if ch == 1:
                    alphas, alphas_eq, lons, pos = alphas1, alphas_eq1, lons1, pos1
                    Ta = Ta[mask1]
                else:   # I do filtering outside of this loop so I don't redo work for each ch > 1
                    alphas, alphas_eq, lons, pos = alphas2, alphas_eq2, lons2, pos2
                    Ta = Ta[mask2]
                assert Ta.shape == alphas_eq.shape == lons.shape, "Ta, alphas, and lons must have same shape!"

                binned_Ta_medians = bin_data(Ta, lons, alphas_eq)   # this should drop values if lons or alphas_eq are nan
                series_Ta[ch-1, :, :, pj-1] = binned_Ta_medians.T   # binned_statistic maps x to rows
                binned_alpha_medians = bin_data(alphas, lons, alphas_eq)    # track which measurement comes from which local pitch angle
                series_alpha[ch-1, :, :, pj-1] = binned_alpha_medians.T
        print(f"Data compilation finished. Skipped PJs: {skipped_PJs}")

        params_str = f"PJs{args.PJs}_CHs{args.ch}_M{M}_dt{args.dt}"
        np.save(f"MWR_Ta_series_{params_str}.npy", series_Ta)
        np.save(f"MWR_alpha_series_{params_str}.npy", series_alpha)

        stacked_Ta_data = stack_data(series_Ta)
        stacked_alpha_data = stack_data(series_alpha)

        np.save(f"MWR_Ta_stacked_{params_str}.npy", stacked_Ta_data)
        np.save(f"MWR_alpha_stacked_{params_str}.npy", stacked_alpha_data)
        plot_data(stacked_Ta_data, chs, params_str)
        plot_data(stacked_alpha_data, chs, params_str)


if __name__ == "__main__":
    init_jm_config()
    main()
