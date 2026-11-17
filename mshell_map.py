# modules
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.ndimage import map_coordinates
from pandas import DataFrame

# local files
from banana_slug import make_subplots
from PDS_helper import load_PJ_data, get_Juno_pos_RJ, get_boresight_dir, PDSError
from synchrotron_map import parse_PJs, stack_data
from coordinates import lat_lonE
from global_vars import ONE_D_NDArray, TWO_D_NDArray, THREE_D_NDArray, VMAX

# default
import argparse
from pathlib import Path
from typing import Literal

COLS_GRDR = ['t_ephem_time', 't_utc_doy',
             'PC_lon_JsB1', 'PC_lon_JsB2',
             'S3RH_x_B1', 'S3RH_y_B1', 'S3RH_z_B1', 'S3RH_x_B2', 'S3RH_y_B2', 'S3RH_z_B2',
             'range_JnJc', 'S3RH_x_JcJn', 'S3RH_y_JcJn', 'S3RH_z_JcJn']
IM_PATH = "images/mshell_map"


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
    return np.asin(np.sin(alpha) * np.sqrt(Beq / B))


def make_Rlos(r_sc: np.ndarray, r_b: np.ndarray, smin: float, smax: float, sN: int):
    # r_sc is vector of normalized S/C pos vectors in SIII
    s = np.linspace(smin, smax, sN + 1)                 # LOS distance
    s_los = r_b[:, None, :] * s[None, :, None]          # num samples x num LOS pts x 3
    return r_sc[:, None, :] + s_los                     # add spacecraft positions to LOS points


def intersect_alphas_lons(r_los: THREE_D_NDArray, r_b: TWO_D_NDArray, MI: MeshInterpolator, M: float, dM: float) -> tuple[ONE_D_NDArray, ONE_D_NDArray, ONE_D_NDArray, TWO_D_NDArray]:
    # r_los is position at each sample time and at each point along LOS (num samples x num LOS pts x 3)
    # r_b is vector of normalized boresight vectors in SIII
    M_los = MI.M(r_los)                                         # num samples x num LOS pts
    mask_M = np.logical_and(M_los > M - dM, M_los < M + dM)     # where the LOS is looking at the desired M-shell
    B_los = MI.B(r_los)                                         # num samples x num LOS pts x 3
    alpha_los = alpha(r_b, B_los)

    Beq_los = MI.Beq(r_los)
    alpha_eq_los = alpha_eq(alpha_los, norm(Beq_los, axis=-1), norm(B_los, axis=-1))

    lon_los = MI.lonxy(r_los)

    # filter regions outside Mshell
    alpha_los[~mask_M] = np.nan
    alpha_eq_los[~mask_M] = np.nan
    lon_los[~mask_M] = np.nan
    # average over valid M-shell region in row (sample)
    alpha_data = np.nanmean(alpha_los, axis=-1)
    alpha_eq_data = np.nanmean(alpha_eq_los, axis=-1)
    lon_data = np.nanmean(lon_los, axis=-1)
    return np.rad2deg(alpha_data), np.rad2deg(alpha_eq_data), lon_data, mask_M    # return everything in degrees


def mask_M(Jn_SIII: TWO_D_NDArray, UGLI_M: UGLI, M: float) -> ONE_D_NDArray:
    # inputs are Juno positions in SIII / RJ, an interpolator for Mshell on the mesh, and the desired M shell
    # Jn_SIII is # num samples x 3
    Jn_Ms = UGLI_M(Jn_SIII)
    return np.logical_and(Jn_Ms > 1.01, Jn_Ms < M * 0.98)


def mask_jupiter(jupiter_intersections: ONE_D_NDArray) -> ONE_D_NDArray:
    # filter out views of Jupiter --- 12 deg/s, so ~1-2 sec for whole beam width to be off Jupiter -> 10-20 extra samples
    n_extra = 15                                                                            # 15 extra samples
    jupiter_mask = ~np.isnan(jupiter_intersections)                                         # masks for where antenna beam is looking at Jupiter
    jupiter_mask = np.convolve(jupiter_mask, np.ones(2*n_extra + 1).astype(bool), 'same')   # expand mask to include beamwidth
    return jupiter_mask


def mask_lat(Jn_SIII: TWO_D_NDArray) -> ONE_D_NDArray:
    Jn_SIII_norm = Jn_SIII / norm(Jn_SIII, axis=-1, keepdims=True)
    lat, _ = lat_lonE(*Jn_SIII_norm.T)
    lat_mask = np.logical_and(lat > -20, lat < 20)
    return lat_mask


def bin_data(values: ONE_D_NDArray, lons: ONE_D_NDArray, alphas: ONE_D_NDArray, num_lons: int = 180, num_alphas: int = 90) -> TWO_D_NDArray:
    # longitudes in degrees!
    lon_bins = np.linspace(0, 360, num_lons + 1, endpoint=True)
    alpha_bins = np.linspace(0, 90, num_alphas + 1, endpoint=True)
    bad_input = np.logical_or(np.isnan(lons), np.isnan(alphas))
    values = values[~bad_input]
    lons = lons[~bad_input]
    alphas = alphas[~bad_input]
    med, _, _, _ = binned_statistic_2d(x=lons, y=alphas, values=values, statistic="median", bins=[lon_bins, alpha_bins])
    return med


def plot_data(data: THREE_D_NDArray, chs: list, params_str: str, vmax: list = None, BLC_file: str = None, save_path: str = IM_PATH):
    fig = plt.figure(figsize=(18,8))
    axes = make_subplots(fig, data.shape[0])
    for i, ax in enumerate(axes):
        if vmax is not None:
            im = ax.imshow(data[i, :], origin='lower', cmap='gist_ncar', vmin=0, vmax=vmax[i], aspect='auto', extent=[0, 360, 0, 90])
        else:
            im = ax.imshow(data[i, :], origin='lower', cmap='gist_ncar', aspect='auto', extent=[0, 360, 0, 90])
        fig.colorbar(im, ax=ax)
        if BLC_file is not None:
            BLC_data = np.loadtxt(BLC_file)
            lons = BLC_data[:, 0]
            BLC_N = BLC_data[:, 1]
            BLC_S = BLC_data[:, 2]
            ax.plot(lons, BLC_N, linestyle='-')
            ax.plot(lons, BLC_S, linestyle='--')
            ax.axhline(np.max(BLC_data[:, 1:]), linestyle='dotted')
        ax.set_title(f"Ch{chs[i]}")
        ax.set_xlabel("JRM33 Dipole Longitude")
        ax.set_ylabel("$\\alpha_{eq}$")
    fig.tight_layout()
    fig.savefig(f"{save_path}/MWR_{params_str}.png", dpi=300)
    plt.close(fig)


def geometry_mask(GRDR_data_pj: DataFrame, UGLI_M: UGLI, M: float, CH: Literal[1, 2]) -> ONE_D_NDArray:
    Jn_SIII = get_Juno_pos_RJ(GRDR_data_pj)
    lat_mask = mask_lat(Jn_SIII)                    # mask for near the equator
    pos_mask = mask_M(Jn_SIII, UGLI_M, M)           # mask for within mshell
    pos_mask = np.logical_and(pos_mask, lat_mask)
    jupiter_mask = mask_jupiter(GRDR_data_pj[f"PC_lon_JsB{CH}"].to_numpy())     # mask where looking at jupiter
    return np.logical_and(~jupiter_mask, pos_mask)


def geometry_intersections(GRDR_data_pj: DataFrame, mask: ONE_D_NDArray, M: float, CH: Literal[1, 2],
                           MI: MeshInterpolator, save_params: str = None) -> tuple[ONE_D_NDArray, ONE_D_NDArray, ONE_D_NDArray]:
    # grab relevant columns
    Jn_SIII = get_Juno_pos_RJ(GRDR_data_pj)
    boresight_SIII = get_boresight_dir(GRDR_data_pj=GRDR_data_pj, CH=CH)
    Jn_SIII = Jn_SIII[mask, :]                      # mask positions
    boresight_SIII = boresight_SIII[mask, :]        # mask boresights

    r_los = make_Rlos(r_sc=Jn_SIII, r_b=boresight_SIII, smin=0.2, smax=7, sN=100)
    alphas, alphas_eq, lons, r_los_mask = intersect_alphas_lons(r_los=r_los, r_b=boresight_SIII, MI=MI, M=M, dM=0.05)
    if save_params is not None:
        np.savez(f"npz/MWR_plottables_{save_params}.npz", r_sc=Jn_SIII, r_b=boresight_SIII, r_los=r_los, r_los_mask=r_los_mask)
    return alphas, alphas_eq, lons


def map_Ta_to_M(IRDR_data_pj: DataFrame, GRDR_data_pj: DataFrame, M: float, pj: int,
                MI: MeshInterpolator, num_lons: int = 180, num_alphas: int = 90) -> tuple[THREE_D_NDArray, THREE_D_NDArray]:
    print("Calculating intersections")
    mask1 = geometry_mask(GRDR_data_pj=GRDR_data_pj, UGLI_M=MI.M, M=M, CH=1)
    mask2 = geometry_mask(GRDR_data_pj=GRDR_data_pj, UGLI_M=MI.M, M=M, CH=2)
    alphas1, alphas_eq1, lons1 = geometry_intersections(GRDR_data_pj=GRDR_data_pj, mask=mask1, M=M, CH=1, MI=MI, save_params=f"M{M}_PJ{pj}_CH{1}")
    alphas2, alphas_eq2, lons2 = geometry_intersections(GRDR_data_pj=GRDR_data_pj, mask=mask2, M=M, CH=2, MI=MI, save_params=f"M{M}_PJ{pj}_CH{1}")
    out_Ta = np.full((6, num_alphas, num_lons), np.nan)
    out_alpha = np.full_like(out_Ta, np.nan)
    for ch in range(1, 7):
        Ta = IRDR_data_pj[f"Ch{ch}"]    # antenna temperature
        if ch == 1:
            alphas, alphas_eq, lons = alphas1, alphas_eq1, lons1
            Ta = Ta[mask1]
        else:   # I do filtering outside of this loop so I don't redo work for each ch > 1
            alphas, alphas_eq, lons = alphas2, alphas_eq2, lons2
            Ta = Ta[mask2]
        assert Ta.shape == alphas_eq.shape == lons.shape, "Ta, alphas, and lons must have same shape!"

        binned_Ta_medians = bin_data(Ta, lons, alphas_eq, num_lons, num_alphas)             # this should drop values if lons or alphas_eq are nan
        binned_alpha_medians = bin_data(alphas, lons, alphas_eq, num_lons, num_alphas)      # track which measurement comes from which local pitch angle
        out_Ta[ch-1, :, :] = binned_Ta_medians.T                                            # binned_statistic maps x to rows, need x as cols for imshow
        out_alpha[ch-1 :, :] = binned_alpha_medians.T
    return out_Ta, out_alpha


def mshell_argparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dt", required=True, type=float, help="Delta time around each perijove in minutes")
    parser.add_argument("--ch", required=False, type=str, default="1,2,3,4,5,6", help="List of channels separated by comma")
    parser.add_argument("--PJs", required=True, type=str, help="Perijove range (e.g. 1,2,5,6 or 1-7 or 1,3-6)")
    parser.add_argument("--M", required=True, type=str, help="M-shell")
    parser.add_argument('-f', "--mesh-file", required=True, type=str, help="npz file containing mesh information")
    return parser


def main():
    parser = argparse.ArgumentParser()
    args = mshell_argparser(parser).parse_args()

    # input validation
    chs = np.array([int(ch) for ch in args.ch.split(',')])
    for ch in chs: assert ch in range(1, 7), "Valid channel numbers are 1-6"
    pjs = parse_PJs(args.PJs)
    for pj in pjs: assert pj in range(1, 78), "Valid perijoves numbers are 1-77"
    Ms = np.array([float(M) for M in args.M.split(',')])

    MI = MeshInterpolator(args.mesh_file)

    for M in Ms:
        save_path = f"{IM_PATH}/{M}"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        # create numpy array that is (#chs, #alpha, #lon, #pjs) so i can take median over pjs
        series_Ta = np.full((len(chs), 90, 180, len(pjs)), np.nan)      # longitudes as cols for plotting; initialize as NaNs
        series_alpha = np.full_like(series_Ta, np.nan)
        skipped_PJs = []
        for i, pj in enumerate(pjs):
            print(f"Loading PJ {pj}")
            try:
                IRDR_data_pj, GRDR_data_pj = load_PJ_data(pj, args.dt, chs, keep_cols_GRDR=COLS_GRDR)
            except PDSError:
                print(f"Skipping PJ: {pj}")
                skipped_PJs.append(pj)
                continue
            mapped_Ta_pj, mapped_alpha_pj = map_Ta_to_M(IRDR_data_pj=IRDR_data_pj, GRDR_data_pj=GRDR_data_pj, M=M, pj=pj, MI=MI)
            series_Ta[..., i] = mapped_Ta_pj
            series_alpha[..., i] = mapped_alpha_pj
        print(f"Data compilation finished. Skipped PJs: {skipped_PJs}")

        params_str = f"PJs{args.PJs}_CHs{args.ch}_M{M}_dt{args.dt}"
        np.save(f"npy/MWR_Ta_series_{params_str}.npy", series_Ta)
        np.save(f"npy/MWR_alpha_series_{params_str}.npy", series_alpha)

        for pj in range(series_Ta.shape[-1]):
            if pj+1 in skipped_PJs: continue
            plot_data(series_Ta[..., pj], chs, f"Ta_series_PJ{pj+1}_CHs{args.ch}_M{M}_dt{args.dt}", save_path=save_path)

        stacked_Ta_data = stack_data(series_Ta)
        stacked_alpha_data = stack_data(series_alpha)

        np.save(f"npy/MWR_Ta_stacked_{params_str}.npy", stacked_Ta_data)
        np.save(f"npy/MWR_alpha_stacked_{params_str}.npy", stacked_alpha_data)

        BLC_file = f"BLC_M{int(M)}.txt" if M in [2, 3, 5, 10] else None
        plot_data(stacked_Ta_data, chs, f"Ta_series_{params_str}", vmax=VMAX[chs - 1], BLC_file=BLC_file, save_path=save_path)
        plot_data(stacked_alpha_data, chs, f"alpha_series_{params_str}", save_path=save_path)


if __name__ == "__main__":
    main()
