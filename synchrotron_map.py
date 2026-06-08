# modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import binned_statistic
import healpy as hp

# local files
from plot_data import make_subplots
from PDS_helper import DATA_DIR, load_PJ_data
from coordinates import lat_lonW, lat_lonE

# default
import argparse
import os

RJ = 71492  # km
# only the cols needed for the map
COLS_GRDR_MAP = ['t_ephem_time', 't_utc_doy',
                 'JMag_x_B1', 'JMag_y_B1', 'JMag_z_B1', 'JMag_x_B2', 'JMag_y_B2', 'JMag_z_B2',
                 'S3RH_x_B1', 'S3RH_y_B1', 'S3RH_z_B1', 'S3RH_x_B2', 'S3RH_y_B2', 'S3RH_z_B2',
                 'range_JnJc', 'JMag_x_JcJn', 'JMag_y_JcJn', 'JMag_z_JcJn', 'S3RH_x_JcJn', 'S3RH_y_JcJn', 'S3RH_z_JcJn']


def parse_PJs(PJ_str: str) -> list[int]:
    res = []
    for PJ_range in PJ_str.split(','):
        if '-' in PJ_range:
            PJ_start = PJ_range.split('-')[0]
            PJ_end = PJ_range.split('-')[1]
            res.append(*range(PJ_start, PJ_end + 1))
        else:
            res.append(int(PJ_range))
    return res


def compile_data(pjs: list[int], dt: int, chs: np.ndarray, da: float, R_sync: float, nside: int = 128) -> np.ndarray:    
    # healpix has pixels ordered by index, so (lat, lon) -> (npix)
    # see https://lambda.gsfc.nasa.gov/toolbox/pixelcoords.html for nside -> npix
    # create numpy array that is (#chs, #pix, #pjs) so i can take median over pjs
    res = np.empty((len(chs), hp.nside2npix(nside), len(pjs)))
    res[:] = np.nan     # initialize as NaNs
    for i, pj in enumerate(pjs):
        IRDR_data_pj, GRDR_data_pj = load_PJ_data(pj, dt, chs, keep_cols_GRDR=COLS_GRDR_MAP)
        # grab relevant columns
        Jn_SIII = GRDR_data_pj[['S3RH_x_JcJn', 'S3RH_y_JcJn', 'S3RH_z_JcJn']].to_numpy()        # apparently this is not normalized
        Jn_range = np.linalg.norm(Jn_SIII, axis=1)                                         
        Jn_SIII_norm = Jn_SIII / np.expand_dims(Jn_range, axis=1)                               # normalized
        boresight_SIII_1 = GRDR_data_pj[[f'S3RH_x_B1', f'S3RH_y_B1', f'S3RH_z_B1']].to_numpy()  # normalized
        boresight_SIII_2 = GRDR_data_pj[[f'S3RH_x_B2', f'S3RH_y_B2', f'S3RH_z_B2']].to_numpy()  # normalized
        # filter by range
        range_mask = Jn_range < R_sync * RJ
        # filter by angle condition -- masks should have same size as original dataframe
        angle_mask_1 = np.einsum('ij,ij->i', Jn_SIII_norm, boresight_SIII_1) > np.cos(np.deg2rad(da))
        angle_mask_2 = np.einsum('ij,ij->i', Jn_SIII_norm, boresight_SIII_2) > np.cos(np.deg2rad(da))
        mask1 = np.logical_and(range_mask, angle_mask_1)
        mask2 = np.logical_and(range_mask, angle_mask_2)
        # now compute true lat/lon
        b_scale = R_sync - np.expand_dims(Jn_range, axis=1) / RJ                                # units of Rj
        for j, ch in enumerate(chs):
            T_a = IRDR_data_pj[f"Ch{ch}"]   # antenna temperature
            if ch == 1:
                s = Jn_SIII[mask1] / RJ + boresight_SIII_1[mask1] * b_scale[mask1]              # units of Rj
                T_a = T_a[mask1]
            else:
                s = Jn_SIII[mask2] / RJ + boresight_SIII_2[mask2] * b_scale[mask2]              # units of Rj
                T_a = T_a[mask2]
            s = s / np.expand_dims(np.linalg.norm(s, axis=1), axis=1)                           # normalized
            s_x, s_y, s_z = np.hsplit(s, 3)
            true_lat, true_lon = lat_lonW(s_x, s_y, s_z)    # lat [-90,90], lon [0, 360)
            true_lat = true_lat.squeeze(); true_lon = true_lon.squeeze()    # inputs to healpy must be (N,)
            assert T_a.shape == true_lat.shape == true_lon.shape, "T_a, true_lat, and true_lon must have same shape!"
            binned_medians = bin_data(T_a, true_lat, true_lon, nside)
            res[j, :, i] = binned_medians     # everything else should still be NaN
    return res


def bin_data(T_a: np.ndarray, true_lat: np.ndarray, true_lon: np.ndarray, nside) -> np.ndarray:
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, true_lon, true_lat, nest=True, lonlat=True)     # compute pixel ids of (lon, lat) pairs, lon in [0, 360]
    med, _, _ = binned_statistic(x=pix, values=T_a, statistic="median", bins=npix, range=(-0.5, npix - 0.5))    # I love scipy
    return med


def stack_data(data: np.ndarray) -> np.ndarray:
    # assume data is (#chs, #pix, #pjs) -- take median of non-NaN entries over pj axis
    return np.nanmedian(data, axis=2)


def plot_swath(chs: np.ndarray, data: np.ndarray):
    fig = plt.figure(figsize=(12,8))
    axes = make_subplots(fig, len(chs))
    for i, ch in enumerate(chs):   # skip time columns
        projected_map = hp.cartview(data[i, :], nest=True, return_projected_map=True)       # by default, E should be to the left
        im = axes[i].imshow(projected_map, origin='lower', cmap='gist_ncar', aspect='auto', extent=[-180, 180, -90, 90])
        axes[i].set_xticks(range(-180, 181, 30))
        axes[i].set_xticklabels(range(180, -181, -30))
        axes[i].set_yticks(range(-90, 91, 15))
        axes[i].grid()
        fig.colorbar(im, ax=axes[i])
        # note ch is "Ch{i}"
        # add_contours(axes[i], ch[2], GRDR_data_pj, samples_per_rot, num_spins, overlay_type)
    fig.tight_layout()
    plt.show()


def add_contours(ax: Axes, nlon: int, nlat: int):
    x = np.arange(nlon)
    y = np.arange(nlat)
    X, Y = np.meshgrid(x, y)    # set up image grid
    lat_SIII = np.linspace(-90, 90, nlat)
    lon_SIII = np.flip(np.linspace(-180, 180, nlon))
    Lat_SIII, Lon_SIII = np.meshgrid(lat_SIII, lon_SIII)
    lat_contours = ax.contour(X, Y, Lat_SIII, levels=range(-90, 91, 15), colors='black', linestyles='solid')
    ax.clabel(lat_contours, inline=True, fmt='%.0f')
    lon_contours = ax.contour(X, Y, Lon_SIII, levels=range(-180, 180, 30), colors='white', linestyles='dashed')
    ax.clabel(lon_contours, inline=True, fmt=lambda x: f"{int(round(x)) % 360}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", required=True, type=float, help="Delta time around each perijove in minutes")
    parser.add_argument("--da", required=True, type=float, help="Delta angle tolerance btwn boresight and s/c lat/lon")
    parser.add_argument("--ch", required=False, type=str, default="1,2,3,4,5,6", help="List of channels separated by comma")
    parser.add_argument("--PJs", required=True, type=str, help="Perijove range (e.g. 1,2,5,6 or 1-7 or 1,3-6)")
    parser.add_argument("--Rsync", required=False, type=float, default=1.4, help="Assumed radius of synchrotron emission (in Rj)")
    args = parser.parse_args()
    chs = np.array([int(ch) for ch in args.ch.split(',')])
    for ch in chs: assert ch in range(1, 7), "Valid channel numbers are 1-6"
    pjs = parse_PJs(args.PJs)
    for pj in pjs: assert pj in range(1, 78), "Valid perijoves numbers are 1-77"

    time_series_data = compile_data(pjs, args.dt, chs, args.da, args.Rsync)
    stacked_data = stack_data(time_series_data)
    plot_swath(chs, stacked_data)


if __name__ == "__main__":
    # e.g. --PJs 1 --dt 90.25 --da 10 --ch 1
    main()
