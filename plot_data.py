# modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from unwrap import unwrap

# local files
from PDS_helper import COLS_GRDR, load_PJ_data, get_SIII_lat_lon, get_VIP4_lat_lon, get_PC_lat_lon
from coordinates import get_PJ_time_ET

# default
import argparse
from typing import Optional


def fill_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    # MWR_SIS specifies that every 6th sample is skipped when returning data in full data mode
    # Unfortunately, visual inspection reveals that data rate is not perfectly 100 ms
    dt = 0.1                                                # samples taken every 100 ms
    time = data['Time_ET'].to_numpy()
    time_diff = np.diff(time)                               # difference between sequential elements (N-1)
    # here I assume that time is always increasing and missing time is never >~0.2 (1 missing sample)
    missing = time_diff > dt * 1.75                         # mask where difference is at least ~2*dt (line up with indices after which should be inserted new value)
    missing_places = np.nonzero(missing)[0] + 1             # indices before which should be inserted new value
    missing_values = np.around(time[:-1][missing] + dt, 3)  # GPT suggests rounding to avoid floating point errors
    time_full = np.insert(time, missing_places, missing_values)
    data = data.set_index('Time_ET')
    data_full = data.reindex(time_full)
    # reindexing causes missing time stamps to have NaN data -- interpolate should fill in
    # only interpolate inside gaps of 1 (preserve NaN values in PC lat/lon columns)
    # limitation is if missing time stamp is at the edge of PC data chunk and should have had data
    for col in data.columns:
        data_full[col] = data_full[col].interpolate(method='linear', limit=1, limit_area='inside')
    data_full = data_full.reset_index()
    return data_full


def channel_jupiter_2Dmask(ch: int, GRDR_data_pj: pd.DataFrame, samples_per_rot: int, num_spins: int) -> np.ndarray:
    # planetocentric lat/long of antenna boresight surface intercept seems to be only provided when antenna is looking at Jupiter
    if ch in range(2, 7): ch = 2
    columns_to_select = [col for col in COLS_GRDR if 'PC_lon_Js' in col and f'B{ch}' in col]
    data = GRDR_data_pj[columns_to_select].to_numpy()
    mask = ~np.isnan(data)
    mask_folded = mask[:(num_spins * samples_per_rot)].reshape(-1, samples_per_rot)
    return mask_folded


def make_subplots(fig: Figure, num_chs: int) -> list[Axes]:
    if num_chs <= 2:
        axes = [fig.add_subplot(1, num_chs, ch) for ch in range(1, num_chs + 1)]
        return axes
    nrows = 2
    ncols = (num_chs + 1) // 2  # return 2 if 3 or 4, 3 if 5 or 6
    if num_chs % 2 == 0:
        axes = [fig.add_subplot(nrows, ncols, ch) for ch in range(1, num_chs + 1)]
    else:
        axes = []
        for ch in range(1, num_chs + 1):
            if ch <= ncols: # top row
                axes.append(fig.add_subplot(nrows, ncols * 2, (2 * ch - 1, 2 * ch)))
            else:           # bottom row
                axes.append(fig.add_subplot(nrows, ncols * 2, (2 * ch, 2 * ch + 1)))
    return axes


def time_series_plot(IRDR_data_pj: pd.DataFrame, pj: int):
    IRDR_data_pj = IRDR_data_pj.iloc[::10]  # downsampled to 1/s
    pj_time = get_PJ_time_ET(pj)
    minutes_delta = (IRDR_data_pj['Time_ET'].to_numpy() - pj_time) / 60

    fig = plt.figure(figsize=(12,8))
    axes = make_subplots(fig, len(IRDR_data_pj.columns[2:]))
    for i, ch in enumerate(IRDR_data_pj.columns[2:]):   # skip time column
        axes[i].plot(minutes_delta, IRDR_data_pj[ch].to_numpy(), label=ch)
        axes[i].axvline(x=0, ls='--', c='k')
        axes[i].set_title(ch)
    fig.tight_layout()
    plt.show()


def banana_plot(IRDR_data_pj: pd.DataFrame, GRDR_data_pj: pd.DataFrame, params_str: str, overlay_type: Optional[str] = None):
    IRDR_data_pj = fill_missing_data(IRDR_data_pj)
    GRDR_data_pj = fill_missing_data(GRDR_data_pj)
    samples_per_rot = 307   # 30.7 s/rot Santos-Costa+2017, 1 sample per 100 ms
    fig = plt.figure(figsize=(12,8))
    axes = make_subplots(fig, len(IRDR_data_pj.columns[2:]))
    for i, ch in enumerate(IRDR_data_pj.columns[2:]):   # skip time columns
        data = IRDR_data_pj[ch].to_numpy()
        num_spins = data.size // samples_per_rot
        data = data[:(num_spins * samples_per_rot)]     # need to truncate data to integer # of rotations
        data_folded = data.reshape(-1, samples_per_rot)
        im = axes[i].imshow(data_folded, cmap='gist_ncar', aspect='auto', origin='lower')
        fig.colorbar(im, ax=axes[i])
        # note ch is "Ch{i}"
        add_contours(axes[i], int(ch[2]), GRDR_data_pj, samples_per_rot, num_spins, overlay_type)
    fig.tight_layout()
    fig.savefig(f"MWR_banana_{params_str}_{overlay_type}.png", dpi=300)


def add_contours(ax: Axes, ch: int, GRDR_data_pj: pd.DataFrame, samples_per_rot: int, num_spins: int, type: Optional[str]):
    # valid types: SIII, SIII_NJ, VIP4, VIP4_NJ
    if type is None:
        return
    bkgd_lat = None; bkgd_lon = None; frgd_lat = None; frgd_lon = None
    if 'SIII' in type:
        bkgd_lat, bkgd_lon = get_SIII_lat_lon(GRDR_data_pj, ch)
    elif 'VIP4' in type:
        bkgd_lat, bkgd_lon = get_VIP4_lat_lon(GRDR_data_pj, ch)
    # if 'PC' in type:
    #     frgd_lat, frgd_lon = get_PC_lat_lon(GRDR_data_pj, ch)
    # options are bkgd no frgd, bkgd no jupiter, bkgd w frgd, frgd no bkgd
    if bkgd_lat is not None:
        bkgd_lat_folded = bkgd_lat[:(num_spins * samples_per_rot)].reshape(-1, samples_per_rot)     # truncate data
        bkgd_lon_folded = bkgd_lon[:(num_spins * samples_per_rot)].reshape(-1, samples_per_rot)
        bkgd_lon_folded = np.rad2deg(unwrap(np.deg2rad(bkgd_lon_folded)))                           # 2D phase unwraping to ensure continuous contours
        if frgd_lat is not None or 'NJ' in type:
            jupiter_mask = channel_jupiter_2Dmask(ch, GRDR_data_pj, samples_per_rot, num_spins)     # could also get this from frgd data?
            bkgd_lat_folded[jupiter_mask] = np.nan
            bkgd_lon_folded[jupiter_mask] = np.nan
        plot_contours(ax, bkgd_lat_folded, bkgd_lon_folded)
    # if frgd_lat is not None:
    #     frgd_lat_folded = frgd_lat[:(num_spins * samples_per_rot)].reshape(-1, samples_per_rot)     # truncate data
    #     frgd_lon_folded = frgd_lon[:(num_spins * samples_per_rot)].reshape(-1, samples_per_rot)
    #     frgd_lon_folded_masked = np.ma.masked_invalid(frgd_lon_folded)                              # unwrap also allows masked arrays
    #     frgd_lon_folded = np.rad2deg(unwrap(np.deg2rad(frgd_lon_folded_masked)))                    # 2D phase unwraping to ensure continuous contours
    #     plot_contours(ax, frgd_lat_folded, frgd_lon_folded)


def plot_contours(ax: Axes, lat_folded: np.ndarray, lon_folded: np.ndarray):
    # lat/long are folded, and long is unwrapped
    assert lat_folded.shape == lon_folded.shape, "Lat / Long are wrong sizes (how did you do this???)"
    num_spins, samples_per_rot = lat_folded.shape
    x = np.arange(samples_per_rot)
    y = np.arange(num_spins)
    X, Y = np.meshgrid(x, y)    # set up image grid
    lat_contours = ax.contour(X, Y, lat_folded, levels=range(-90, 91, 30), colors='white', linestyles='solid', negative_linestyles='dotted')
    ax.clabel(lat_contours, inline=True, fmt='%.0f')
    # long contours use levels outside [0, 360) since data is phase unwrapped
    lon_contours = ax.contour(X, Y, lon_folded, levels=range(-360, 720, 30), colors='white', linestyles='dashed')
    ax.clabel(lon_contours, inline=True, fmt=lambda x: f"{int(round(x)) % 360}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--PJ", required=True, type=int, help="Perijove")
    parser.add_argument("--dt", required=True, type=float, help="Delta time in minutes")
    parser.add_argument("--ch", required=False, type=str, default="1,2,3,4,5,6", help="List of channels separated by comma")
    parser.add_argument("--type", required=False, type=str, default=None, choices=["SIII", "SIII_NJ", "VIP4", "VIP4_NJ"], help="Contour overlay type")
    args = parser.parse_args()
    chs = [int(ch) for ch in args.ch.split(',')]
    for ch in chs: assert ch in range(1, 7), "Valid channel numbers are 1-6"    # validate channel inputs

    IRDR_data_pj, GRDR_data_pj = load_PJ_data(args.PJ, args.dt, np.array(chs))
    time_series_plot(IRDR_data_pj, args.PJ)
    banana_plot(IRDR_data_pj, GRDR_data_pj, f"PJs{args.PJ}_CHs{args.ch}_dt{args.dt}", args.type)


if __name__ == "__main__":
    main()
