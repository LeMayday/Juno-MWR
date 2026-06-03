import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy.time import Time
import argparse
from datetime import datetime, timedelta
import os
from PDS_helper import DATA_DIR, CHANNELS, COLS_GRDR, PDS_query
from coordinates import lat_longE, lat_longW
from unwrap import unwrap


def get_PJ_time(pj: int) -> datetime:
    df = pd.read_table('perijove_times.txt')
    return datetime.strptime(df["Time (UTC/SCET)"].iloc[pj], "%Y-%m-%d %H:%M:%S.%f")


def get_PJ_time_ET(pj: int) -> np.float64:
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html#In%20the%20Toolkit%20ET%20Means%20TDB
    t = Time(get_PJ_time(pj), scale='utc')
    j2000 = Time('2000-01-01T12:00:00', scale='tt')
    return (t - j2000).sec


def load_PJ_data(filepaths_df: pd.DataFrame, t_min: datetime, t_max: datetime, chs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = [[], []]
    keep_cols_IRDR = ['t_ephem_time', 't_utc_doy', *CHANNELS[chs - 1].tolist()]
    keep_cols_GRDR = []
    for col in COLS_GRDR:   # 'B' is boresight for channel, keep all non-channel cols and only those channels specified
        if 'B' not in col: keep_cols_GRDR.append(col)
        else:
            for ch in [f'B{ch}' for ch in chs]:
                if ch in col: keep_cols_GRDR.append(col)
    for IRDR_path, GRDR_path in zip(filepaths_df['IRDR_CSV'], filepaths_df['GRDR_CSV']):
        IRDR_data = pd.read_csv(IRDR_path, usecols=keep_cols_IRDR)  # filter by desired channels
        GRDR_data = pd.read_csv(GRDR_path, usecols=keep_cols_GRDR)
        IRDR_data['t_utc_doy'] = pd.to_datetime(IRDR_data['t_utc_doy'], format="%Y-%jT%H:%M:%S.%f")
        GRDR_data['t_utc_doy'] = pd.to_datetime(GRDR_data['t_utc_doy'], format="%Y-%jT%H:%M:%S.%f")
        time_mask = (IRDR_data['t_utc_doy'] >= t_min) & (IRDR_data['t_utc_doy'] <= t_max)
        IRDR_data_filtered = IRDR_data[time_mask]
        GRDR_data_filtered = GRDR_data[time_mask]   # assume sample correspondence btwn IRDR and GRDR
        dfs[0].append(IRDR_data_filtered)
        dfs[1].append(GRDR_data_filtered)
    IRDR_data_pj = pd.concat(dfs[0], ignore_index=True)
    GRDR_data_pj = pd.concat(dfs[1], ignore_index=True)
    IRDR_data_pj = IRDR_data_pj.rename(columns={'t_ephem_time': 'Time_ET', 't_utc_doy': 'Time_UTC'})
    GRDR_data_pj = GRDR_data_pj.rename(columns={'t_ephem_time': 'Time_ET', 't_utc_doy': 'Time_UTC'})
    IRDR_data_pj = IRDR_data_pj.rename(columns={str(CHANNELS[ch - 1]): f"Ch{ch}" for ch in chs})
    return IRDR_data_pj, GRDR_data_pj


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
    for col in data.columns:
        data_full[col] = data_full[col].interpolate(method='linear')
    data_full = data_full.reset_index()
    return data_full


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


def banana_plot(IRDR_data_pj: pd.DataFrame, GRDR_data_pj: pd.DataFrame):
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
        add_SIII_contours(axes[i], ch[2], GRDR_data_pj, samples_per_rot, num_spins)
        # add_VIP4_contours(axes[i], ch[2], GRDR_data_pj, samples_per_rot, num_spins)
    fig.tight_layout()
    plt.show()


def add_SIII_contours(ax: Axes, ch: int, GRDR_data_pj: pd.DataFrame, samples_per_rot: int, num_spins: int):
    columns_to_select = [col for col in COLS_GRDR if 'S3RH' in col and f'B{ch}' in col]      # expect this to be [x,y,z] for given channel
    data_x, data_y, data_z = np.hsplit(GRDR_data_pj[columns_to_select].to_numpy(), 3)
    lat, longW = lat_longW(data_x, data_y, data_z)
    add_contours(ax, lat, longW, samples_per_rot, num_spins)


def add_VIP4_contours(ax: Axes, ch: int, GRDR_data_pj: pd.DataFrame, samples_per_rot: int, num_spins: int):
    columns_to_select = [col for col in COLS_GRDR if 'JMag' in col and f'B{ch}' in col]      # expect this to be [x,y,z] for given channel
    data_x, data_y, data_z = np.hsplit(GRDR_data_pj[columns_to_select].to_numpy(), 3)
    lat, longW = lat_longE(data_x, data_y, data_z)
    add_contours(ax, lat, longW, samples_per_rot, num_spins)


def add_contours(ax: Axes, lat: np.ndarray, long: np.ndarray, samples_per_rot: int, num_spins: int):
    x = np.arange(samples_per_rot)
    y = np.arange(num_spins)
    X, Y = np.meshgrid(x, y)    # set up image grid
    lat_folded = lat[:(num_spins * samples_per_rot)].reshape(-1, samples_per_rot)           # truncate data
    long_folded = long[:(num_spins * samples_per_rot)].reshape(-1, samples_per_rot)
    long_folded = np.rad2deg(unwrap(np.deg2rad(long_folded)))
    lat_contours = ax.contour(X, Y, lat_folded, levels=range(-90, 91, 30), colors='white', linestyles='solid', negative_linestyles='dotted')
    ax.clabel(lat_contours, inline=True, fmt='%.0f')
    long_contours = ax.contour(X, Y, long_folded, levels=range(-360, 720, 30), colors='white', linestyles='dashed') # need to include larger range since data is wrapped
    ax.clabel(long_contours, inline=True, fmt=lambda x: f"{int(round(x)) % 360}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--PJ", required=True, type=int, help="Perijove")
    parser.add_argument("--dt", required=True, type=float, help="Delta time in minutes")
    parser.add_argument("--ch", required=False, type=str, default="1,2,3,4,5,6", help="List of channels separated by comma")
    args = parser.parse_args()
    chs = [int(ch) for ch in args.ch.split(',')]
    # validate channel inputs
    for ch in chs: assert ch in range(1, 7), "Valid channel numbers are 1-6"
    pj_time = get_PJ_time(args.PJ)
    pj_dt = timedelta(minutes=args.dt)
    t_min = pj_time - pj_dt
    t_max = pj_time + pj_dt
    filepaths_df = PDS_query(t_min, t_max)
    IRDR_data_pj, GRDR_data_pj = load_PJ_data(filepaths_df, t_min, t_max, np.array(chs))
    time_series_plot(IRDR_data_pj, args.PJ)
    banana_plot(IRDR_data_pj, GRDR_data_pj)


if __name__ == "__main__":
    main()
