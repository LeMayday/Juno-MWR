import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import argparse
from datetime import datetime, timedelta
import os
from PDS_helper import data_dir, PDS_query

CHANNELS = np.array(['R1_1TA', 'R2_1TA', 'R3TA', 'R4TA', 'R5TA', 'R6TA'])


def get_PJ_time(pj: int) -> datetime:
    df = pd.read_table('perijove_times.txt')
    return datetime.strptime(df["Time (UTC/SCET)"].iloc[pj], "%Y-%m-%d %H:%M:%S.%f")


def get_PJ_time_ET(pj: int) -> np.float64:
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html#In%20the%20Toolkit%20ET%20Means%20TDB
    t = Time(get_PJ_time(pj), scale='utc')
    j2000 = Time('2000-01-01T12:00:00', scale='tt')
    return (t - j2000).sec


def load_PJ_data(filepaths_df: pd.DataFrame, t_min: datetime, t_max: datetime, chs: np.ndarray) -> pd.DataFrame:
    dfs = []
    keep_cols = ['t_ephem_time', 't_utc_doy', *CHANNELS[chs - 1].tolist()]
    for IRDR_path in filepaths_df['IRDR_CSV']:
        IRDR_data = pd.read_csv(IRDR_path, usecols=keep_cols)  # filter by desired channels
        IRDR_data['t_utc_doy'] = pd.to_datetime(IRDR_data['t_utc_doy'], format="%Y-%jT%H:%M:%S.%f")
        time_mask = (IRDR_data['t_utc_doy'] >= t_min) & (IRDR_data['t_utc_doy'] <= t_max)
        IRDR_data_filtered = IRDR_data[time_mask]
        IRDR_data_downsampled = IRDR_data_filtered.iloc[::10]
        dfs.append(IRDR_data_downsampled)
    IRDR_data_pj = pd.concat(dfs, ignore_index=True)
    IRDR_data_pj = IRDR_data_pj.rename(columns={'t_ephem_time': 'Time_ET', 't_utc_doy': 'Time_UTC'})
    IRDR_data_pj = IRDR_data_pj.rename(columns={str(CHANNELS[ch - 1]): f"Ch{ch}" for ch in chs})
    return IRDR_data_pj


def time_series_plot(pj: int, IRDR_data_pj: pd.DataFrame):
    pj_time = get_PJ_time_ET(pj)
    minutes_delta = (IRDR_data_pj['Time_ET'].to_numpy() - pj_time) / 60

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    for ch in IRDR_data_pj.columns[2:]:  # skip time column
        ax.plot(minutes_delta, IRDR_data_pj[ch].to_numpy(), label=ch)
    ax.legend()
    ax.axvline(x=0, ls='--', c='k')
    fig.tight_layout()
    plt.show()


def main():
    os.makedirs(data_dir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--PJ", required=True, type=int, help="Perijove")
    parser.add_argument("--dt", required=True, type=int, help="Delta time in minutes")
    args = parser.parse_args()
    pj_time = get_PJ_time(args.PJ)
    pj_dt = timedelta(minutes=args.dt)
    t_min = pj_time - pj_dt
    t_max = pj_time + pj_dt
    filepaths_df = PDS_query(t_min, t_max)
    chs = [1]
    IRDR_data_pj = load_PJ_data(filepaths_df, t_min, t_max, np.array(chs))
    time_series_plot(args.PJ, IRDR_data_pj)

if __name__ == "__main__":
    main()
