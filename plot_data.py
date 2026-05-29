import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta
import os
from PDS_helper import data_dir, PDS_query


def get_PJ_time(pj: int) -> datetime:
    df = pd.read_table('perijove_times.txt')
    return datetime.strptime(df["Time (UTC/SCET)"].iloc[pj], "%Y-%m-%d %H:%M:%S.%f")


def load_data(data_dir: str):
    data = pd.read_csv(data_dir)
    data_ch1 = data["R1_1TA"].to_numpy()[::10]
    data_utc = data["t_utc_doy"].to_numpy()[::10]

    ref_str = "2016-08-27 12:50:44.060"

    ref_dt = pd.to_datetime(ref_str, format="%Y-%m-%d %H:%M:%S.%f")
    data_utc_df = pd.to_datetime(data["t_utc_doy"], format="%Y-%jT%H:%M:%S.%f")

    minutes_deltas = (data_utc_df - ref_dt).dt.total_seconds() / 60
    minutes_deltas = minutes_deltas.to_numpy()[::10]

    minutes = minutes_deltas[np.logical_and(minutes_deltas > -10, minutes_deltas < 10)]
    data_ch1 = data_ch1[np.logical_and(minutes_deltas > -10, minutes_deltas < 10)]

    fig, ax = plt.subplots(1, 1)
    ax.plot(minutes, data_ch1)
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
    PDS_query(t_min, t_max)

    # load_data(args.data_dir)

if __name__ == "__main__":
    main()
