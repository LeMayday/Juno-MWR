# modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local files
from coordinates import get_PJ_time_ET
from banana_slug import make_subplots
from PDS_helper import load_PJ_data

# default
import argparse


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


def time_series_argparser(parser: argparse.ArgumentParser):
    parser.add_argument("--PJ", required=True, type=int, help="Perijove")
    parser.add_argument("--dt", required=True, type=float, help="Delta time in minutes")
    parser.add_argument("--ch", required=False, type=str, default="1,2,3,4,5,6", help="List of channels separated by comma")
    return parser


def main():
    parser = argparse.ArgumentParser()
    args = time_series_argparser(parser).parse_args()
    chs = process_ch_inputs(args.ch)
    
    IRDR_data_pj, _ = load_PJ_data(args.PJ, args.dt, chs)
    time_series_plot(IRDR_data_pj, args.PJ)


if __name__ == "__main__":
    main()
