# modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local files
from coordinates import get_PJ_time_ET
from banana_slug import make_subplots, fill_missing_data, PJ_DSPR, IM_PATH
from PDS_helper import load_PJ_data, get_SIII_lat_lon
from global_vars import process_ch_inputs, CHS

# default
import argparse


def time_series_plot(IRDR_data_pj: pd.DataFrame, pj: int):
    time_series_plot_np(IRDR_data_pj=IRDR_data_pj.to_numpy(),
                        CHS=[CHS[ch] for ch in IRDR_data_pj.columns[2:]], pj=pj)


def time_series_plot_np(IRDR_data_pj: np.ndarray, CHS: list[CHS], pj: int):
    IRDR_data_pj = IRDR_data_pj[::10]                       # downsampled to 1/s
    pj_time = get_PJ_time_ET(pj)
    minutes_delta = (IRDR_data_pj[:, 0] - pj_time) / 60     # ['Time_ET'] is 0th column

    fig = plt.figure(figsize=(12,8))
    axes = make_subplots(fig, IRDR_data_pj.shape[1] - 2)    # all but time columns
    for i, CH in enumerate(CHS):
        axes[i].plot(minutes_delta, IRDR_data_pj[:, i + 2], label=CH.name)
        axes[i].axvline(x=0, ls='--', c='k')
        axes[i].set_title(CH.name)
    fig.tight_layout()
    plt.show()


def single_spin_plot(IRDR_data_pj: pd.DataFrame, GRDR_data_pj: pd.DataFrame, CH: CHS, spin: int, pj: int):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    IRDR_data_pj = fill_missing_data(IRDR_data_pj)
    GRDR_data_pj = fill_missing_data(GRDR_data_pj)
    samples_per_rot = PJ_DSPR.get(pj)

    data = IRDR_data_pj[CH.name].to_numpy()
    num_spins = data.size // samples_per_rot
    data = data[:(num_spins * samples_per_rot)]     # need to truncate data to integer # of rotations
    data_folded = data.reshape(-1, samples_per_rot)

    time_series = data_folded[spin, :]
    time = np.arange(samples_per_rot)/10
    ax.plot(time, time_series, color='black')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{CH.name} Brightness Temp (K)")
    _, bkgd_lon = get_SIII_lat_lon(GRDR_data_pj, CH)
    bkgd_lon_folded = bkgd_lon[:(num_spins * samples_per_rot)].reshape(-1, samples_per_rot)[spin, :]
    ax.set_xlim([0, samples_per_rot/10])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    N = 20
    ticks = time[::N]
    longs = [f"${round(lons)}^\\circ$" for lons in bkgd_lon_folded[::N]]
    ax2.set_xticks(ticks, labels=longs, rotation=60)
    ax2.set_xlabel("Boresight Direction (SIII W Longitude)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.show()
    fig.savefig(f"{IM_PATH}/MWR_onerot_PJ25_CH1.png", dpi=300)


def time_series_argparser(parser: argparse.ArgumentParser):
    parser.add_argument("--PJ", required=True, type=int, help="Perijove")
    parser.add_argument("--dt", required=True, type=float, help="Delta time in minutes")
    parser.add_argument("--ch", required=False, type=str, default="1,2,3,4,5,6", help="List of channels separated by comma")
    parser.add_argument("--spin", required=False, type=int, help="If provided, will plot the provided row of the banana slug plot corresponding to PJ, dt, ch")
    return parser


def main():
    parser = argparse.ArgumentParser()
    args = time_series_argparser(parser).parse_args()
    chs = process_ch_inputs(args.ch)
    
    IRDR_data_pj, GRDR_data_pj = load_PJ_data(args.PJ, args.dt, chs)
    time_series_plot(IRDR_data_pj, args.PJ)
    if args.spin is not None:
        for CH in chs: single_spin_plot(IRDR_data_pj, GRDR_data_pj, CH, args.spin, args.PJ)


if __name__ == "__main__":
    main()
