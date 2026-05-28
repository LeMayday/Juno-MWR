import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pds.peppi as pep
import argparse
from datetime import datetime
import requests


def get_data(t_min: datetime, t_max, chs: list):
    client = pep.PDSRegistryClient()
    products = pep.Products(client) \
        .has_target("Jupiter") \
        .has_instrument("urn:nasa:pds:context:instrument:mwr.jno") \
        .after(t_min) \
        .before(t_max) \
        .has_processing_level("calibrated") \
        .observationals()

    # Convert to DataFrame for analysis
    df = products.as_dataframe(max_rows=50)



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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--data-dir", required=True, type=str, help="Data Directory")
    # args = parser.parse_args()

    t_min = datetime.strptime("2016-08-27 10:50:44.060", "%Y-%m-%d %H:%M:%S.%f")
    t_max = datetime.strptime("2016-08-27 14:50:44.060", "%Y-%m-%d %H:%M:%S.%f")
    get_data(t_min, t_max, [1])

    # load_data(args.data_dir)

if __name__ == "__main__":
    main()
