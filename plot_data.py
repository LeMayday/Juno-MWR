import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pds.peppi as pep
import argparse
from datetime import datetime
import requests
import os

data_dir = os.path.join('.', 'data')

def find_file(filename: str):
    for root, _, files in os.walk(data_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def download_file(url: str, fname: str):
    # see https://www.geeksforgeeks.org/python/how-to-download-files-from-urls-with-python/
    response = requests.get(url)
    fpath = os.path.join(data_dir, fname)
    if response.status_code == 200:
        with open(fpath, 'wb') as file:
            file.write(response.content)
        return fpath
    else:
        print(f'Failed to download file {fname}')
        return None


def PDS_query(t_min: datetime, t_max) -> pd.DataFrame:
    # query PDS and return time-sorted DataFrame of filenames with desired information
    client = pep.PDSRegistryClient()
    products = pep.Products(client) \
        .has_target("Jupiter") \
        .has_instrument("urn:nasa:pds:context:instrument:mwr.jno") \
        .after(t_min) \
        .before(t_max) \
        .has_processing_level("calibrated") \
        .observationals()
    # grab file urls and names into dataframe
    file_ref_lbl = 'ops:Data_File_Info.ops:file_ref'
    file_name_lbl = 'ops:Data_File_Info.ops:file_name'
    PDS_data_df = products.as_dataframe()[[file_ref_lbl, file_name_lbl]]
    # rename for easier access
    PDS_data_df = PDS_data_df.rename(columns={file_ref_lbl: 'urls', file_name_lbl: 'filenames'})
    # replace _V03 with _V04 (for some reason, peppi doesn't grab newest data version)
    cols = ['urls', 'filenames']
    PDS_data_df[cols] = PDS_data_df[cols].map(lambda item: [str(i).replace('_V03', '_V04') for i in item])
    return download_clean_data(PDS_data_df)


def download_data(PDS_data_df: pd.DataFrame) -> pd.DataFrame:
    IRDR_list = []
    # GRDR_list = []
    for url_list, fname_list in zip(PDS_data_df['urls'], PDS_data_df['filenames']):
        for url, fname in zip(url_list, fname_list):
            if ".LBL" in fname: # skip LBL files
                continue
            if "RG" in fname:   # skip GRDR files for now
                continue
            fpath = find_file(fname)
            if fpath is None:   # if file not already downloaded, download file
                fpath = download_file(url, fname)
            if "RI" in fname:
                IRDR_list.append(fpath)
            # elif "RG" in fname:
            #     GRDR_list.append(fpath)
    IRDR_list.sort()
    # GRDR_list.sort()
    # return pd.DataFrame({'IRDR_CSV': IRDR_list, 'GRDR_CSV': GRDR_list})
    return pd.DataFrame({'IRDR_CSV': IRDR_list})


def download_clean_data(PDS_data_df) -> pd.DataFrame:
    sorted_filepaths_df = download_data(PDS_data_df)
    # remove unneeded csv columns to reduce file size
    for IRDR_csv_path in sorted_filepaths_df['IRDR_CSV']:
        keep_cols = ['t_ephem_time', 't_utc_doy', 'R1_1TA', 'R1_2TA', 'R2_1TA', 'R2_2TA', 'R3TA', 'R4TA', 'R5TA', 'R6TA']
        df = pd.read_csv(IRDR_csv_path, usecols=keep_cols)
        df.to_csv(IRDR_csv_path, index=False)
    return sorted_filepaths_df


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
    os.makedirs(data_dir, exist_ok=True)

    t_min = datetime.strptime("2016-08-27 10:50:44.060", "%Y-%m-%d %H:%M:%S.%f")
    t_max = datetime.strptime("2016-08-27 14:50:44.060", "%Y-%m-%d %H:%M:%S.%f")
    PDS_query(t_min, t_max)

    # load_data(args.data_dir)

if __name__ == "__main__":
    main()
