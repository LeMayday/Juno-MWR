# helper functions for downloading and loading data from PDS

import pandas as pd
import pds.peppi as pep
import numpy as np
from datetime import datetime
import requests
import os
from coordinates import get_tmin_tmax, lat_lonE, lat_lonW

DATA_DIR = os.path.join('.', 'data')
COLS_IRDR = ['t_ephem_time', 't_utc_doy', 'R1_1TA', 'R1_2TA', 'R2_1TA', 'R2_2TA', 'R3TA', 'R4TA', 'R5TA', 'R6TA']
CHANNELS = np.array(['R1_1TA', 'R2_1TA', 'R3TA', 'R4TA', 'R5TA', 'R6TA'])
# time
# values are only provided when given antenna is pointed at Jupiter -- can use as Jupiter mask
# boresight components in VIP4 magnetic coordinate frame
# boresight components in SIII coordinate frame
# Juno position to Jupiter center (range, long, lat, VIP4, SIII)
COLS_GRDR = ['t_ephem_time', 't_utc_doy', 
             'PC_lon_JsB1', 'PC_lat_JsB1', 'PC_lon_JsB2', 'PC_lat_JsB2', 'PC_lon_JsB3', 'PC_lat_JsB3', 'PC_lon_JsB4', 'PC_lat_JsB4', 'PC_lon_JsB5', 'PC_lat_JsB5', 'PC_lon_JsB6', 'PC_lat_JsB6',
             'JMag_x_B1', 'JMag_y_B1', 'JMag_z_B1', 'JMag_x_B2', 'JMag_y_B2', 'JMag_z_B2', 'JMag_x_B3',	'JMag_y_B3', 'JMag_z_B3', 'JMag_x_B4', 'JMag_y_B4', 'JMag_z_B4', 'JMag_x_B5', 'JMag_y_B5', 'JMag_z_B5', 'JMag_x_B6', 'JMag_y_B6', 'JMag_z_B6',
             'S3RH_x_B1', 'S3RH_y_B1', 'S3RH_z_B1', 'S3RH_x_B2', 'S3RH_y_B2', 'S3RH_z_B2', 'S3RH_x_B3', 'S3RH_y_B3', 'S3RH_z_B3', 'S3RH_x_B4', 'S3RH_y_B4', 'S3RH_z_B4', 'S3RH_x_B5', 'S3RH_y_B5', 'S3RH_z_B5', 'S3RH_x_B6', 'S3RH_y_B6', 'S3RH_z_B6',
             'range_JnJc', 'PC_lon_JsJnJc', 'PC_lat_JsJnJc', 'JMag_x_JcJn', 'JMag_y_JcJn', 'JMag_z_JcJn', 'S3RH_x_JcJn', 'S3RH_y_JcJn', 'S3RH_z_JcJn']


def find_file(filename: str):
    for root, _, files in os.walk(DATA_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None


def download_file(url: str, fname: str):
    # see https://www.geeksforgeeks.org/python/how-to-download-files-from-urls-with-python/
    response = requests.get(url)
    fpath = os.path.join(DATA_DIR, fname)
    if response.status_code == 200:
        with open(fpath, 'wb') as file:
            file.write(response.content)
        return fpath
    else:
        print(f'Failed to download file {fname}')
        return None


def PDS_query(t_min: datetime, t_max: datetime) -> pd.DataFrame:
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
    # rename for easier access (this affects other functions!)
    PDS_data_df = PDS_data_df.rename(columns={file_ref_lbl: 'urls', file_name_lbl: 'filenames'})
    # replace _V03 with _V04 (for some reason, peppi doesn't grab newest data version)
    cols = ['urls', 'filenames']
    PDS_data_df[cols] = PDS_data_df[cols].map(lambda item: [str(i).replace('_V03', '_V04') for i in item])
    return download_clean_data(PDS_data_df)


def download_data(PDS_data_df: pd.DataFrame) -> pd.DataFrame:
    IRDR_list = []
    GRDR_list = []
    for url_list, fname_list in zip(PDS_data_df['urls'], PDS_data_df['filenames']):
        for url, fname in zip(url_list, fname_list):
            if ".LBL" in fname: # skip LBL files
                continue
            fpath = find_file(fname)
            if fpath is None:   # if file not already downloaded, download file
                fpath = download_file(url, fname)
            if "RI" in fname:
                IRDR_list.append(fpath)
            elif "RG" in fname:
                GRDR_list.append(fpath)
    IRDR_list.sort()    # assumes file naming convention sorts in ascending time order
    GRDR_list.sort()
    return pd.DataFrame({'IRDR_CSV': IRDR_list, 'GRDR_CSV': GRDR_list})


def download_clean_data(PDS_data_df) -> pd.DataFrame:
    sorted_filepaths_df = download_data(PDS_data_df)
    # remove unneeded csv columns to reduce file size (Note: This overwrites the csv file!)
    for IRDR_csv_path in sorted_filepaths_df['IRDR_CSV']:
        df = pd.read_csv(IRDR_csv_path, usecols=COLS_IRDR)
        df.to_csv(IRDR_csv_path, index=False)
    for GRDR_csv_path in sorted_filepaths_df['GRDR_CSV']:
        df = pd.read_csv(GRDR_csv_path, usecols=COLS_GRDR)
        df.to_csv(GRDR_csv_path, index=False)
    return sorted_filepaths_df


def load_PJ_data(pj: int, dt: float, chs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    t_min, t_max = get_tmin_tmax(pj, dt)        # PDS queries by time range
    filepaths_df = PDS_query(t_min, t_max)      # get list of filepaths
    
    # read only the columns pertinent to this processing (e.g. channel specific -- does not change csv)
    keep_cols_IRDR = ['t_ephem_time', 't_utc_doy', *CHANNELS[chs - 1].tolist()]
    keep_cols_GRDR = []
    for col in COLS_GRDR:   # 'B' is boresight for channel, keep all non-channel cols and only those channels specified
        if 'B' not in col: keep_cols_GRDR.append(col)
        else:
            for ch in [f'B{ch}' for ch in chs]:
                if ch in col: keep_cols_GRDR.append(col)
    
    # read and truncate the data by time and selected columns
    dfs = [[], []]
    for IRDR_path, GRDR_path in zip(filepaths_df['IRDR_CSV'], filepaths_df['GRDR_CSV']):
        IRDR_data = pd.read_csv(IRDR_path, usecols=keep_cols_IRDR)  # filter by desired channels
        GRDR_data = pd.read_csv(GRDR_path, usecols=keep_cols_GRDR)
        IRDR_data['t_utc_doy'] = pd.to_datetime(IRDR_data['t_utc_doy'], format="%Y-%jT%H:%M:%S.%f")
        GRDR_data['t_utc_doy'] = pd.to_datetime(GRDR_data['t_utc_doy'], format="%Y-%jT%H:%M:%S.%f")
        time_mask = (IRDR_data['t_utc_doy'] >= t_min) & (IRDR_data['t_utc_doy'] <= t_max)   # csv's include data outside desired time range
        IRDR_data_filtered = IRDR_data[time_mask]
        GRDR_data_filtered = GRDR_data[time_mask]   # assume sample correspondence btwn IRDR and GRDR
        dfs[0].append(IRDR_data_filtered)
        dfs[1].append(GRDR_data_filtered)
    IRDR_data_pj = pd.concat(dfs[0], ignore_index=True)     # combine multiple files into 1 dataframe
    GRDR_data_pj = pd.concat(dfs[1], ignore_index=True)
    IRDR_data_pj = IRDR_data_pj.rename(columns={'t_ephem_time': 'Time_ET', 't_utc_doy': 'Time_UTC'})
    GRDR_data_pj = GRDR_data_pj.rename(columns={'t_ephem_time': 'Time_ET', 't_utc_doy': 'Time_UTC'})
    IRDR_data_pj = IRDR_data_pj.rename(columns={str(CHANNELS[ch - 1]): f"Ch{ch}" for ch in chs})
    return IRDR_data_pj, GRDR_data_pj


def get_SIII_lat_lon(GRDR_data_pj: pd.DataFrame, ch: int) -> tuple[np.ndarray, np.ndarray]:
    columns_to_select = [col for col in COLS_GRDR if 'S3RH' in col and f'B{ch}' in col]      # expect this to be [x,y,z] for given channel
    data_x, data_y, data_z = np.hsplit(GRDR_data_pj[columns_to_select].to_numpy(), 3)
    return lat_lonW(data_x, data_y, data_z)


def get_VIP4_lat_lon(GRDR_data_pj: pd.DataFrame, ch: int) -> tuple[np.ndarray, np.ndarray]:
    columns_to_select = [col for col in COLS_GRDR if 'JMag' in col and f'B{ch}' in col]      # expect this to be [x,y,z] for given channel
    data_x, data_y, data_z = np.hsplit(GRDR_data_pj[columns_to_select].to_numpy(), 3)
    return lat_lonE(data_x, data_y, data_z)


def get_PC_lat_lon(GRDR_data_pj: pd.DataFrame, ch: int) -> tuple[np.ndarray, np.ndarray]:
    lat = GRDR_data_pj[f'PC_lat_JsB{ch}'].to_numpy()
    lon = GRDR_data_pj[f'PC_lon_JsB{ch}'].to_numpy()
    return lat, lon
