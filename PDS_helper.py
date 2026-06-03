import pandas as pd
import pds.peppi as pep
import numpy as np
from datetime import datetime
import requests
import os

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
    # rename for easier access
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
    IRDR_list.sort()
    GRDR_list.sort()
    return pd.DataFrame({'IRDR_CSV': IRDR_list, 'GRDR_CSV': GRDR_list})


def download_clean_data(PDS_data_df) -> pd.DataFrame:
    sorted_filepaths_df = download_data(PDS_data_df)
    # remove unneeded csv columns to reduce file size
    for IRDR_csv_path in sorted_filepaths_df['IRDR_CSV']:
        df = pd.read_csv(IRDR_csv_path, usecols=COLS_IRDR)
        df.to_csv(IRDR_csv_path, index=False)
    for GRDR_csv_path in sorted_filepaths_df['GRDR_CSV']:
        df = pd.read_csv(GRDR_csv_path, usecols=COLS_GRDR)
        df.to_csv(GRDR_csv_path, index=False)
    return sorted_filepaths_df
