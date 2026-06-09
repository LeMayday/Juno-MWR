# helper functions for downloading and loading data from PDS

# modules
import pandas as pd
import pds.peppi as pep
import numpy as np
import requests

# local files
from coordinates import get_tmin_tmax, lat_lonE, lat_lonW, lat_lon_to_pos

# default
from datetime import datetime
import os


DATA_DIR = os.path.join('.', 'data')
COLS_IRDR = ['t_ephem_time', 't_utc_doy', 'R1_1TA', 'R1_2TA', 'R2_1TA', 'R2_2TA', 'R3TA', 'R4TA', 'R5TA', 'R6TA']
CHANNELS = np.array(['R1_1TA', 'R2_1TA', 'R3TA', 'R4TA', 'R5TA', 'R6TA'])
# time
# values are only provided when given antenna is pointed at Jupiter -- can use as Jupiter mask
# boresight components in VIP4 magnetic coordinate frame (Ch 2-6 are all the same)
# boresight components in SIII coordinate frame (Ch 2-6 are all the same)
# Juno position to Jupiter center (range, long, lat, VIP4, SIII)
COLS_GRDR = ['t_ephem_time', 't_utc_doy', 
             'PC_lon_JsB1', 'PC_lat_JsB1', 'PC_lon_JsB2', 'PC_lat_JsB2',
             'JMag_x_B1', 'JMag_y_B1', 'JMag_z_B1', 'JMag_x_B2', 'JMag_y_B2', 'JMag_z_B2',
             'S3RH_x_B1', 'S3RH_y_B1', 'S3RH_z_B1', 'S3RH_x_B2', 'S3RH_y_B2', 'S3RH_z_B2',
             'range_JnJc', 'PC_lon_JsJnJc', 'PC_lat_JsJnJc', 'JMag_x_JcJn', 'JMag_y_JcJn', 'JMag_z_JcJn', 'S3RH_x_JcJn', 'S3RH_y_JcJn', 'S3RH_z_JcJn']


class NoProductsError(Exception):
    # Raised when PDS does not return any products for query
    # e.g. there is no data on PDS for PJ2 (2016-293T18:10:53)
    pass


class FileDownloadError(Exception):
    # Raised when requests fails to download all of the files, even if PDS returns products
    pass


def find_file(filename: str, pj: int):
    for root, _, files in os.walk(DATA_DIR, f"PJ_{pj}"):
        if filename in files:
            return os.path.join(root, filename)
    return None


def download_file(url: str, fname: str, pj: int):
    # see https://www.geeksforgeeks.org/python/how-to-download-files-from-urls-with-python/
    response = requests.get(url)
    dir_path = os.path.join(DATA_DIR, f"PJ_{pj}")
    os.makedirs(dir_path, exist_ok=True)
    fpath = os.path.join(DATA_DIR, f"PJ_{pj}", fname)
    if response.status_code == 200:
        with open(fpath, 'wb') as file:
            file.write(response.content)
        return fpath
    else:
        print(f'Failed to download file {fname}')
        return None


def PDS_query(t_min: datetime, t_max: datetime, pj: int) -> pd.DataFrame:
    # PJ# is not needed for PDS query, but it helps organize files
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
    try:
        PDS_data_df = products.as_dataframe()[[file_ref_lbl, file_name_lbl]]
    except TypeError as err:
        msg = str(err)
        if "'NoneType' object is not subscriptable" == msg:
            print(f"Query returned no results for PJ {pj}")
            raise NoProductsError
        raise
    # rename for easier access (this affects other functions!)
    PDS_data_df = PDS_data_df.rename(columns={file_ref_lbl: 'urls', file_name_lbl: 'filenames'})
    # replace _V03 with _V04 (for some reason, peppi doesn't grab newest data version)
    cols = ['urls', 'filenames']
    PDS_data_df[cols] = PDS_data_df[cols].map(lambda item: [str(i).replace('_V03', '_V04') for i in item])
    return download_clean_data(PDS_data_df, pj)


def download_data(PDS_data_df: pd.DataFrame, pj: int) -> pd.DataFrame:
    GRDR_list = []
    IRDR_list = []
    files_to_skip = []
    for url_list, fname_list in zip(PDS_data_df['urls'], PDS_data_df['filenames']):
        for url, fname in zip(url_list, fname_list):
            # note RG files are always listed first
            if ".LBL" in fname:         # skip LBL files
                continue
            if fname in files_to_skip:  # skip files whose partner couldn't be downloaded
                continue
            fpath = find_file(fname, pj)
            if fpath is None:           # if file not already downloaded, download file
                fpath = download_file(url, fname, pj)
                if fpath is None:       # if couldn't download, skip and need to remove
                    if 'RG' in fname:
                        files_to_skip.append(fname.replace('RG', 'RI'))
                    elif 'RI' in fname:
                        GRDR_list.remove(fpath.replace('RI', 'RG'))
                    continue
            if "RG" in fname:
                GRDR_list.append(fpath)
            elif "RI" in fname:
                IRDR_list.append(fpath)
    assert len(GRDR_list) == len(IRDR_list), "Lists of GRDR and IRDR file paths should have the same length"
    if len(GRDR_list) == 0:
        raise FileDownloadError
    GRDR_list.sort()    # assumes file naming convention sorts in ascending time order
    IRDR_list.sort()
    return pd.DataFrame({'IRDR_CSV': IRDR_list, 'GRDR_CSV': GRDR_list})


def download_clean_data(PDS_data_df: pd.DataFrame, pj: int) -> pd.DataFrame:
    sorted_filepaths_df = download_data(PDS_data_df, pj)
    # remove unneeded csv columns to reduce file size (Note: This overwrites the csv file!)
    for IRDR_csv_path in sorted_filepaths_df['IRDR_CSV']:
        df = pd.read_csv(IRDR_csv_path, usecols=COLS_IRDR)
        df.to_csv(IRDR_csv_path, index=False)
    for GRDR_csv_path in sorted_filepaths_df['GRDR_CSV']:
        df = pd.read_csv(GRDR_csv_path, usecols=COLS_GRDR)
        df.to_csv(GRDR_csv_path, index=False)
    return sorted_filepaths_df


def load_PJ_data(pj: int, dt: float, chs: np.ndarray, keep_cols_GRDR: list[str] = COLS_GRDR) -> tuple[pd.DataFrame, pd.DataFrame]:
    t_min, t_max = get_tmin_tmax(pj, dt)        # PDS queries by time range
    filepaths_df = PDS_query(t_min, t_max, pj)  # get list of filepaths
    
    # read only the columns pertinent to this processing (e.g. channel specific -- does not change csv)
    keep_cols_IRDR = ['t_ephem_time', 't_utc_doy', *CHANNELS[chs - 1].tolist()]
    
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

    # # transform lat/lon data to x,y,z since interpolation requires continuous fields
    # for lat_col in GRDR_data_pj.columns:
    #     if 'PC_lat' in lat_col:
    #         lon_col = lat_col.replace('lat', 'lon')         # assume column name is the same minus lat/lon
    #         x, y, z = lat_lon_to_pos(GRDR_data_pj[lat_col], GRDR_data_pj[lon_col])
    #         GRDR_data_pj[lat_col.replace('lat', 'x')] = x
    #         GRDR_data_pj[lat_col.replace('lat', 'y')] = y
    #         GRDR_data_pj[lat_col.replace('lat', 'z')] = z

    return IRDR_data_pj, GRDR_data_pj


def get_SIII_lat_lon(GRDR_data_pj: pd.DataFrame, ch: int) -> tuple[np.ndarray, np.ndarray]:
    if ch in range(2, 7): ch = 2
    columns_to_select = [f'S3RH_x_B{ch}', f'S3RH_y_B{ch}', f'S3RH_z_B{ch}']
    data_x, data_y, data_z = np.hsplit(GRDR_data_pj[columns_to_select].to_numpy(), 3)
    return lat_lonW(data_x, data_y, data_z)


def get_VIP4_lat_lon(GRDR_data_pj: pd.DataFrame, ch: int) -> tuple[np.ndarray, np.ndarray]:
    if ch in range(2, 7): ch = 2
    columns_to_select = [f'JMag_x_B{ch}', f'JMag_y_B{ch}', f'JMag_z_B{ch}']
    data_x, data_y, data_z = np.hsplit(GRDR_data_pj[columns_to_select].to_numpy(), 3)
    return lat_lonE(data_x, data_y, data_z)


def get_PC_lat_lon(GRDR_data_pj: pd.DataFrame, ch: int) -> tuple[np.ndarray, np.ndarray]:
    if ch in range(2, 7): ch = 2
    columns_to_select = [f'PC_x_JsB{ch}', f'PC_y_JsB{ch}', f'PC_z_JsB{ch}']
    data_x, data_y, data_z = np.hsplit(GRDR_data_pj[columns_to_select].to_numpy(), 3)
    return lat_lonE(data_x, data_y, data_z)
