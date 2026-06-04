# helper functions for time and space coordinates

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from astropy.time import Time


def get_PJ_time(pj: int) -> datetime:
    df = pd.read_table('perijove_times.txt')
    return datetime.strptime(df["Time (UTC/SCET)"].iloc[pj], "%Y-%m-%d %H:%M:%S.%f")


def get_PJ_time_ET(pj: int) -> np.float64:
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html#In%20the%20Toolkit%20ET%20Means%20TDB
    t = Time(get_PJ_time(pj), scale='utc')
    j2000 = Time('2000-01-01T12:00:00', scale='tt')
    return (t - j2000).sec


def get_tmin_tmax(pj: int, dt: float) -> tuple[datetime, datetime]:
    pj_time = get_PJ_time(pj)
    pj_dt = timedelta(minutes=dt)
    return pj_time - pj_dt, pj_time + pj_dt


def pos_to_angle(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.acos(z)
    phi = np.atan2(y, x)
    return theta, phi


def lat_longE(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # longitude increasing to the east
    theta, phi = pos_to_angle(x, y, z)
    lat = 90 - np.rad2deg(theta)
    long = np.rad2deg(phi)
    return lat, long


def lat_longW(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # longitude increasing to the west
    lat, long = lat_longE(x, y, z)
    long = 360 - long
    return lat, long
