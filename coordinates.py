# helper functions for time and space coordinates

# modules
import numpy as np
import pandas as pd
from astropy.time import Time

# default
from datetime import datetime, timedelta


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


def lat_lonE(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # longitude increasing to the east
    theta, phi = pos_to_angle(x, y, z)
    lat = 90 - np.rad2deg(theta)
    lon = np.rad2deg(phi)
    return lat, lon


def lat_lonW(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # longitude increasing to the west
    lat, lon = lat_lonE(x, y, z)
    lon = (360 - lon) % 360
    return lat, lon


def lat_lon_to_pos(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.sin(np.deg2rad(lat))
    lon_rad = np.deg2rad(lon)
    x = np.cos(lon_rad)
    y = np.sin(lon_rad)
    return x, y, z
