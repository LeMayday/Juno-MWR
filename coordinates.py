import numpy as np


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
