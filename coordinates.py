import numpy as np


def pos_to_angle(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(x**2 + y**2)
    theta = np.acos(z / r)
    phi = np.atan2(y, x) + np.pi
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
