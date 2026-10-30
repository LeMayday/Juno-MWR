import numpy as np

# type hinting
ONE_D_NDArray   = np.ndarray[tuple[int,], np.dtype[np.float32]]
TWO_D_NDArray   = np.ndarray[tuple[int, int], np.dtype[np.float32]]
THREE_D_NDArray = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
FOUR_D_NDArray  = np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]

# Jupiter
RJ = 71492  # km

# plotting
VMAX = np.array([5000, 1100, 250, 70, 25, 15])