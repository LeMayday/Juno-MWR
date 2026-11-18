import numpy as np
from enum import IntEnum

# type hinting
ONE_D_NDArray   = np.ndarray[tuple[int,], np.dtype[np.float32]]
TWO_D_NDArray   = np.ndarray[tuple[int, int], np.dtype[np.float32]]
THREE_D_NDArray = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
FOUR_D_NDArray  = np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]

# Jupiter
RJ = 71492  # km

# Juno
NU_O = np.array([0.6, 1.248, 2.597, 5.215, 10.004, 21.9, 1.4])   # GHz

class CHS(IntEnum):
    CH1 = 0
    CH2 = 1
    CH3 = 2
    CH4 = 3
    CH5 = 4
    CH6 = 5
    VLA1 = 6

def process_ch_inputs(arg_chs: str) -> list[CHS]:
    try:
        return [CHS(int(ch)-1) for ch in arg_chs.split(',')]
    except ValueError:
        print("Valid channel numbers are 1-6")
        exit(1)

# plotting
VMAX = np.array([5000, 1100, 250, 70, 25, 15])