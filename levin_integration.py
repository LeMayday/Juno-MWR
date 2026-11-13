# modules
import numpy as np
from numpy.linalg import norm
import pandas as pd

# local files
from global_vars import ONE_D_NDArray, TWO_D_NDArray, THREE_D_NDArray, CHS, NU_O
from mshell_map import alpha, alpha_eq, MeshInterpolator, mshell_argparser, make_Rlos
from bessel_functions import F, F_p
from banana_slug import banana_argparser, banana_plot
from PDS_helper import load_PJ_data, get_Juno_pos_RJ, get_boresight_dir

# default
import argparse


E0 = 5000; EPS = 2.25   # eqn 11
MJ = 4.218              # [gauss RJ^3] eqn 9
C = 1.09E7              # C (erg/s/Gauss) * c^2 (cm/s)^2 / k (erg/K) / (Hz/GHz)^2 * (cm/RJ)
E_2PIMC = 2.79E-3       # GHz / Gauss
nT_to_Gauss = 1E-5
                    # n1, n2,   A_L,    B_L
TABLE_1 = np.array([[1.0, 50,   6.0,   28.0],
                    [1.0, 46,   8.5,   14.5],
                    [1.0, 40,  22.0,   10.4],
                    [1.0, 40,  44.0,   25.5],
                    [1.0, 40,  45.0,   35.7],
                    [1.0, 40,  90.0,  130.0],
                    [1.0, 40, 125.0,  450.0],
                    [1.0, 40, 120.0, 1160.0],
                    [np.nan, np.nan, np.nan, np.nan]])


def table1_lookup(L_los: TWO_D_NDArray) -> THREE_D_NDArray:
    L_los[(L_los > 1.36) & (L_los <= 1.44)] = 0
    L_los[(L_los > 1.44) & (L_los <= 1.78)] = 1
    L_los[(L_los > 1.78) & (L_los <= 2.00)] = 2
    L_los[(L_los > 2.00) & (L_los <= 2.24)] = 3
    L_los[(L_los > 2.24) & (L_los <= 2.48)] = 4
    L_los[(L_los > 2.48) & (L_los <= 2.80)] = 5
    L_los[(L_los > 2.80) & (L_los <= 3.30)] = 6
    L_los[(L_los > 3.30) & (L_los <= 3.90)] = 7
    L_los[(L_los <= 1.36) | (L_los > 3.90)] = 8     # populate out of bounds with nan
    L_los[~np.isfinite(L_los)] = 8                  # populate nonfinite values with nan
    return TABLE_1[L_los]   # should be N x M x 4


def gamma(E: np.ndarray) -> np.ndarray:
    """
    E in MeV
    """
    return E / 0.511 + 1


def N(alpha: TWO_D_NDArray, B: TWO_D_NDArray, Beq: TWO_D_NDArray, E_range: ONE_D_NDArray) -> THREE_D_NDArray:
    """
    Number density (assume cm^-3) per unit energy per unit pitch angle --- eqns 8-11

    alpha: local pitch angles (radians) (num samples, num LOS pts)

    B: local B magnitude (Gauss) (num samples, num LOS pts)

    Beq: mapped equatorial B magnitude (Gauss) (num samples, num LOS pts)

    E_range: range of energies over which to integrate (MeV)
    """
    a_eq = alpha_eq(alpha, Beq, B)                  # num samples x num LOS pts
    L = (MJ / Beq)**(1/3)                           # eqn 9 --- num samples x num LOS pts
    mp = table1_lookup(L)                           # model params
    n_ea = mp[..., 2] * np.sin(a_eq)**(mp[..., 0]) + mp[..., 3] * np.sin(a_eq)**(mp[..., 1])    # eqn 10 --- num samples x num LOS pts
    n_eE = E0 / ( E0 + ( E_range[None, None, :] / np.sqrt(B)[:, :, None] )**(EPS + 0.75) )      # eqn 11 --- num samples x num LOS pts x num E pts
    return n_ea[:, :, None] * n_eE


def I(B_Gauss: THREE_D_NDArray, cos2chi: TWO_D_NDArray, alpha: TWO_D_NDArray, n_e: THREE_D_NDArray, E_range: ONE_D_NDArray, s_range: ONE_D_NDArray, CH: CHS) -> TWO_D_NDArray:
    """
    Return stokes I/C per unit RJ

    B_vec: local B magnitude (Gauss) (num samples, num LOS pts)

    cos2chi: cos(2*chi), where chi is depends on MWR pointing geometry (num samples, num LOS pts)

    alpha: local pitch angle (radians) (num samples, num LOS pts)

    n_e: electron density (cm^-3 rad^-1 MeV^-1)

    E_range: vector of energies over which to integrate (MeV)

    s_range: vector of distances over which to integrate (RJ)

    CH: channel enum
    """
    sin_alpha = np.sin(alpha)
    nu_c = 3/2 * gamma(E_range)**2[None, None, :] * E_2PIMC * B_Gauss[:, :, None] * sin_alpha[:, :, None]   # num samples x num LOS pts x num E pts

    # num samples x num LOS pts x num E pts
    stokes_I_C = B_Gauss[:, :, None] * sin_alpha[:, :, None] *   F(NU_O[CH] / nu_c[...]) * n_e[...]    # missing C*2pi
    stokes_Q_C = B_Gauss[:, :, None] * sin_alpha[:, :, None] * F_p(NU_O[CH] / nu_c[...]) * n_e[...]    # missing C*2pi

    stokes_I_C_prime = 0.5 * (stokes_I_C + stokes_Q_C * cos2chi[:, :, None])      # polarized receiver

    stokes_I_C_prime = np.trapezoid(stokes_I_C_prime, E_range, axis=-1)     # integrate over E
    stokes_I_C_prime = np.trapezoid(stokes_I_C_prime, s_range, axis=-1)     # integrate over LOS --- output should be num channels x num samples
    return stokes_I_C_prime


def T(I_C: TWO_D_NDArray, CH: CHS) -> TWO_D_NDArray:
    """
    Takes Stokes parameter as (num channels, num samples)
    """
    return C * I_C / NU_O[CH]**2


def los_integrate(r_los: THREE_D_NDArray, r_b: TWO_D_NDArray, MI: MeshInterpolator, E_range: ONE_D_NDArray, s_range: ONE_D_NDArray, CH: CHS) -> ONE_D_NDArray:
    B_los = MI.B(r_los)
    B_los_norm = norm(B_los, axis=-1)
    Beq_los = MI.Beq(r_los)
    Beq_los_norm = norm(Beq_los, axis=-1)
    Bin_los = MI.Bin(r_los)
    Bin_los_norm = norm(Bin_los, axis=-1)
    Bis_los = MI.Bis(r_los)
    Bis_los_norm = norm(Bis_los, axis=-1)

    alpha_los = alpha(r_b, B_los)
    sin_alpha = np.sin(alpha_los)

    B_mirror = B_los_norm / sin_alpha**2
    loss_cone_mask = B_mirror > np.minimum(Bin_los_norm, Bis_los_norm)      # perhaps not valid if not dipole?

    n_e = N(alpha=alpha_los, B=B_los_norm, Beq=Beq_los_norm, E_range=E_range)
    n_e[loss_cone_mask, :] = np.nan     # num samples x num LOS pts x num E pts
    n_e[~np.isfinite(n_e)] = 0          # reset nan values to 0 for integration

    # Bx = np.einsum('ijk, ik -> ij', B_vec, x_vec)
    # By = np.einsum('ijk, ik -> ij', B_vec, y_vec)
    # cos2chi = (Bx**2 - By**2) / (Bx**2 + By**2)     # comes from tan(chi) = By / Bx

    I_C = I(B_Gauss=(B_los_norm * nT_to_Gauss), cos2chi=np.ones_like(B_los_norm), alpha=alpha_los, n_e=n_e, E_range=E_range, s_range=s_range, CH=CH)
    return T(I_C, CH=CH)


def imitate_IRDR(GRDR_data_pj: pd.DataFrame, MI: MeshInterpolator, E_range: ONE_D_NDArray, sN: int) -> pd.DataFrame:
    smin=0.2; smax=7
    s_range = np.linspace(smin, smax, sN + 1)
    Jn_SIII = get_Juno_pos_RJ(GRDR_data_pj)

    boresight_SIII_ch1 = get_boresight_dir(GRDR_data_pj=GRDR_data_pj, CH=1)
    boresight_SIII_ch2 = get_boresight_dir(GRDR_data_pj=GRDR_data_pj, CH=2)
    r_los1 = make_Rlos(Jn_SIII, boresight_SIII_ch1, smin=smin, smax=smax, sN=sN)
    r_los2 = make_Rlos(Jn_SIII, boresight_SIII_ch2, smin=smin, smax=smax, sN=sN)

    Ta_data = np.zeros(Jn_SIII.shape[0], 6)  # initialize array of brightness temps per channel
    for CH in CHS:
        if CH == CHS.CH1:
            r_los, r_b = r_los1, boresight_SIII_ch1
        else:
            r_los, r_b = r_los2, boresight_SIII_ch2
        Ta_data[CH] = los_integrate(r_los=r_los, r_b=r_b, MI=MI, E_range=E_range, s_range=s_range, CH=CH)

    time_data = GRDR_data_pj[['t_ephem_time', 't_utc_doy']].to_numpy()
    return pd.DataFrame(np.concatenate((time_data, Ta_data), axis=1), columns=['Time_ET', 'Time_UTC', f"Ch{ch}" for ch in CHS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--mesh-file", required=True, type=str, help="npz file containing mesh information")
    subparsers = parser.add_subparsers(dest="analysis", required=True)
    banana_parser = subparsers.add_parser("banana", help="Make banana plot")
    banana_argparser(banana_parser)
    mshell_parser = subparsers.add_parser("mshell", help="Make M-shell plot")
    mshell_argparser(mshell_parser)
    mshell_parser.set_defaults(mesh_file=argparse.SUPPRESS)
    args = parser.parse_args()

    chs = np.array([int(ch) for ch in args.ch.split(',')])
    for ch in chs: assert ch in range(1, 7), "Valid channel numbers are 1-6"    # validate channel inputs
    MI = MeshInterpolator(args.mesh_file)
    E_range = np.linspace(0.1, 50, 101)

    if args.analysis == "banana":
        _, GRDR_data_pj = load_PJ_data(args.PJ, args.dt, np.array(chs))
        IRDR_data_pj = imitate_IRDR(GRDR_data_pj=GRDR_data_pj, MI=MI, E_range=E_range, sN=50)
        banana_plot(IRDR_data_pj, GRDR_data_pj, args.dspr, f"Levin_PJs{args.PJ}_CHs{args.ch}_dt{args.dt}", args.type)

    elif args.analysis == "mshell":
        pass


if __name__ == "__main__":
    main()
