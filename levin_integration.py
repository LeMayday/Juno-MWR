# modules
import numpy as np
from numpy.linalg import norm

# local files
from global_vars import ONE_D_NDArray, TWO_D_NDArray, THREE_D_NDArray
from mshell_map import alpha, alpha_eq, MeshInterpolator
from bessel_functions import F, F_p


E0 = 5000; EPS = 2.25   # eqn 11
MJ = 4.218              # [gauss RJ^3] eqn 9
C = 1.52E-3             # C (erg/s/Gauss) * c^2 (cm/s)^2 / k (erg/K) / nu^2 (GHz)^2
E_2PIMC = 2.79E-3       # GHz / Gauss
NU_O = np.array([0.6, 1.248, 2.597, 5.215, 10.004, 21.9])   # GHz
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
    Number density (assume cm^-3) --- eqns 8-11
    """
    a_eq = alpha_eq(alpha, Beq, B)                  # num samples x num LOS pts
    L = (MJ / Beq)**(1/3)                           # eqn 9 --- num samples x num LOS pts
    mp = table1_lookup(L)                           # model params
    n_ea = mp[..., 2] * np.sin(a_eq)**(mp[..., 0]) + mp[..., 3] * np.sin(a_eq)**(mp[..., 1])    # eqn 10 --- num samples x num LOS pts
    n_eE = E0 / ( E0 + ( E_range[None, None, :] / np.sqrt(B)[:, :, None] )**(EPS + 0.75) )      # eqn 11 --- num samples x num LOS pts x num E pts
    return n_ea[:, :, None] * n_eE


def I(B_Gauss: THREE_D_NDArray, cos2chi: TWO_D_NDArray, alpha: TWO_D_NDArray, n_e: THREE_D_NDArray, E_range: ONE_D_NDArray, s_range: ONE_D_NDArray) -> TWO_D_NDArray:
    sin_alpha = np.sin(alpha)
    nu_c = 3/2 * gamma(E_range)**2[None, None, :] * E_2PIMC * B_Gauss[:, :, None] * sin_alpha[:, :, None]   # num samples x num LOS pts x num E pts
    
    # num channels x num samples x num LOS pts x num E pts
    stokes_I_C = B_Gauss[None, :, :, None] * sin_alpha[None, :, :, None] *   F(NU_O[:, None, None, None] / nu_c[None, :, :, :]) * n_e[None, :, :, :]    # missing C*2pi
    stokes_Q_C = B_Gauss[None, :, :, None] * sin_alpha[None, :, :, None] * F_p(NU_O[:, None, None, None] / nu_c[None, :, :, :]) * n_e[None, :, :, :]    # missing C*2pi

    stokes_I_C_prime = 0.5 * (stokes_I_C + stokes_Q_C * cos2chi[None, :, :, None])

    stokes_I_C_prime = np.trapezoid(stokes_I_C_prime, E_range, axis=-1)     # integrate over E
    stokes_I_C_prime = np.trapezoid(stokes_I_C_prime, s_range, axis=-1)     # integrate over LOS --- output should be num channels x num samples
    return stokes_I_C_prime


def T(I_C: TWO_D_NDArray) -> TWO_D_NDArray:
    return C * I_C


def los_integrate(r_los: THREE_D_NDArray, r_b: TWO_D_NDArray, MI: MeshInterpolator, E_range: ONE_D_NDArray, s_range: ONE_D_NDArray) -> ONE_D_NDArray:
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

    I_C = I(B_Gauss=(B_los_norm * nT_to_Gauss), cos2chi=np.ones_like(B_los_norm), alpha=alpha_los, n_e=n_e, E_range=E_range, s_range=s_range)
    return T(I_C)


