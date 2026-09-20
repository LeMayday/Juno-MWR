# modules
import numpy as np
import JupiterMag as jm
jm.Con2020.Config(equation_type='analytic')

TWO_D_NDArray = np.ndarray[tuple[int, int], np.dtype[np.float32]]


def B(X, Y, Z):
    jm.Internal.Config(Model="jrm33", CartesianIn=True, CartesianOut=True)
    Bx_int, By_int, Bz_int = jm.Internal.Field(X, Y, Z)
    Bx_ext, By_ext, Bz_ext = jm.Con2020.Field(X, Y, Z)
    return Bx_int + Bx_ext, By_int + By_ext, Bz_int + Bz_ext


def trace_batch_mshell(r: TWO_D_NDArray):
    T = jm.TraceField(*r.T, IntModel='jrm33', ExtModel='Con2020')
    mshells = np.array(T.equator.mshell, dtype=np.float32)
    return mshells


def find_lats_M(phi_vec, M, tol=1e-4):
    theta = np.arcsin(np.sqrt(1/M))                                 # r/R = 1 = M cos^2(lat)
    theta_vec = np.full_like(phi_vec, theta)

    n = 0
    prev_res = np.zeros_like(phi_vec)
    damping = np.ones_like(phi_vec)
    while True:
        x0 = np.cos(phi_vec) * np.sin(theta_vec)
        y0 = np.sin(phi_vec) * np.sin(theta_vec)
        z0 = np.cos(theta_vec)
        M_trace = jm.TraceField(x0, y0, z0, Verbose=True, IntModel='jrm33', ExtModel='Con2020').equator.mshell
        res = M_trace - M
        if np.max(np.abs(res)) < tol:
            print(f"Newton-Raphson completed in {n} iterations.")
            break
        overshoot = (prev_res * res) < 0
        damping[overshoot] *= 0.5                                   # fixes oscillations near magnetic great red spot
        step = res * np.tan(theta_vec) / (2 * M) * damping          # dM/dtheta
        theta_vec += np.clip(step, -0.1, 0.1)
        prev_res = np.copy(res)
        n += 1

    return theta_vec


def pre_compute_mshell_traces(M: float, ntraces: int = 100) -> jm.TraceField:
    jm.Con2020.Config(equation_type='analytic')
    phi = np.linspace(0, 2*np.pi, ntraces, endpoint=False)
    theta = find_lats_M(phi, M)
    x0 = np.cos(phi) * np.sin(theta)
    y0 = np.sin(phi) * np.sin(theta)
    z0 = np.cos(theta)
    # see https://github.com/mattkjames7/JupiterMag/blob/a3fc24f20e0860296a11a55ee14f0e5f5e8fc577/JupiterMag/TraceField.py#L16 for args
    return jm.TraceField(x0, y0, z0, Verbose=False, IntModel='jrm33', ExtModel='Con2020', MaxStep=0.1)
