import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import kv
import scipy.integrate as integrate


def F_prim(x):
    return x * integrate.quad(lambda y: kv(5/3, y), x, np.inf)[0]

def F(x):
    return np.vectorize(F_prim)(x)

def F_p(x):
    return x * kv(2/3, x)


def main():
    eta = np.logspace(-2.3, 1.1, 300)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.axvline(x=1, linestyle=':', color='k', linewidth=0.7)
    ax.semilogx(eta, np.log(10) * eta * F(eta), linestyle='-', label=r'$F(\log x)$')
    ax.semilogx(eta, np.log(10) * eta * F_p(eta), linestyle='--', label=r'$F_p(\log x)$')

    ax.set_xlabel(r'$\nu/\nu_c$', fontsize=14)
    ax.legend(fontsize=14, loc='upper left')

    ax.set_xlim(eta[0], eta[-1])
    ax.set_ylim([0,1.6])

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yticks([0,1])

    fig.tight_layout()
    fig.savefig(f"frequency_response_curves.png", dpi=300)


if __name__ == '__main__':
    main()
