
import numpy as np
from scipy.fft import fft, ifft
from burgers1d_etdrk import burgers1d_etdrk4


def initial_GRF(N, m, gamma, tau, sigma, bctype, interval=(0.0, 1.0)):
    a, b = interval
    L = b - a 
    M = N * 2
    
    const = 2 * np.pi / L

    k = np.arange(1, N + 1)
    eigs = np.sqrt(2) * sigma * ((const * k) ** 2 + tau ** 2) ** (-gamma / 2)

    alpha = eigs * np.random.randn(N)

    beta = eigs * np.random.randn(N)

    a_k = alpha / 2
    b_k = -beta / 2

    c = np.concatenate([
        np.flip(a_k) - 1j * np.flip(b_k),
        np.array([m + 0j]),
        a_k + 1j * b_k
    ])

    u = np.fft.ifft(np.fft.ifftshift(c), n=M).real * len(c)

    return u


def linear(k):
    return 0.01 * k ** 2

def nonlinear(k, uhat):
    return - 0.5 * k * fft(np.real(ifft(uhat)) ** 2)

# from convergence import *

xa, xb = -np.pi, np.pi
nx = 1024
x = np.linspace(xa, xb, nx, endpoint=False)
interval = (xa, xb)
vis = 0.03
T = 2.0
dt = 1e-2

u0 = initial_GRF(N=nx//2, m=0, gamma=2.5, tau=7, sigma=7**2, bctype='periodic', interval=interval)
_, _, uu = burgers1d_etdrk4(u0, interval, dt, int(T/dt), linear, nonlinear, show_plot=True)
