import numpy as np
from burgers1d_etdrk import initial_GRF, burgers1d_etdrk4
# from convergence import *

xa, xb = -np.pi, np.pi
nx = 1024
x = np.linspace(xa, xb, nx, endpoint=False)
interval = (xa, xb)
vis = 0.03
T = 2.0
dt = 1e-2

x, u0 = initial_GRF(N=nx//2, m=0, gamma=2.5, tau=7, sigma=7**2, bctype='periodic', interval=interval)
_, _, uu = burgers1d_etdrk4(u0, interval, dt, int(T/dt), vis, show_plot=True)
