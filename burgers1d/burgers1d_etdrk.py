import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def setup_fft(xa, xb, nx):
    L = xb - xa
    h = L / nx
    x = np.linspace(xa, xb - h, nx)
    k = fftfreq(nx, d=L / nx) * 2 * np.pi
    return x, k

def compute_etdrk4_coefficients(L, dt, m=64):
    r = np.exp(2j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    LR = dt * L[:, None] + r[None, :]
    Q1 = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    Q2 = dt * np.real(np.mean((np.exp(LR      ) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))
    return Q1, Q2, f1, f2, f3

def burgers1d_etdrk2(u0, interval, dt, nt, vis, plot_interval=10, show_plot=True):
    xa, xb = interval
    nx = u0.shape[0]
    x, k = setup_fft(xa, xb, nx)
    
    L = -vis * k**2
    E = np.exp(dt * L)

    m = 64
    r = np.exp(2j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    LR = dt * L[:, None] + r[None, :]
    f1 = dt * np.real(np.mean((np.exp(LR) - 1) / LR, axis=1))
    f2 = dt * np.real(np.mean((np.exp(LR) - 1 - LR) / LR**2, axis=1))

    g = -0.5j * k
    v = fft(u0)
    
    tt = [0.0]
    uu = [u0.copy()]
    
    for i in range(nt):
        t = (i + 1) * dt
        Nv = g * fft(np.real(ifft(v)) ** 2)
        a = E * v + f1 * Nv
        Na = g * fft(np.real(ifft(a)) ** 2)
        v = a + f2 * (Na - Nv)
        
        if (i + 1) % plot_interval == 0:
            u = np.real(ifft(v))
            if show_plot:
                plt.clf()
                plt.plot(x, u, linewidth=1.5)
                plt.ylim(-1, 1)
                plt.title(f"ETDRK2 - Time: {t:.2f}")
                plt.pause(0.01)
            uu.append(u.copy())
            tt.append(t)
    
    return x, np.array(tt), np.array(uu)

def burgers1d_etdrk3(u0, interval, dt, nt, vis, plot_interval=10, show_plot=True):
    xa, xb = interval
    nx = u0.shape[0]
    x, k = setup_fft(xa, xb, nx)
    
    L = -vis * k**2
    E  = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)
    Q1, Q2, f1, f2, f3 = compute_etdrk4_coefficients(L, dt)

    g = -0.5j * k
    v = fft(u0)
    
    tt = [0.0]
    uu = [u0.copy()]
    
    for i in range(nt):
        t = (i + 1) * dt

        Nv = g * fft(np.real(ifft(v)) ** 2)
        a = E2 * v + Q1 * Nv
        Na = g * fft(np.real(ifft(a)) ** 2)
        b = E * v + Q2 * (2 * Na - Nv)
        Nb = g * fft(np.real(ifft(b)) ** 2)

        v = E * v + Nv * f1 + 4.0 * Na * f2 + Nb * f3
        
        if (i + 1) % plot_interval == 0:
            u = np.real(ifft(v))
            if show_plot:
                plt.clf()
                plt.plot(x, u, linewidth=1.5)
                plt.ylim(-1, 1)
                plt.title(f"ETDRK4 - Time: {t:.2f}")
                plt.pause(0.01)
            uu.append(u.copy())
            tt.append(t)
    
    return x, np.array(tt), np.array(uu)

def burgers1d_etdrk4(u0, interval, dt, nt, vis, plot_interval=10, show_plot=True):
    xa, xb = interval
    nx = u0.shape[0]
    x, k = setup_fft(xa, xb, nx)
    
    L = -vis * k**2
    E  = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)
    Q, Q2, f1, f2, f3 = compute_etdrk4_coefficients(L, dt)

    g = -0.5j * k
    v = fft(u0)
    
    tt = [0.0]
    uu = [u0.copy()]
    
    for i in range(nt):
        t = (i + 1) * dt

        Nv = g * fft(np.real(ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * fft(np.real(ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * fft(np.real(ifft(b)) ** 2)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = g * fft(np.real(ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        
        if (i + 1) % plot_interval == 0:
            u = np.real(ifft(v))
            if show_plot:
                plt.clf()
                plt.plot(x, u, linewidth=1.5)
                plt.ylim(-1, 1)
                plt.title(f"ETDRK4 - Time: {t:.2f}")
                plt.pause(0.01)
            uu.append(u.copy())
            tt.append(t)
    
    return x, np.array(tt), np.array(uu)
