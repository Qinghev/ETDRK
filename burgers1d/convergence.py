import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import norm

def run_etdrk_solver(etdrk_solver, dt_values, u0, interval, nt_base, vis, reference_u):
    errors = []
    for dt in dt_values:
        nt = int(nt_base * dt_values[0] / dt)
        x, tt, uu = etdrk_solver(u0, interval, dt, nt, vis, show_plot=False)
        assert abs(tt[-1] - 1.) < 1e-12
        u_final = uu[-1]
        error = norm(u_final - reference_u, ord=2) / np.sqrt(len(u_final))  # L2 norm
        errors.append(error)
    return errors

def plot_convergence(dt_values, errors, method_name):
    plt.figure()
    plt.loglog(dt_values, errors, 'o-', label=method_name)
    plt.xlabel("Time step size (dt)")
    plt.ylabel("L2 error at final time")
    plt.grid(True, which="both")
    plt.title(f"Time Convergence of {method_name}")
    
    # 拟合斜率
    coeffs = np.polyfit(np.log(dt_values), np.log(errors), 1)
    rate = coeffs[0]
    plt.legend([f"{method_name} (order ≈ {abs(rate):.2f})"])
    plt.show()
    return rate

# 初始条件
def initial_condition(x):
    return -np.sin(np.pi * x)

xa, xb = 0.0, 2.0
nx = 2048  # 较细空间网格以减少空间误差
x = np.linspace(xa, xb, nx, endpoint=False)
u0 = initial_condition(x)

interval = (xa, xb)
vis = 0.01
T = 1.0  # 模拟时间
dt_values = np.array([0.1, 0.05, 0.025, 0.0125])  # 时间步长
nt_base = int(T / dt_values[0])

# 使用 ETDRK4 最小步长生成参考解
from copy import deepcopy
_, _, uu_ref = burgers1d_etdrk4(deepcopy(u0), interval, dt_values[-1]/4, int(T / dt_values[-1]*4), vis, show_plot=False)
reference_u = uu_ref[-1]


rate2 = plot_convergence(
    dt_values,
    run_etdrk_solver(burgers1d_etdrk2, dt_values, u0, interval, nt_base, vis, reference_u),
    "ETDRK2"
)

rate3 = plot_convergence(
    dt_values,
    run_etdrk_solver(burgers1d_etdrk3, dt_values, u0, interval, nt_base, vis, reference_u),
    "ETDRK3"
)

rate4 = plot_convergence(
    dt_values,
    run_etdrk_solver(burgers1d_etdrk4, dt_values, u0, interval, nt_base, vis, reference_u),
    "ETDRK4"
)

print(f"Estimated order: ETDRK2 ≈ {rate2:.2f}, ETDRK3 ≈ {rate3:.2f}, ETDRK4 ≈ {rate4:.2f}")
