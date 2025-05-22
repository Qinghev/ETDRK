# ETDRK PDE Solvers

This repository contains modular implementations of Exponential Time-Differencing Runge–Kutta (ETDRK) schemes (orders 2–4) for solving nonlinear PDEs, including:

- [x] Burgers’ Equation
- [ ] Korteweg–de Vries (KdV)
- [ ] Allen–Cahn Equation
- [ ] FitzHugh–Nagumo (FN) Model

## 🧠 Background

We consider nonlinear PDEs of the form:

$$
\frac{\partial u}{\partial t} = \mathcal{L} u + \mathcal{N}(u)
$$

where $L$ is the linear part and $N$ the nonlinear part.

We implement ETDRK2/3/4 methods as described by Cox & Matthews (2002) and Kassam & Trefethen (2005).


## 🧪 Example: Burgers’ Equation

Run the time convergence test:

```bash
cd burgers
python run_convergence.py
