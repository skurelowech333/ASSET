# -*- coding: utf-8 -*-

import numpy as np
import time
import asset_asrl as ast
import matplotlib.pyplot as plt

plt.style.use("dark_background")

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


# =========================================================
# IMPORT CYTHON GRAVITY EXTENSION
# =========================================================
from asset_asrl.Forces.Gravity import GravityVF as GravityVF_CY


# =========================================================
# CONSTANTS
# =========================================================
mu_earth = 3.986004418e14
Re = 6378137.0


# =========================================================
# PYTHON GRAVITY
# =========================================================
def gravity_python(r):
    return -mu_earth * r.normalized_power3()


# =========================================================
# CYTHON GRAVITY OBJECT
# =========================================================
grav_cy = GravityVF_CY(mu_earth)


# =========================================================
# ODE BASELINE (PYTHON)
# =========================================================
class PythonGravityODE(oc.ODEBase):

    def __init__(self):

        X = Args(6)
        r = X.head3()
        v = X.segment3(3)

        ode = vf.stack(v, gravity_python(r))

        super().__init__(ode, 6, 0)


# =========================================================
# ODE CYTHON
# =========================================================
class CythonGravityODE(oc.ODEBase):

    def __init__(self):

        X = Args(6)
        v = X.segment3(3)

        ode = vf.stack(v, grav_cy.vf())

        super().__init__(ode, 6, 0)


# =========================================================
# INITIAL CONDITION
# =========================================================
def circular_orbit_ic():

    r0 = Re + 200e3
    v0 = np.sqrt(mu_earth / r0)

    return np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])


def orbital_period(r0):
    return 2*np.pi*np.sqrt(r0**3 / mu_earth)


# =========================================================
# TIMING UTILITY
# =========================================================
def time_func(func, runs=20):

    out = []

    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        out.append(t1 - t0)

    return np.array(out)


# =========================================================
# BENCHMARK BUILD
# =========================================================
def benchmark_build():

    py = time_func(lambda: PythonGravityODE())
    cy = time_func(lambda: CythonGravityODE())

    return py, cy


# =========================================================
# PROPAGATION BENCHMARK
# =========================================================
def benchmark_prop(ode_class, Xt0, tf, runs=5):

    times = []
    steps = []

    for _ in range(runs):

        ode = ode_class()
        integ = ode.integrator("DOPRI87", 20)

        t0 = time.perf_counter()
        sol = integ.integrate_dense(Xt0, tf)
        t1 = time.perf_counter()

        traj = np.array(sol)

        times.append(t1 - t0)
        steps.append(len(traj))

    return np.array(times), np.array(steps)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # INITIAL STATE
    X0 = circular_orbit_ic()
    Xt0 = np.hstack([X0, 0.0])

    r0 = np.linalg.norm(X0[:3])
    tf = 3 * orbital_period(r0)

    # WARMUP
    print("\nWARMUP...")
    _ = PythonGravityODE().integrator("DOPRI87", 20).integrate_dense(Xt0, tf)
    _ = CythonGravityODE().integrator("DOPRI87", 20).integrate_dense(Xt0, tf)

    # =====================================================
    # BUILD BENCHMARK
    # =====================================================
    print("\n==============================")
    print("ODE BUILD")
    print("==============================")

    py_b, cy_b = benchmark_build()

    print(f"Python mean  : {np.mean(py_b)*1e6:.2f} us")
    print(f"Cython mean  : {np.mean(cy_b)*1e6:.2f} us")
    print("Speedup:", np.mean(py_b)/np.mean(cy_b))


    # =====================================================
    # PROPAGATION BENCHMARK
    # =====================================================
    print("\n==============================")
    print("PROPAGATION")
    print("==============================")

    py_t, py_s = benchmark_prop(PythonGravityODE, Xt0, tf)
    cy_t, cy_s = benchmark_prop(CythonGravityODE, Xt0, tf)

    print(f"Python mean time : {np.mean(py_t):.6f} s")
    print(f"Cython mean time : {np.mean(cy_t):.6f} s")
    print("Speedup:", np.mean(py_t)/np.mean(cy_t))


    # =====================================================
    # TRAJECTORY CHECK
    # =====================================================
    py_sol = PythonGravityODE().integrator("DOPRI87", 20).integrate_dense(Xt0, tf)
    cy_sol = CythonGravityODE().integrator("DOPRI87", 20).integrate_dense(Xt0, tf)

    py_traj = np.array(py_sol)
    cy_traj = np.array(cy_sol)

    n = min(len(py_traj), len(cy_traj))
    py_traj = py_traj[:n]
    cy_traj = cy_traj[:n]

    r_err = np.linalg.norm(py_traj[:, :3] - cy_traj[:, :3], axis=1)

    print("\nMax position error:", np.max(r_err))
    print("Mean position error:", np.mean(r_err))


    # =====================================================
    # PLOT
    # =====================================================
    t = np.linspace(0, tf, n)

    plt.figure()
    plt.plot(t/3600, r_err)
    plt.yscale("log")
    plt.title("Position Error: Python vs Cython Gravity")
    plt.xlabel("Time (hr)")
    plt.ylabel("Error (m)")
    plt.grid()
    plt.show()