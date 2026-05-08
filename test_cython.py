# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import asset_asrl as ast

plt.style.use("dark_background")

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

# =========================================================
# IMPORT CYTHON FORCE MODEL
# =========================================================
from asset_asrl.Forces.Forces import build_basic_model

# =========================================================
# CONSTANTS (PYTHON BASELINE)
# =========================================================
mu_earth = 3.986004418e14
Re = 6378137.0


# =========================================================
# PYTHON BASELINE FORCE
# =========================================================
def gravity_accel(r):
    return -mu_earth * r.normalized_power3()


class PythonBasic(oc.ODEBase):
    def __init__(self):
        X = Args(6)
        r, v = X.tolist([(0,3),(3,3)])
        ode = vf.stack(v, gravity_accel(r))
        super().__init__(ode, 6, 0)


# =========================================================
# CYTHON MODEL
# =========================================================
class CythonBasic(oc.ODEBase):
    def __init__(self):
        X = Args(6)
        r, v = X.tolist([(0,3),(3,3)])
        ode = vf.stack(v, build_basic_model())
        super().__init__(ode, 6, 0)


# =========================================================
# INITIAL CONDITION
# =========================================================
def circular_orbit_ic():
    r0 = Re + 200e3
    v0 = np.sqrt(mu_earth / r0)
    return np.array([r0, 0, 0, 0, v0, 0])


def orbital_period(r0):
    return 2*np.pi*np.sqrt(r0**3 / mu_earth)


# =========================================================
# SPEED BENCHMARK FUNCTION
# =========================================================
def benchmark(name, ode, Xt0, tf, runs=3):

    times = []
    steps = []

    for _ in range(runs):

        integ = ode.integrator("DOPRI87", 20)

        start = time.perf_counter()
        sol = integ.integrate_dense(Xt0, tf)
        end = time.perf_counter()

        traj = np.array(sol)

        times.append(end - start)
        steps.append(len(traj))

    return {
        "name": name,
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "avg_steps": np.mean(steps)
    }


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # -----------------------------
    # INITIAL STATE
    # -----------------------------
    X0 = circular_orbit_ic()
    Xt0 = np.hstack([X0, 0.0])

    r0 = np.linalg.norm(X0[:3])
    tf = 3 * orbital_period(r0)

    # -----------------------------
    # MODELS
    # -----------------------------
    models = {
        "Python Basic": PythonBasic(),
        "Cython Basic": CythonBasic()
    }

    results = {}

    # =========================================================
    # PROPAGATION (for accuracy tests)
    # =========================================================
    for name, ode in models.items():

        integ = ode.integrator("DOPRI87", 20)
        sol = integ.integrate_dense(Xt0, tf)
        traj = np.array(sol)

        results[name] = {
            "r": traj[:, 0:3],
            "v": traj[:, 3:6],
            "t": np.linspace(0, tf, len(traj))
        }

    # =========================================================
    # SPEED BENCHMARK
    # =========================================================
    print("\n==============================")
    print("SPEED BENCHMARK")
    print("==============================\n")

    speed_results = []

    for name, ode in models.items():
        res = benchmark(name, ode, Xt0, tf)
        speed_results.append(res)

    for r in speed_results:
        print(f"{r['name']}")
        print(f"  Avg time  : {r['avg_time']:.6f} s")
        print(f"  Std time  : {r['std_time']:.6f} s")
        print(f"  Avg steps : {r['avg_steps']:.1f}\n")

    speedup = speed_results[0]["avg_time"] / speed_results[1]["avg_time"]
    print(f"🚀 SPEEDUP (Python / Cython): {speedup:.2f}x\n")


    # =========================================================
    # TRAJECTORY PLOT
    # =========================================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for name, data in results.items():
        r = data["r"]
        ax.plot(r[:,0], r[:,1], r[:,2], label=name)

    ax.set_title("Orbit Comparison")
    ax.legend()
    plt.show()


    # =========================================================
    # ERROR ANALYSIS
    # =========================================================
    truth = results["Python Basic"]["r"]

    plt.figure()

    for name, data in results.items():
        if name == "Python Basic":
            continue
        err = np.linalg.norm(data["r"] - truth, axis=1)
        plt.plot(data["t"]/3600, err, label=name)

    plt.yscale("log")
    plt.title("Cython vs Python Error")
    plt.xlabel("Time (hours)")
    plt.ylabel("Position Error (m)")
    plt.grid()
    plt.legend()
    plt.show()


    # =========================================================
    # ENERGY CHECK
    # =========================================================
    def specific_energy(r, v):
        rnorm = np.linalg.norm(r, axis=1)
        v2 = np.sum(v*v, axis=1)
        return 0.5*v2 - mu_earth/rnorm

    plt.figure()

    for name, data in results.items():
        e = specific_energy(data["r"], data["v"])
        plt.plot(data["t"]/3600, e, label=name)

    plt.title("Specific Energy Drift")
    plt.xlabel("Time (hours)")
    plt.ylabel("Energy (J/kg)")
    plt.grid()
    plt.legend()
    plt.show()


    # =========================================================
    # NUMERICAL EQUALITY CHECK
    # =========================================================
    import numpy as np

    RTOL = 1e-8
    ATOL = 1e-6

    r_py = results["Python Basic"]["r"]
    r_cy = results["Cython Basic"]["r"]

    v_py = results["Python Basic"]["v"]
    v_cy = results["Cython Basic"]["v"]

    n = min(len(r_py), len(r_cy))

    r_py, r_cy = r_py[:n], r_cy[:n]
    v_py, v_cy = v_py[:n], v_cy[:n]

    r_err = np.linalg.norm(r_py - r_cy, axis=1)
    v_err = np.linalg.norm(v_py - v_cy, axis=1)

    print("\n==============================")
    print("NUMERICAL COMPARISON")
    print("==============================")

    print("Max position error:", np.max(r_err))
    print("Mean position error:", np.mean(r_err))
    print("Max velocity error:", np.max(v_err))

    pos_pass = np.allclose(r_py, r_cy, rtol=RTOL, atol=ATOL)
    vel_pass = np.allclose(v_py, v_cy, rtol=RTOL, atol=ATOL)

    print("\nPASS/FAIL")
    print("Position:", "PASS" if pos_pass else "FAIL")
    print("Velocity:", "PASS" if vel_pass else "FAIL")

    worst = np.argmax(r_err)

    print("\nWorst step:", worst)
    print("Error (m):", r_err[worst])
    print("Time (hr):", results["Python Basic"]["t"][worst] / 3600)