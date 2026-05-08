# asset_asrl/Forces/Forces.pyx
# cython: language_level=3

import numpy as np
import asset_asrl as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


# =========================================================
# CONSTANTS
# =========================================================
cdef double mu_earth = 3.986004418e14
cdef double Re = 6378137.0

cdef double Cd = 2.2
cdef double A  = 1.0
cdef double m  = 100.0

cdef double P0_srp = 4.56e-6
cdef double Cr = 1.2

cdef double J2 = 1.08262668e-3
cdef double J3 = -2.5327e-6

cdef double omega_earth = 7.2921159e-5


# =========================================================
# DENSITY TABLES
# =========================================================
rho_layers = np.array([
    1.225,
    3.899e-2,
    1.774e-4,
    3.972e-6,
    1.057e-7,
    3.206e-9,
    5.297e-10
])

H_layers = np.array([
    8500.0,
    7000.0,
    6000.0,
    5500.0,
    5000.0,
    4500.0,
    4000.0
])


# =========================================================
# BASE FORCE WRAPPER
# =========================================================
cdef class ForceVF:

    cdef object expr

    def __init__(self, expr):
        self.expr = expr

    cpdef object get(self):
        return self.expr


# =========================================================
# FORCE COMPOSER
# =========================================================
cdef class ForceModelVF:

    cdef list forces

    def __init__(self):
        self.forces = []

    def add(self, ForceVF f):
        self.forces.append(f)

    cpdef object build(self):

        cdef object expr = None
        cdef ForceVF f

        for f in self.forces:

            if expr is None:
                expr = f.get()

            else:
                expr = expr + f.get()

        return expr


# =========================================================
# CENTRAL GRAVITY
# =========================================================
def gravity_force():

    X = Args(6)
    r, v = X.tolist([(0,3),(3,3)])

    a = -mu_earth * r.normalized_power3()

    return ForceVF(a)


# =========================================================
# J2/J3 GRAVITY
# =========================================================
def adaptive_gravity_force():

    X = Args(6)

    r, v = X.tolist([(0,3),(3,3)])

    x = r[0]
    y = r[1]
    z = r[2]

    r2 = r.dot(r)
    rnorm = vf.sqrt(r2)

    # central gravity
    a_central = -mu_earth * r.normalized_power3()

    # altitude
    h = rnorm - Re

    # smooth fade
    tJ2 = (h - 2.0e6) / 1.0e6
    tJ3 = (h - 8.0e5) / 4.0e5

    wJ2 = 1.0 - tJ2
    wJ3 = 1.0 - tJ3

    z2 = z*z

    # -----------------------------
    # J2
    # -----------------------------
    k2 = 1.5 * J2 * (Re**2) / r2

    aJ2 = vf.stack(
        k2 * (5*z2/r2 - 1) * x,
        k2 * (5*z2/r2 - 1) * y,
        k2 * (5*z2/r2 - 3) * z
    )

    # -----------------------------
    # J3
    # -----------------------------
    k3 = 0.5 * J3 * (Re**3) / r2

    aJ3 = vf.stack(
        k3 * x * (5*z/rnorm) * (7*z2/r2 - 3),

        k3 * y * (5*z/rnorm) * (7*z2/r2 - 3),

        k3 * (
            ((2*z2 - 3*r2)/rnorm)
            + 5*z*(3 - 7*z2/r2)
        )
    )

    return ForceVF(
        a_central + wJ2*aJ2 + wJ3*aJ3
    )


# =========================================================
# ATMOSPHERIC DENSITY
# =========================================================
def density_model(r):

    rnorm = r.norm()

    h = rnorm - Re

    rho = 0.0

    cdef int i

    for i in range(len(rho_layers)):

        rho += rho_layers[i] * vf.exp(
            -h / H_layers[i]
        )

    return rho


# =========================================================
# DRAG
# =========================================================
def drag_force():

    X = Args(6)

    r, v = X.tolist([(0,3),(3,3)])

    rho = density_model(r)

    vnorm = v.norm()

    a = -0.5 * Cd * A / m * rho * vnorm * v

    return ForceVF(a)


# =========================================================
# DRAG WITH ROTATING ATMOSPHERE
# =========================================================
def drag_rotation_force():

    X = Args(6)

    r, v = X.tolist([(0,3),(3,3)])

    rho = density_model(r)

    omega_vec = vf.stack(
        0.0,
        0.0,
        omega_earth
    )

    v_atm = vf.cross(omega_vec, r)

    v_rel = v - v_atm

    v_rel_norm = v_rel.norm()

    a = (
        -0.5
        * Cd
        * A
        / m
        * rho
        * v_rel_norm
        * v_rel
    )

    return ForceVF(a)


# =========================================================
# SOLAR RADIATION PRESSURE
# =========================================================
def srp_force():

    X = Args(1)

    zero = X[0] * 0.0

    sx = 1.0
    sy = 0.0
    sz = 0.0

    scale = P0_srp * Cr * A / m

    return ForceVF(

        vf.stack(

            zero + scale*sx,
            zero + scale*sy,
            zero + scale*sz
        )
    )


# =========================================================
# FULL COMPOSITE MODEL
# =========================================================
def build_full_model():

    model = ForceModelVF()

    model.add(adaptive_gravity_force())

    model.add(drag_rotation_force())

    model.add(srp_force())

    return model.build()


# =========================================================
# BASIC MODEL
# =========================================================
def build_basic_model():

    model = ForceModelVF()

    model.add(gravity_force())

    return model.build()


# =========================================================
# DRAG MODEL
# =========================================================
def build_drag_model():

    model = ForceModelVF()

    model.add(gravity_force())

    model.add(drag_force())

    return model.build()