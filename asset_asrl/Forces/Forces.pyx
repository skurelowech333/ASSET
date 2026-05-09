# asset_asrl/Forces/Forces.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

import asset_asrl as ast
cimport cython

vf = ast.VectorFunctions
Args = vf.Arguments


# =========================================================
# CONSTANTS (PURE C DOUBLES)
# =========================================================
cdef double mu_earth   = 3.986004418e14
cdef double Re         = 6378137.0

cdef double Cd         = 2.2
cdef double A          = 1.0
cdef double m          = 100.0

cdef double P0_srp     = 4.56e-6
cdef double Cr         = 1.2

cdef double J2         = 1.08262668e-3
cdef double J3         = -2.5327e-6

cdef double omega_earth = 7.2921159e-5


# =========================================================
# PREBUILT CONSTANT VECTORFUNCTIONS
# =========================================================
OMEGA_VEC = vf.stack(0.0, 0.0, omega_earth)

SRP_VEC = vf.stack(
    P0_srp * Cr * A / m,
    0.0,
    0.0
)


# =========================================================
# DENSITY CONSTANTS (NO NUMPY)
# =========================================================
cdef double rho0 = 1.225
cdef double rho1 = 3.899e-2
cdef double rho2 = 1.774e-4
cdef double rho3 = 3.972e-6
cdef double rho4 = 1.057e-7
cdef double rho5 = 3.206e-9
cdef double rho6 = 5.297e-10

cdef double H0 = 8500.0
cdef double H1 = 7000.0
cdef double H2 = 6000.0
cdef double H3 = 5500.0
cdef double H4 = 5000.0
cdef double H5 = 4500.0
cdef double H6 = 4000.0


# =========================================================
# GLOBAL CACHED VECTOR FUNCTIONS
# =========================================================
_GRAVITY_MODEL = None
_ADAPTIVE_GRAVITY_MODEL = None
_DRAG_MODEL = None
_SRP_MODEL = None

_BASIC_MODEL = None
_FULL_MODEL = None
_DRAG_ONLY_MODEL = None


# =========================================================
# DENSITY MODEL
# =========================================================
cdef density_model(r):

    cdef:
        object r2
        object rnorm
        object h

    r2 = r.dot(r) + 1.0e-12
    rnorm = vf.sqrt(r2)

    h = rnorm - Re

    # fully unrolled
    return (
        rho0 * vf.exp(-h / H0) +
        rho1 * vf.exp(-h / H1) +
        rho2 * vf.exp(-h / H2) +
        rho3 * vf.exp(-h / H3) +
        rho4 * vf.exp(-h / H4) +
        rho5 * vf.exp(-h / H5) +
        rho6 * vf.exp(-h / H6)
    )


# =========================================================
# CENTRAL GRAVITY
# =========================================================
def gravity_force():

    global _GRAVITY_MODEL

    if _GRAVITY_MODEL is not None:
        return _GRAVITY_MODEL

    cdef:
        object X
        object r
        object r2
        object inv_r
        object inv_r3
        object a

    X = Args(6)

    # avoid tolist()
    r = X.head3()

    r2 = r.dot(r) + 1.0e-12

    # cheaper graph than division
    inv_r = vf.pow(r2, -0.5)
    inv_r3 = inv_r * inv_r * inv_r

    a = (-mu_earth * inv_r3) * r

    _GRAVITY_MODEL = a

    return _GRAVITY_MODEL


# =========================================================
# ADAPTIVE J2/J3 GRAVITY
# =========================================================
def adaptive_gravity_force():

    global _ADAPTIVE_GRAVITY_MODEL

    if _ADAPTIVE_GRAVITY_MODEL is not None:
        return _ADAPTIVE_GRAVITY_MODEL

    cdef:
        object X
        object r

        object x
        object y
        object z

        object r2
        object rnorm

        object inv_r
        object inv_r2
        object inv_r3

        object a_central

        object h
        object tJ2
        object tJ3
        object wJ2
        object wJ3

        object z2

        object k2
        object k3

        object c1
        object c2

        object ax2
        object ay2
        object az2

        object ax3
        object ay3
        object az3

        object aJ2
        object aJ3

    X = Args(6)

    r = X.head3()

    x = r[0]
    y = r[1]
    z = r[2]

    r2 = r.dot(r) + 1.0e-12

    inv_r = vf.pow(r2, -0.5)
    inv_r2 = 1.0 / r2
    inv_r3 = inv_r * inv_r * inv_r

    rnorm = r2 * inv_r

    # -----------------------------------------------------
    # CENTRAL GRAVITY
    # -----------------------------------------------------
    a_central = (-mu_earth * inv_r3) * r

    # -----------------------------------------------------
    # ALTITUDE
    # -----------------------------------------------------
    h = rnorm - Re

    # -----------------------------------------------------
    # SMOOTH FADES
    # -----------------------------------------------------
    tJ2 = (h - 2.0e6) / 1.0e6
    tJ3 = (h - 8.0e5) / 4.0e5

    wJ2 = 1.0 - tJ2
    wJ3 = 1.0 - tJ3

    z2 = z * z

    # =====================================================
    # J2
    # =====================================================
    k2 = 1.5 * J2 * (Re * Re) * inv_r2

    c1 = k2 * (5.0 * z2 * inv_r2 - 1.0)
    c2 = k2 * (5.0 * z2 * inv_r2 - 3.0)

    ax2 = c1 * x
    ay2 = c1 * y
    az2 = c2 * z

    aJ2 = vf.stack(ax2, ay2, az2)

    # =====================================================
    # J3
    # =====================================================
    k3 = 0.5 * J3 * (Re * Re * Re) * inv_r2

    c1 = (5.0 * z * inv_r) * (7.0 * z2 * inv_r2 - 3.0)

    ax3 = k3 * x * c1
    ay3 = k3 * y * c1

    az3 = k3 * (
        ((2.0 * z2 - 3.0 * r2) * inv_r)
        + 5.0 * z * (3.0 - 7.0 * z2 * inv_r2)
    )

    aJ3 = vf.stack(ax3, ay3, az3)

    _ADAPTIVE_GRAVITY_MODEL = (
        a_central
        + wJ2 * aJ2
        + wJ3 * aJ3
    )

    return _ADAPTIVE_GRAVITY_MODEL


# =========================================================
# DRAG WITH ROTATING ATMOSPHERE
# =========================================================
def drag_rotation_force():

    global _DRAG_MODEL

    if _DRAG_MODEL is not None:
        return _DRAG_MODEL

    cdef:
        object X
        object r
        object v

        object rho

        object v_atm
        object v_rel
        object v_rel_norm

        object coeff
        object a

    X = Args(6)

    r = X.head3()
    v = X.segment3(3)

    rho = density_model(r)

    # rotating atmosphere
    v_atm = vf.cross(OMEGA_VEC, r)

    v_rel = v - v_atm

    v_rel_norm = v_rel.norm()

    coeff = -0.5 * Cd * A / m

    a = coeff * rho * v_rel_norm * v_rel

    _DRAG_MODEL = a

    return _DRAG_MODEL


# =========================================================
# SOLAR RADIATION PRESSURE
# =========================================================
def srp_force():

    global _SRP_MODEL

    if _SRP_MODEL is not None:
        return _SRP_MODEL

    _SRP_MODEL = SRP_VEC

    return _SRP_MODEL


# =========================================================
# BASIC MODEL
# =========================================================
def build_basic_model():

    global _BASIC_MODEL

    if _BASIC_MODEL is not None:
        return _BASIC_MODEL

    _BASIC_MODEL = gravity_force()

    return _BASIC_MODEL


# =========================================================
# DRAG MODEL
# =========================================================
def build_drag_model():

    global _DRAG_ONLY_MODEL

    if _DRAG_ONLY_MODEL is not None:
        return _DRAG_ONLY_MODEL

    _DRAG_ONLY_MODEL = (
        gravity_force()
        + drag_rotation_force()
    )

    return _DRAG_ONLY_MODEL


# =========================================================
# FULL MODEL
# =========================================================
def build_full_model():

    global _FULL_MODEL

    if _FULL_MODEL is not None:
        return _FULL_MODEL

    _FULL_MODEL = (
        adaptive_gravity_force()
        + drag_rotation_force()
        + srp_force()
    )

    return _FULL_MODEL