# =========================================================
# FILE: Gravity.pyx  (Option A - VF Factory Pattern)
# =========================================================
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import asset_asrl as ast
cimport cython

vf = ast.VectorFunctions
Args = vf.Arguments


# =========================================================
# GRAVITY VECTOR FUNCTION WRAPPER
# (BUILT ONCE, USED MANY TIMES)
# =========================================================
cdef class GravityVF:

    cdef:
        double mu
        object _vf   # cached ASSET VectorFunction

    # -----------------------------------------------------
    # CONSTRUCTOR (BUILD ONCE HERE)
    # -----------------------------------------------------
    def __cinit__(self, double mu):

        self.mu = mu

        # Build ASSET vector function once
        self._vf = self._build()

    # -----------------------------------------------------
    # BUILD VECTOR FUNCTION (CALLED ONLY ONCE)
    # -----------------------------------------------------
    cdef object _build(self):

        cdef object X
        cdef object r
        cdef object r2
        cdef object acc

        X = Args(6)

        r = X.head3()

        # -----------------------------
        # r norm squared
        # -----------------------------
        r2 = r.dot(r) + 1.0e-12

        # -----------------------------
        # central gravity (fully ASSET-native)
        # -----------------------------
        acc = (-self.mu) * r.normalized_power3()

        return acc

    # -----------------------------------------------------
    # RETURN CACHED VECTOR FUNCTION
    # -----------------------------------------------------
    cpdef object vf(self):
        return self._vf

    # -----------------------------------------------------
    # CALLABLE INTERFACE
    # -----------------------------------------------------
    def __call__(self):
        return self._vf

    # -----------------------------------------------------
    # DEBUG PRINT
    # -----------------------------------------------------
    def __repr__(self):
        return f"GravityVF(mu={self.mu})"