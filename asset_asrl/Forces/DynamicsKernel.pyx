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
# GRAVITY FORCE
# =========================================================
cdef class GravityVF:

    cdef double mu
    cdef object _vf

    def __cinit__(self, double mu):
        self.mu = mu
        self._vf = self._build()

    cdef object _build(self):

        cdef object X = Args(6)
        cdef object r = X.head3()

        return (-self.mu) * r.normalized_power3()

    cpdef object vf(self):
        return self._vf


# =========================================================
# DRAG FORCE (simple quadratic model)
# =========================================================
cdef class DragVF:

    cdef double Cd
    cdef double A
    cdef double rho
    cdef object _vf

    def __cinit__(self, double Cd, double A, double rho):

        self.Cd = Cd
        self.A = A
        self.rho = rho

        self._vf = self._build()

    cdef object _build(self):

        cdef object X = Args(6)
        cdef object r = X.head3()
        cdef object v = X.segment3(3)

        cdef object v2 = v.dot(v)

        # drag direction: -v normalized scaled
        cdef object drag_dir = -v.normalized()

        cdef object mag = 0.5 * self.rho * self.Cd * self.A * v2

        return mag * drag_dir

    cpdef object vf(self):
        return self._vf


# =========================================================
# COMBINED DYNAMICS BUILDER (KEY SPEEDUP POINT)
# =========================================================
cdef class DynamicsModel:

    cdef object grav
    cdef object drag
    cdef object _vf

    def __cinit__(self, GravityVF grav, DragVF drag):

        self.grav = grav
        self.drag = drag

        self._vf = self._build()

    cdef object _build(self):

        cdef object X = Args(6)
        cdef object v = X.segment3(3)

        # -------------------------------------------------
        # IMPORTANT: SUM ONLY VECTOR FUNCTIONS
        # -------------------------------------------------
        cdef object acc = (
            self.grav.vf() +
            self.drag.vf()
        )

        return vf.stack(v, acc)

    cpdef object vf(self):
        return self._vf