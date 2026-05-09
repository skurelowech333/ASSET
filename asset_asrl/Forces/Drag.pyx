# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:46:28 2026

@author: Sarah
"""

# =========================================================
# FILE: Drag.pyx
# ASSET VectorFunction Drag Model (Cython wrapper)
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
# DRAG FORCE MODEL
# =========================================================
cdef class DragVF:

    cdef double Cd
    cdef double A
    cdef double rho

    cdef object _vf   # cached ASSET VectorFunction

    # -----------------------------------------------------
    # CONSTRUCTOR (BUILD ONCE)
    # -----------------------------------------------------
    def __cinit__(self, double Cd, double A, double rho):

        self.Cd = Cd
        self.A = A
        self.rho = rho

        self._vf = self._build()


    # -----------------------------------------------------
    # BUILD VECTOR FUNCTION (ASSET GRAPH)
    # -----------------------------------------------------
    cdef object _build(self):

        cdef object X
        cdef object v
        cdef object speed
        cdef object drag

        X = Args(6)

        v = X.segment3(3)

        # speed = ||v||
        speed = v.norm()

        # quadratic drag:
        # F = -0.5 * rho * Cd * A * |v| * v
        drag = -0.5 * self.rho * self.Cd * self.A * speed * v

        return drag


    # -----------------------------------------------------
    # RETURN CACHED VF
    # -----------------------------------------------------
    cpdef object vf(self):
        return self._vf


    # -----------------------------------------------------
    # CALLABLE INTERFACE
    # -----------------------------------------------------
    def __call__(self):
        return self._vf


    # -----------------------------------------------------
    # DEBUG
    # -----------------------------------------------------
    def __repr__(self):
        return f"DragForce(Cd={self.Cd}, A={self.A}, rho={self.rho})"