from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
import burnman.minerals as minerals

def endmember_equilibrium_constant_P(P, endmember_A, endmember_B, T_guess):
    """
    For isochemical endmember calculations at fixed P or T,
    we have one unknown (T or P) and one equation
    (the gibbs free energies of the two phases must be equal).
    """
    def affinity(T):
        for phase in [endmember_A, endmember_B]:
            phase.set_state(P, T)
        return endmember_A.gibbs - endmember_B.gibbs

    sol = root(affinity, T_guess)
    return sol.x[0], sol.success

def endmember_equilibrium_constant_T(T, endmember_A, endmember_B, P_guess):
    """
    For isochemical endmember calculations at fixed P or T,
    we have one unknown (T or P) and one equation
    (the gibbs free energies of the two phases must be equal).
    """
    def affinity(P):
        for phase in [endmember_A, endmember_B]:
            phase.set_state(P, T)
        return endmember_A.gibbs - endmember_B.gibbs

    sol = root(affinity, P_guess)
    return sol.x[0], sol.success


def binary_equilibrium(P, T, phase_A, phase_B, guess):
    """
    For binary equilibria at fixed P and T, we have two unknowns
    (composition of A, composition of B) and two equations
    (the chemical potentials of Fe and Si must be the same in both phases)
    """
    def affinities(args):
        x_A, x_B = args
        for phase in [phase_A, phase_B]:
            phase.set_state(P, T)

        phase_A.set_composition([x_A, 1.-x_A])
        phase_B.set_composition([x_B, 1.-x_B])

        mu_A = phase_A.partial_gibbs
        mu_B = phase_B.partial_gibbs

        return np.array([mu_A[0] - mu_B[0],
                         mu_A[1] - mu_B[1]])

    sol = root(affinities, guess)
    return sol.x, sol.success

def ternary_equilibrium_constant_P(P, phase_A, phase_B, phase_C, guess):
    """
    For equilibria between three binary phases at fixed P or T,
    we have four unknowns (T or P, and the compositions of the three phases)
    and four corresponding equations
    (the chemical potentials of Fe and Si must be the same in all three phases).
    """
    def affinities(args):
        T, x_A, x_B, x_C = args
        for phase in [phase_A, phase_B, phase_C]:
            phase.set_state(P, T)

        phase_A.set_composition([x_A, 1.-x_A])
        phase_B.set_composition([x_B, 1.-x_B])
        phase_B.set_composition([x_C, 1.-x_C])

        mu_A = phase_A.partial_gibbs
        mu_B = phase_B.partial_gibbs
        mu_C = phase_C.partial_gibbs

        return np.array([mu_A[0] - mu_B[0],
                         mu_A[1] - mu_B[1],
                         mu_A[0] - mu_C[0],
                         mu_A[1] - mu_C[1]])

    sol = root(affinities, guess)
    return sol.x, sol.success
