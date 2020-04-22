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
        cargs = args
        mu = []
        for i, ph in enumerate([phase_A, phase_B]):
            # Set state and composition
            ph.set_state(P, T)
            ph.set_composition([cargs[i], 1.-cargs[i]])

            # Calculate the chemical potentials of Fe and Si
            if ph.name == 'B2-ordered bcc Fe-Si':  # Fe and Fe0.5Si0.5
                pgs = ph.partial_gibbs
                mu.append([pgs[0], 2.*pgs[1] - pgs[0]])
            elif (('fcc' in ph.name) or ('hcp' in ph.name) or ('liq' in ph.name)):
                mu.append(ph.partial_gibbs)
            else:
                raise Exception('Phase not recognised')

        return np.array([mu[0][0] - mu[1][0],
                         mu[0][1] - mu[1][1]])

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
        T = args[0]
        cargs = args[1:]
        mu = []
        for i, ph in enumerate([phase_A, phase_B, phase_C]):
            # Set state and composition
            ph.set_state(P, T)
            ph.set_composition([cargs[i], 1.-cargs[i]])

            # Calculate the chemical potentials of Fe and Si
            if ph.name == 'B2-ordered bcc Fe-Si':  # Fe and Fe0.5Si0.5
                pgs = ph.partial_gibbs
                mu.append([pgs[0], 2.*pgs[1] - pgs[0]])
            elif (('fcc' in ph.name) or ('hcp' in ph.name) or ('liq' in ph.name)):
                mu.append(ph.partial_gibbs)
            else:
                raise Exception('Phase not recognised')

        return np.array([mu[0][0] - mu[1][0],
                         mu[0][1] - mu[1][1],
                         mu[0][0] - mu[2][0],
                         mu[0][1] - mu[2][1]])

    sol = root(affinities, guess)
    return sol.x, sol.success
