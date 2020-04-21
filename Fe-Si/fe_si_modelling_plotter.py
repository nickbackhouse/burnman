#my attempts this morning, both times I haven't got past calculating the invariant point


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

from fe_si_phases import endmembers, solutions
liq = solutions['liq_fe_si']
fcc = solutions['fcc_fe_si']
hcp = solutions['hcp_fe_si']
B2 = solutions['B2_fe_si']

from endmember_and_binary_equilibrium_functions import *


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


def invariant_point(phase_A, phase_B, phase_C, P_guess, T_guess):
    def affinity(args):
        P, T = args
        for phase in [phase_A, phase_B, phase_C]:
            phase.set_state(P, T)
        return [phase_A.gibbs - phase_B.gibbs,
                phase_A.gibbs - phase_C.gibbs]

    sol = root(affinity, [P_guess, T_guess])
    return sol.x, sol.success


#attempt 1

#find the invariant point

P_inv, T_inv = ternary_equilibrium_constant_P(100.e9, fcc, hcp, liq, 3000.)[0]

# Create pressure arrays above and below the invariant point
low_ps = np.linspace(0, P_inv, 101)
high_ps = np.linspace(P_inv, 330.e9, 101)

# Find the transition temperatures for each transition
fcc_liq_temperatures = [binary_equilibrium(p, fcc, liq, T_guess=2000.)[0] for p in low_ps]
fcc_hcp_temperatures = [binary_equilibrium(p, fcc, hcp, T_guess=2000.)[0] for p in low_ps]
hcp_liq_temperatures = [binary_equilibrium(p, hcp, liq, T_guess=2000.)[0] for p in high_ps]

# Plot the transition pressures
plt.plot(low_ps/1.e9, fcc_liq_temperatures, color='blue', label='FCC-liq')
plt.plot(low_ps/1.e9, fcc_hcp_temperatures, color='blue',linestyle='dotted', label='FCC-HCP')
plt.plot(high_ps/1.e9, hcp_liq_temperatures, color='blue',linestyle='dashed', label='HCP-liq')


#I've updated the parameters based on an inversion, but when I try and plot them the ternary equilibrium function
#throws up ValueError: not enough values to unpack (expected 4, got 1). Ternary equilibrium constant is the wrong 
#function as this is a binary system



#attempt 2: added the invariant_point funtion

#find the invariant point

P_inv, T_inv = invariant_point(fcc, hcp, liq, 100.e9, 3000.)[0]

# Create pressure arrays above and below the invariant point
low_ps = np.linspace(0, P_inv, 101)
high_ps = np.linspace(P_inv, 330.e9, 101)

# Find the transition temperatures for each transition
fcc_liq_temperatures = [binary_equilibrium(p, fcc, liq, T_guess=2000.)[0] for p in low_ps]
fcc_hcp_temperatures = [binary_equilibrium(p, fcc, hcp, T_guess=2000.)[0] for p in low_ps]
hcp_liq_temperatures = [binary_equilibrium(p, hcp, liq, T_guess=2000.)[0] for p in high_ps]

# Plot the transition pressures
plt.plot(low_ps/1.e9, fcc_liq_temperatures, color='blue', label='FCC-liq')
plt.plot(low_ps/1.e9, fcc_hcp_temperatures, color='blue',linestyle='dotted', label='FCC-HCP')
plt.plot(high_ps/1.e9, hcp_liq_temperatures, color='blue',linestyle='dashed', label='HCP-liq')

#AttributeError: 'SolidSolution' object has no attribute 'molar_fractions'. I assume I've misubderstood how to calculate the invariant point
