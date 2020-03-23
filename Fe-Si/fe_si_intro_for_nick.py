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


# Here we define our minerals and free parameters
# The following iron endmembers are from the Sundman and Eriksson (2015) paper
fcc_iron = minerals.SE_2015.fcc_iron()
hcp_iron = minerals.SE_2015.hcp_iron()
bcc_iron = minerals.SE_2015.bcc_iron()
liq_iron = minerals.SE_2015.liquid_iron()

# The silicon solid endmember parameters are from a variety of sources
# (see ../burnman/minerals/Fe_Si.py)
fcc_silicon = minerals.Fe_Si.Si_fcc_A1()
bcc_silicon = minerals.Fe_Si.Si_bcc_A2()
hcp_silicon = minerals.Fe_Si.Si_hcp_A3()

"""
liq_silicon = ?
"""

# Here we define our solid solution parameters
bcc = burnman.SolidSolution(name = 'disordered bcc Fe-Si',
                          solution_type = 'symmetric',
                          endmembers = [[bcc_iron, '[Fe]'],
                                        [bcc_silicon, '[Si]']],
                          energy_interaction = [[-100.0e3]],
                          entropy_interaction = [[0.0e3]],
                          volume_interaction = [[0.0e3]])


fcc = burnman.SolidSolution(name = 'disordered fcc Fe-Si',
                          solution_type = 'symmetric',
                          endmembers = [[fcc_iron,    '[Fe]'],
                                        [fcc_silicon, '[Si]']],
                          energy_interaction = [[-100.0e3]],
                          entropy_interaction = [[0.0e3]],
                          volume_interaction = [[0.0e3]])

hcp = burnman.SolidSolution(name = 'disordered hcp Fe-Si',
                          solution_type = 'symmetric',
                          endmembers = [[hcp_iron,    '[Fe]'],
                                        [hcp_silicon, '[Si]']],
                          energy_interaction = [[-100.0e3]],
                          entropy_interaction = [[0.0e3]],
                          volume_interaction = [[0.0e3]])

"""
liq = burnman.SolidSolution(name = 'disordered liq Fe-Si',
                          solution_type = 'symmetric',
                          endmembers = [[liq_iron,    '[Fe]'],
                                        [liq_silicon, '[Si]']],
                          energy_interaction = [[10.0e3]],
                          entropy_interaction = [[0.0e3]],
                          volume_interaction = [[0.0e3]])
"""


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


# examples of the eqm functions, for fcc_iron and hcp_iron equilibria
print(endmember_equilibrium_constant_P(P = 80.e9,
                                       endmember_A = fcc_iron,
                                       endmember_B = hcp_iron,
                                       T_guess = 2000.))

print(endmember_equilibrium_constant_T(T = 2750.,
                                      endmember_A = fcc_iron,
                                      endmember_B = hcp_iron,
                                      P_guess = 80.e9))


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

# The next few lines plot the gibbs free energy of the hcp and fcc phases
# (both simple mixing of Fe and Si)
T = 3000.
for P in [80.e9, 100.e9]:
    xs = np.linspace(0., 1., 101)
    g_hcp = []
    g_fcc = []
    for x in xs:
        hcp.set_composition([1.-x, x])
        fcc.set_composition([1.-x, x])
        hcp.set_state(P, T)
        fcc.set_state(P, T)
        g_hcp.append(hcp.gibbs)
        g_fcc.append(fcc.gibbs)
    plt.plot(xs, g_hcp, label='hcp')
    plt.plot(xs, g_fcc, label='fcc')
plt.legend()
plt.show()


# Finally, we plot the binary phase loop
# (or rather, the part of it which is stable)
temperatures = np.linspace(1000., 3500., 101)
P = 80.e9
Ts = []
x_As = []
x_Bs = []

for T in temperatures:
    guess = [0.5, 0.5]
    sol, success = binary_equilibrium(P=P, T=T, phase_A=fcc, phase_B=hcp, guess=guess)
    if success:
        Ts.append(T)
        x_As.append(sol[0])
        x_Bs.append(sol[1])

plt.plot(x_As, Ts)
plt.plot(x_Bs, Ts)
plt.show()
