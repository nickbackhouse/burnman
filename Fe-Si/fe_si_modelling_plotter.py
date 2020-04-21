


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
from fe_si_experimental_equilibria import *


def invariant_point(P, phase_A, phase_B, phase_C, guess):
    """
    for the invariant point in a binary system we have to output three compositions and the temperature
    """
    def affinities(args):
        T, x_A, x_B, x_C = args
        for phase in [phase_A, phase_B, phase_C]:
            phase.set_state(P, T)

        phase_A.set_composition([x_A, 1.-x_A])
        phase_B.set_composition([x_B, 1.-x_B])
        phase_C.set_composition([x_C, 1.-x_C])

        mu_A = phase_A.partial_gibbs
        mu_B = phase_B.partial_gibbs
        mu_C = phase_C.partial_gibbs

        return np.array([mu_A[0] - mu_B[0],
                         mu_A[1] - mu_B[1],
                         mu_A[0] - mu_C[0],
                         mu_A[1] - mu_C[1]])


    sol = root(affinities, guess)
    return sol.x, sol.success




#I've managed to make it output three compositions and a temperature (the function I now realise is exactly the same as the ternary_equilibrium function but nevermind!) but I dont believe the temperature its outputting (590K)

guess1 = [3000,0.25,0.25,0.5]
TX_inv, success = invariant_point(P=50.e9, phase_A=fcc, phase_B=hcp, phase_C=liq, guess=guess1)

T_inv = TX_inv[0]
XA_inv = TX_inv[1]
XB_inv = TX_inv[2]
XC_inv = TX_inv[3]


temperatures = np.linspace(1000, 5000., 101)
temperatures1 = np.linspace(1000., 5000., 101)
P = 50.e9
Ts = []
T1s = []
x_As = []
x_Bs = []
x_Cs = []
x_Ds = []

for T in temperatures:
    guess = [0.5, 0.5]
    sol, success = binary_equilibrium(P=P, T=T, phase_A=liq, phase_B=hcp, guess=guess)
    if success:
        Ts.append(T)
        x_As.append(sol[0])
        x_Bs.append(sol[1])
        
       
        
for T1 in temperatures1:
    guess = [0.5, 0.5]
    sol1, success = binary_equilibrium(P=P, T=T1, phase_A=fcc, phase_B=liq, guess=guess)
    if success:
        T1s.append(T1)
        x_Cs.append(sol1[0])
        x_Ds.append(sol1[1])        

plt.plot(x_As, Ts)
plt.plot(x_Bs, Ts)
plt.plot(x_Cs, T1s)
plt.plot(x_Ds, T1s)
plt.show()
