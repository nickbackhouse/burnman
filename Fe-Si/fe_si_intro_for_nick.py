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

# Here we define our minerals and free parameters
# The following iron endmembers are from the Sundman and Eriksson (2015) paper
fcc_iron = endmembers['fcc_iron']
hcp_iron = endmembers['hcp_iron']
bcc_iron = endmembers['bcc_iron']
liq_iron = endmembers['liq_iron']

liq = solutions['liq_fe_si']
fcc = solutions['fcc_fe_si']
hcp = solutions['hcp_fe_si']
B2 = solutions['B2_fe_si']

from endmember_and_binary_equilibrium_functions import *


# examples of the eqm functions, for fcc_iron and hcp_iron equilibria
print(endmember_equilibrium_constant_P(P = 80.e9,
                                       endmember_A = fcc_iron,
                                       endmember_B = hcp_iron,
                                       T_guess = 2000.))

print(endmember_equilibrium_constant_T(T = 2750.,
                                      endmember_A = fcc_iron,
                                      endmember_B = hcp_iron,
                                      P_guess = 80.e9))


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
