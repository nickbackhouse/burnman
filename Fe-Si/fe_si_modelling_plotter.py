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





P = 50.e9
guess1 = [3000,0.25,0.25,0.5]
TX_inv, success = ternary_equilibrium_constant_P(P=P, phase_A=fcc, phase_B=B2, phase_C=liq, guess=guess1)

T_inv = TX_inv[0]
XA_inv = TX_inv[1]
XB_inv = TX_inv[2]
XC_inv = TX_inv[3]


temperatures = np.linspace(1000, 5000., 101)
temperatures1 = np.linspace(1000., 5000., 101)
Ts = []
T1s = []
x_As = []
x_Bs = []
x_Cs = []
x_Ds = []

for T in temperatures:
    guess = [0.5, 0.5]
    sol, success = binary_equilibrium(P=P, T=T, phase_A=B2, phase_B=liq, guess=guess)
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

plt.plot(x_As, Ts,color='blue')
plt.plot(x_Bs, Ts,color='blue')
plt.plot(x_Cs, T1s,color='red')
plt.plot(x_Ds, T1s,color='red')
plt.show()

plt.scatter(XA_inv,T_inv)
plt.scatter(XB_inv,T_inv)
plt.scatter(XC_inv,T_inv)

print(XA_inv)
print(XB_inv)
print(XC_inv)
print(T_inv)

#trying to plot the eutectic in T-X space, I get assertion errors

pressures = np.linspace(1, 330e9, 101)
guess1 = [3000,0.25,0.25,0.5]

T_inv = []
XA_inv = []
XB_inv = []
XC_inv = []


for P in pressures:
    TX_inv, success = ternary_equilibrium_constant_P(P=P, phase_A=fcc, phase_B=B2, phase_C=liq, guess=guess1)
    if success:
        T_inv.append(TX_inv[0])
        XA_inv.append(TX_inv[1])
        XB_inv.append(TX_inv[2])
        XC_inv.append(TX_inv[3])

