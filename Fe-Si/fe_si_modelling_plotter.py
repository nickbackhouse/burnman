#!/usr/bin/env python
# coding: utf-8

# In[27]:


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
from scipy.optimize import curve_fit

import burnman
import burnman.minerals as minerals

from fe_si_phases import endmembers, solutions
from endmember_and_binary_equilibrium_functions import *

liq = solutions['liq_fe_si']
fcc = solutions['fcc_fe_si']
hcp = solutions['hcp_fe_si']
B2 = solutions['B2_fe_si']

bcc_iron = endmembers['bcc_iron']
fcc_iron = endmembers['fcc_iron']
hcp_iron = endmembers['hcp_iron']
liq_iron = endmembers['liq_iron']
B2_FeSi = endmembers['B2_FeSi']


# In[28]:


P = 80.e9

B2_FeSi = endmembers['B2_FeSi']
liq.set_composition([0.5, 0.5])

iron_melting_T = endmember_equilibrium_constant_P(P, fcc_iron, liq_iron, 2000.)[0]
FeSi_melting_T = endmember_equilibrium_constant_P(P, B2_FeSi, liq, 3000.)[0]

guess1 = [3000,0.8,0.8,0.64]
TX_inv, success = ternary_equilibrium_constant_P(P=P, phase_A=fcc, phase_B=liq, phase_C=B2, guess=guess1)

T_inv = TX_inv[0]
XA_inv = TX_inv[1]
XB_inv = TX_inv[2]
XC_inv = TX_inv[3]


temperatures = np.linspace(T_inv, FeSi_melting_T, 101)
temperatures1 = np.linspace(T_inv, iron_melting_T, 101)
Ts = []
T1s = []
x_As = []
x_Bs = []
x_Cs = []
x_Ds = []
x_B2 = []

for T in temperatures:
    guess = [XB_inv, XC_inv]
    sol, success = binary_equilibrium(P=P, T=T, phase_A=liq, phase_B=B2, guess=guess)
    if success:
        Ts.append(T)
        x_As.append(liq.formula['Si'])
        x_Bs.append(B2.formula['Si'])


for T1 in temperatures1:
    guess = [XA_inv, XB_inv]
    sol1, success = binary_equilibrium(P=P, T=T1, phase_A=fcc, phase_B=liq, guess=guess)
    if success:
        T1s.append(T1)
        x_Cs.append(fcc.formula['Si'])
        x_Ds.append(liq.formula['Si'])


# In[29]:


P2 = 120e9

B2_FeSi = endmembers['B2_FeSi']
liq.set_composition([0.5, 0.5])

iron_melting_T = endmember_equilibrium_constant_P(P2, hcp_iron, liq_iron, 4000.)[0]
FeSi_melting_T = endmember_equilibrium_constant_P(P2, B2_FeSi, liq, 5000.)[0]

guess2 = [3000,0.8,0.8,0.64]
TX2_inv, success = ternary_equilibrium_constant_P(P=P2, phase_A=hcp, phase_B=liq, phase_C=B2, guess=guess2)

T2_inv = TX2_inv[0]
XA2_inv = TX2_inv[1]
XB2_inv = TX2_inv[2]
XC2_inv = TX2_inv[3]


temperatures3 = np.linspace(T2_inv, FeSi_melting_T, 101)
temperatures4 = np.linspace(T2_inv, iron_melting_T, 101)
T2s = []
T2 = []
T3s = []
x_A1s = []
x_B1s = []
x_C1s = []
x_D1s = []
x_B3 = []
T4s = []

for T3 in temperatures3:
    guess3 = [XB2_inv, XC2_inv]
    sol2, success2 = binary_equilibrium(P=P2, T=T3, phase_A=liq, phase_B=B2, guess=guess3)
    if success:
        T2s.append(T2)
        x_A1s.append(liq.formula['Si'])
        x_B1s.append(B2.formula['Si'])


for T4 in temperatures4:
    guess4 = [XA2_inv, XB2_inv]
    sol3, success3 = binary_equilibrium(P=P2, T=T4, phase_A=hcp, phase_B=liq, guess=guess4)
    if success:
        T4s.append(T4)
        x_C1s.append(hcp.formula['Si'])
        x_D1s.append(liq.formula['Si'])


# In[30]:


P3 = 50.e9

B2_FeSi = endmembers['B2_FeSi']
liq.set_composition([0.5, 0.5])

iron_melting_T = endmember_equilibrium_constant_P(P, fcc_iron, liq_iron, 2000.)[0]
FeSi_melting_T = endmember_equilibrium_constant_P(P, B2_FeSi, liq, 3000.)[0]

guess3 = [3000,0.8,0.8,0.64]
TX3_inv, success = ternary_equilibrium_constant_P(P=P3, phase_A=fcc, phase_B=liq, phase_C=B2, guess=guess3)

T3_inv = TX3_inv[0]
XA3_inv = TX3_inv[1]
XB3_inv = TX3_inv[2]
XC3_inv = TX3_inv[3]


temperatures5 = np.linspace(T3_inv, FeSi_melting_T, 101)
temperatures6 = np.linspace(T3_inv, iron_melting_T, 101)

T5s = []
x_A2s = []
x_B2s = []
x_C2s = []
x_D2s = []
x_B22 = []
T6s = []


for T5 in temperatures5:
    guess5 = [XB3_inv, XC3_inv]
    sol5, success = binary_equilibrium(P=P3, T=T5, phase_A=liq, phase_B=B2, guess=guess5)
    if success:
        T5s.append(T5)
        x_As.append(liq.formula['Si'])
        x_Bs.append(B2.formula['Si'])


for T6 in temperatures6:
    guess6 = [XA3_inv, XB3_inv]
    sol6, success = binary_equilibrium(P=P3, T=T6, phase_A=fcc, phase_B=liq, guess=guess6)
    if success:
        T6s.append(T6)
        x_C2s.append(fcc.formula['Si'])
        x_D2s.append(liq.formula['Si'])


# In[34]:


plt.ylabel('Temperature (K)')
plt.xlabel('Si content (mol%)')
#plt.plot(x_Cs, T1s,color='red',linewidth=0.5)
#plt.plot(x_Ds, T1s,color='red',linewidth=0.5)
#plt.plot(x_C2s,T6s,color='green')
#plt.plot(x_D2s,T6s,color='green')

plt.plot(x_C1s,T4s,color='blue')
plt.plot(x_D1s,T4s,color='blue')
#plt.scatter(1-XA_inv,T_inv,color='red')
#plt.scatter(1-XB_inv,T_inv,color='blue')
#plt.scatter(1-XC_inv,T_inv,color='green')

#plt.xlim(0, 0.20)
#plt.ylim(2700, 3500)
plt.show()


# In[35]:


pressures = np.linspace(30e9, 90e9, 101)
guess2 = [3000, 0.86, 0.82, 0.62]

T_inv = []
XA_inv = []
XB_inv = []
XC_inv = []

assemblage = [fcc, B2, liq]

for P in pressures:
    TX_inv, success = ternary_equilibrium_constant_P(P=P,
                                                     phase_A=assemblage[0],
                                                     phase_B=assemblage[1],
                                                     phase_C=assemblage[2],
                                                     guess=guess2)
    
    if success:
        guess = [assemblage[0].temperature,
                 assemblage[0].molar_fractions[0],
                 assemblage[1].molar_fractions[0],
                 assemblage[2].molar_fractions[0]]

        T_inv.append(TX_inv[0])
        XA_inv.append(assemblage[0].formula['Si'])
        XB_inv.append(assemblage[1].formula['Si'])
        XC_inv.append(assemblage[2].formula['Si'])


pressures2 = np.linspace(90e9, 300e9, 101)
guess2 = [5000, 0.93, 0.75, 0.89]

T2_inv = []
XA2_inv = []
XB2_inv = []
XC2_inv = []
pressure2new = []
assemblage2 = [hcp, B2, liq]

for P in pressures2:
    TX2_inv, success2 = ternary_equilibrium_constant_P(P=P,
                                                     phase_A=assemblage2[0],
                                                     phase_B=assemblage2[1],
                                                     phase_C=assemblage2[2],
                                                     guess=guess2)
    if success2:
        guess2 = [assemblage2[0].temperature,
                 assemblage2[0].molar_fractions[0],
                 assemblage2[1].molar_fractions[0],
                 assemblage2[2].molar_fractions[0]]

        T2_inv.append(TX2_inv[0])
        XA2_inv.append(assemblage2[0].formula['Si'])
        XB2_inv.append(assemblage2[1].formula['Si'])
        XC2_inv.append(assemblage2[2].formula['Si'])
        pressure2new.append(P)
    else:
        print(P, ' is a failure')


# In[36]:


plt.plot(XC_inv,T_inv,color='blue')
plt.plot(XC2_inv,T2_inv,color='red')
plt.show()
plt.plot(XC_inv, pressures)
plt.plot(XC2_inv,pressure2new)
plt.show()
plt.plot(pressures, T_inv)


# In[37]:


file = open("/Users/nicholasbackhouse/Documents/MATLAB.nosync/Data Files/msciexp.txt")
pressure = []
temperature = []
temperatureerror = []


for i in file:
    row = i.split(",")
    if len(row) == 3:
        pressure.append(row[1])
        temperature.append(row[0])
        temperatureerror.append(row[2].strip("\n"))
file.close()
pressure = pressure[1:]
temperature = temperature[1:]
temperatureerror = temperatureerror[1:]

p = [float(i) for i in pressure]
t = [float(i) for i in temperature]
t_err = [float(i) for i in temperatureerror]


# In[40]:


Tm0 = 1473
plt.scatter(p,t,color='red',label='Eutectic Data - This Study')
def simonglatzel(P,A,C):
    Tm = ((P/A)+1.)**(1./C)*Tm0
    return Tm
popt, pcov = curve_fit(simonglatzel, p, t)
x = np.linspace(0,100,100)
plt.plot(x,simonglatzel(x,popt[0],popt[1]), color = 'red', label='Fe-Si eutectic (This study)')

plt.plot(pressures/1e9, T_inv,label='Modelled Eutectic')
#plt.plot(pressures2/1e9,TX2_inv)
plt.legend()



# In[ ]:




