#small program to plot iron phase diagram
#the first section is copied straight from fe_si_intro_for_nick.py
#this funtion should generate two plots, one with just the iron phase diagram the other with the Fe-Si curve

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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
import numpy as np
%matplotlib inline

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

def invariant_point(endmember_A, endmember_B, endmember_C, P_guess, T_guess):
    """
    For isochemical endmember calculations at fixed P or T,
    we have one unknown (T or P) and one equation
    (the gibbs free energies of the two phases must be equal).
    """
    def affinity(args):
        P, T = args
        for phase in [endmember_A, endmember_B, endmember_C]:
            phase.set_state(P, T)
        return [endmember_A.gibbs - endmember_B.gibbs,
                endmember_A.gibbs - endmember_C.gibbs]

    sol = root(affinity, [P_guess, T_guess])
    return sol.x, sol.success

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

print(endmember_equilibrium_constant_P(100.e9,fcc_iron, liq_iron, 1800))

#this is where my code begins, I thought the best way to approach it was to create two arrays across the
#PT range of the phase changes and iterate over them in parallel to create an array of temperature values.

#first section is for the melting curve, I assume that if I'm going to extrapolate this to higher pressures
#I'd need to calculate again for hcp-liq transition


# Find the invariant point
P_inv, T_inv = invariant_point(fcc_iron, hcp_iron, liq_iron, 100.e9, 3000.)[0]

# Create pressure arrays above and below the invariant point
low_ps = np.linspace(0, P_inv, 101)
high_ps = np.linspace(P_inv, 200.e9, 101)

# Find the transition temperatures for each transition
fcc_liq_temperatures = [endmember_equilibrium_constant_P(p, fcc_iron, liq_iron, T_guess=2000.)[0] for p in low_ps]
fcc_hcp_temperatures = [endmember_equilibrium_constant_P(p, fcc_iron, hcp_iron, T_guess=2000.)[0] for p in low_ps]
hcp_liq_temperatures = [endmember_equilibrium_constant_P(p, hcp_iron, liq_iron, T_guess=2000.)[0] for p in high_ps]

# Plot the transition pressures
plt.plot(low_ps/1.e9, fcc_liq_temperatures, color='blue', label='FCC-liq')
plt.plot(low_ps/1.e9, fcc_hcp_temperatures, color='blue',linestyle='dotted', label='FCC-HCP')
plt.plot(high_ps/1.e9, hcp_liq_temperatures, color='blue',linestyle='dashed', label='HCP-liq')


#this section just loads in my eutectic data
print('put data files in your repository data directory and use relative paths so that this also works on other machines')
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

#plotting my data
print('use plt.errorbar and plt.scatter to plot your data points')

Tm0 = 1473
def simonglatzel(P,A,C):
    Tm = ((P/A)+1.)**(1./C)*Tm0
    return Tm
popt, pcov = curve_fit(simonglatzel, p, t)
x = np.linspace(0,100,100)
plt.plot(x,simonglatzel(x,popt[0],popt[1]), color = 'red')

#with the pure iron data

x = np.linspace(0,100,101)
plt.plot(x,r,color='blue')

ps2 = np.linspace(0,80,101)
plt.plot(ps2,s,color='blue',linestyle='dotted')

#adding legend and stuff and saving it to my files, obviously you need to change the save location

plt.legend()
plt.ylabel('Temperature (K)')
plt.xlabel('Pressure (GPa)')


print('save to a relative path somewhere in your repository (maybe a figures directory)')
print('you can also change the file format by appending the correct suffix to the path. pdfs are best for me, but if your report is in word, jpg might be best')
plt.savefig('/Users/nicholasbackhouse/Documents/fesivsfe', dpi = 300)
