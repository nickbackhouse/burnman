# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../../burnman'):
    sys.path.insert(1, os.path.abspath('../..'))

import burnman
from burnman import Mineral, minerals
import matplotlib.image as mpimg
from scipy.optimize import fsolve, root


fcc = minerals.SE_2015.fcc_iron()
bcc = minerals.SE_2015.bcc_iron()
hcp = minerals.SE_2015.hcp_iron()
liq = minerals.SE_2015.liquid_iron()

calib = [[bcc, 1.e5, 1., 69538., 7.0496],
         [bcc, 1.e5, 300., -8183., 7.0947],
         [bcc, 1.e9, 1., 76567., 7.0094],
         [bcc, 1.e5, 1000., -42271., 7.3119],
         [bcc, 1.e9, 1000., -34987., 7.2583],
         [fcc, 3.e9, 1000., -20604., 7.0250],
         [fcc, 10.e9, 1000., 27419., 6.7139],
         [hcp, 40.e9, 1000., 214601., 5.8583]]

print('P, T, Diffs: Gibbs (J/mol), V (fractional)')
for (m, P, T, G, V) in calib:
    m.set_state(P, T)
    print('{0} GPa, {1} K: {2}, {3}'.format(P/1.e9, T, m.gibbs-G, (m.V*1.e6 - V)/V))

Fe_diag_img = mpimg.imread('figures/iron_phase_diagram_SE2015_perplex.jpg') # from SE15ver.dat bundled with PerpleX 6.8.3 (September 2018)
#Fe_diag_img = mpimg.imread('figures/fe_brosh.png') # alternative, from Brosh paper
plt.imshow(Fe_diag_img, extent=[0.0, 350.0, 1, 6400], alpha=0.3, aspect='auto')

def invariant(m1, m2, m3, P_guess=5.e9, T_guess=2000.):

    assemblage = burnman.Composite([m1, m2, m3])

    def diff_gibbs(args):
        P, T = args
        assemblage.set_state(P, T)
        return np.array([m1.gibbs - m2.gibbs,
                         m2.gibbs - m3.gibbs])

    sol = root(diff_gibbs, [P_guess, T_guess])
    return sol.x[0:2]

def univariant(m1, m2, condition_constraints, P_guess=5.e9, T_guess=2000.):
    assemblage = burnman.Composite([m1, m2])

    def diff_gibbs_P(T, P):
        assemblage.set_state(P, T)
        return m1.gibbs - m2.gibbs

    def diff_gibbs_T(P, T):
        assemblage.set_state(P, T)
        return m1.gibbs - m2.gibbs

    pressures = []
    temperatures = []

    if condition_constraints[0] == 'P':
        for P in condition_constraints[1]:
            sol = root(diff_gibbs_P, [T_guess], args=(P))
            temperatures.append(sol.x[0])
            T_guess = sol.x[0]
            pressures.append(P)

    if condition_constraints[0] == 'T':
        for T in condition_constraints[1]:
            sol = root(diff_gibbs_T, [P_guess], args=(T))
            pressures.append(sol.x[0])
            P_guess = sol.x[0]
            temperatures.append(T)


    pressures = np.array(pressures)
    temperatures = np.array(temperatures)
    return pressures, temperatures


print(univariant(fcc, liq, ('P', np.array([21.e9])),
                 P_guess=21.e9, T_guess=2500.))

Tmin = 1.
Pmin = 1.e5
Pmax = 350.e9

# BCC-FCC-LIQ invariant
Pinv, Tinv = invariant(bcc, fcc, liq, P_guess=5.e9, T_guess=2000.)

pressures, temperatures =  univariant(bcc, liq, ('P', np.linspace(Pmin, Pinv, 11)), P_guess=Pmin, T_guess=1800.)
plt.plot(pressures/1.e9, temperatures)

pressures, temperatures =  univariant(bcc, fcc, ('P', np.linspace(Pmin, Pinv, 11)), P_guess=Pmin, T_guess=1800.)
plt.plot(pressures/1.e9, temperatures, label='bcc-fcc')


# FCC-HCP-LIQ invariance
Pinv2, Tinv2 = invariant(fcc, hcp, liq, P_guess=90.e9, T_guess=3000.)

pressures, temperatures =  univariant(fcc, liq, ('P', np.linspace(Pinv, Pinv2, 11)), P_guess=Pinv, T_guess=Tinv)
plt.plot(pressures/1.e9, temperatures, label='fcc-liq')

pressures, temperatures =  univariant(hcp, liq, ('P', np.linspace(Pinv2, Pmax, 11)), P_guess=Pinv2, T_guess=Tinv2)
plt.plot(pressures/1.e9, temperatures, label='hcp-liq')



# BCC-FCC-HCP invariance
Pinv3, Tinv3 = invariant(bcc, fcc, hcp, P_guess=10.e9, T_guess=1000.)

pressures, temperatures =  univariant(bcc, hcp, ('T', np.linspace(Tmin, Tinv3, 11)), P_guess=10.e9, T_guess=Tmin)
plt.plot(pressures/1.e9, temperatures, label='bcc-hcp')

pressures, temperatures =  univariant(bcc, fcc, ('P', np.linspace(Pmin, Pinv3, 11)), P_guess=Pmin, T_guess=Tinv3)
plt.plot(pressures/1.e9, temperatures, label='bcc-fcc')

pressures, temperatures =  univariant(fcc, hcp, ('P', np.linspace(Pinv3, Pinv2, 11)), P_guess=Pinv3, T_guess=Tinv3)
plt.plot(pressures/1.e9, temperatures, label='fcc-hcp')


plt.scatter([21.], [2516], label='Saxena and Eriksson, 2015, Fig2')

plt.xlim(0., 350.)
plt.ylim(0., 8000.)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.show()
