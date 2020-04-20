# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from burnman import Mineral, minerals
import matplotlib.image as mpimg
from scipy.optimize import fsolve, root

from fe_si_phases import solutions

R = 8.31446


def Gex_liq_Lacaze_Sundman(T, x_Si):
    return Hex_liq_Lacaze_Sundman(x_Si) - T*Sex_liq_Lacaze_Sundman(x_Si)

def Hex_liq_Lacaze_Sundman(x_Si):
    L0 = -164434.6
    L1 = 0.
    L2 = -18821.542
    L3 = 9695.8
    x_Fe = 1. - x_Si
    return x_Fe*x_Si*(L0 + (x_Fe - x_Si)*L1 + (x_Fe - x_Si)**2*L2 + (x_Fe - x_Si)**3*L3)

def Sex_liq_Lacaze_Sundman(x_Si):
    return Sex_conf_liq_Lacaze_Sundman(x_Si) + Sex_nonconf_liq_Lacaze_Sundman(x_Si)

def Sex_conf_liq_Lacaze_Sundman(x_Si):
    x_Fe = 1. - x_Si
    return -R*(x_Si*np.log(x_Si) + x_Fe*np.log(x_Fe))

def Sex_nonconf_liq_Lacaze_Sundman(x_Si):
    L0 = -41.9773
    L1 = 21.523
    L2 = -22.07
    L3 = 0.
    x_Fe = 1. - x_Si

    return x_Fe*x_Si*(L0 + (x_Fe - x_Si)*L1 + (x_Fe - x_Si)**2*L2 + (x_Fe - x_Si)**3*L3)



T = 1873.
liq = solutions['liq_fe_si']
liq.set_state(1.e5, T)

fig = plt.figure(figsize=(15, 5))
ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)]
x_Si = np.linspace(1.e-5, 1.-1.e-5, 101)
Hex = np.empty_like(x_Si)
Gex = np.empty_like(x_Si)
Sex = np.empty_like(x_Si)
loggammaSi = np.empty_like(x_Si)

for i, x in enumerate(x_Si):
    liq.set_composition([1.-x, x])
    Hex[i] = liq.excess_enthalpy
    Sex[i] = liq.excess_entropy
    Gex[i] = liq.excess_gibbs
    loggammaSi[i] = np.log10(liq.activity_coefficients[1])


ax[0].plot(x_Si, Hex, label='Model Enthalpy')
ax[0].plot(x_Si, Gex_liq_Lacaze_Sundman(T, x_Si), label='G (L+S)')
ax[0].plot(x_Si, Hex_liq_Lacaze_Sundman(x_Si), label='H (L+S)')
ax[1].plot(x_Si, Sex, label='Model Entropy')
ax[1].plot(x_Si, Sex_liq_Lacaze_Sundman(x_Si), label='S (L+S)')
ax[1].plot(x_Si, Sex_nonconf_liq_Lacaze_Sundman(x_Si), label='Snonconf (L+S)')
ax[2].plot(x_Si, loggammaSi, label='log$_{10}\\gamma_{Si}$')

for i in range(3):
    ax[i].legend()
plt.show()
