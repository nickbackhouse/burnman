# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


"""
example_seismic3D
---------------




"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman

if __name__ == "__main__":

    # List of seismic 3D models
    models = ['SEMUCBWM1_Lmax18']
    for mod in models:
        model3D = burnman.seismic3D.Seismic3DModel(mod)


        for i in range(101):
            plt.plot(model3D.ref['depths'],model3D.percentile_profile('dVs',i),color='k',alpha=0.1)
            plt.plot(model3D.ref['depths'],np.mean(model3D.ref['dVs'], axis=1),'r')



    plt.show()
