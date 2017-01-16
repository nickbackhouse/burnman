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
    models = ['SEMUCBWM1_Lmax18','GyPSuM_Lmax18','S40RTS_Lmax18','TX2011_Lmax18','SAVANI_Lmax18']
    

    
    for mod in models[:3]:
        model3D = burnman.seismic3D.Seismic3DModel(mod)
        plt.title(mod)
        depthrange = model3D.internal_depth_list(mindepth=400.e3, maxdepth=2800.e3)
        plt.subplot(2,1,1)
        for i in range(101):

            prof = model3D.percentile_profile('dVs',i, depth=2800.e3)
            plt.plot(depthrange,prof.dv_s(depthrange),color='k',alpha=0.1)
        
        prof = model3D.location_profile(0,0)
        plt.plot(depthrange,prof.dv_s(depthrange),color='r',alpha=1)

        plt.subplot(2,1,2)

        plt.show()
