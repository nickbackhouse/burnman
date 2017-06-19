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
    

    mindepth=400.e3
    maxdepth=2800.e3
    for mod in models[:4]:
        model3D = burnman.seismic3D.Seismic3DModel(mod)
        plt.title(mod)
        depthrange = model3D.internal_depth_list(mindepth=400.e3, maxdepth=2800.e3)
        plt.subplot(2,1,1)
        for i in range(101):

            prof = model3D.percentile_profile('dVs',i, depth=2800.e3)
            plt.plot(depthrange/1.e3,prof.dv_s(depthrange),color='k',alpha=0.1)#, label = '100 percentiles' if i==0)

        prof = model3D.maximum_profile('dVs', depth=2800.e3)
        plt.plot(depthrange/1.e3,prof.dv_s(depthrange),color='b',alpha=1, label = 'maximum profile')
        prof = model3D.median_profile('dVs', depth=2800.e3)
        plt.plot(depthrange/1.e3,prof.dv_s(depthrange),color=[1.0,0,1.0],alpha=1, label = 'median profile')
        prof = model3D.minimum_profile('dVs', depth=2800.e3)
        plt.plot(depthrange/1.e3,prof.dv_s(depthrange),color='r',alpha=1, label = 'minimum profile')
        prof = model3D.location_profile(latitude = 0, longitude = 0)
        plt.plot(depthrange/1.e3,prof.dv_s(depthrange),color='g',alpha=1, label = 'profile at (0,0)')
        plt.xlim([mindepth/1.e3,maxdepth/1.e3])
        plt.xlabel('Depth (km)')
        plt.ylabel('dlnV_S')
        plt.ylim([-0.04,0.06])
        plt.legend(fontsize=10)
        
        
        plt.subplot(2,1,2)
        
        for depth in depthrange:
            values = model3D.range_at_depth('dVs', depth)
            toplot,xedge=np.histogram(values,bins=20)
            toplot=np.rot90([toplot])
        
            print(toplot)
            plt.imshow(toplot,extent=[depth/1.e3-50., depth/1.e3+50,xedge[0],xedge[-1]],aspect='auto')
            plt.xlabel('depth ($km$)')
            plt.ylabel('dlnVs')
        
        plt.xlim(np.min(depthrange)/1.e3,np.max(depthrange)/1.e3)

        plt.show()
