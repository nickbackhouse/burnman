# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


from __future__ import absolute_import

import numpy as np
import warnings
import scipy.integrate
import matplotlib.pyplot as plt
from burnman.seismic import Seismic1DModel
from . import tools
from . import constants


class Seismic3DModel(object):

    """
    Base class for all the seismological models.
    """

    def __init__(self, modelname):
        self.ref = np.load('../burnman/data/input_seismic/'+modelname+'.npy').item()
    
    
    def single_profile(self, latitude, longitude):
        index = np.argmin(np.power(self.ref['lats']-latitude,2) + np.power(self.ref['lons']-longitude,2))
        return SeismicProfile(self,index)

    def percentile_profile(self, variable, percentile, depths=None):
        if depths== None:
            depths = self.ref['depths']
        profile = np.percentile(self.ref[variable],percentile,axis=1)
        return np.interp(depths, self.ref['depths'],profile)






class SeismicProfile(Seismic1DModel):
    """

    """

    def __init__(self, model3D, index):
        if isinstance(model3D, str):
            model3D=Seismic3DModel.__init__(model3D)
    
        self.table_depth = model3D.ref['depths']
        #self.table_dvp = model3D.ref['dVp'][:, index]
        self.table_dvs = model3D.ref['dVs'][:, index]
        ### Need reference absolute velocities...assume PREM???
        self.earth_radius = 6371.0e3
            
            


    def internal_depth_list(self, mindepth=0., maxdepth=1.e10):
        depths = np.array([self.table_depth[x] for x in range(len(self.table_depth)) if self.table_depth[x] >= mindepth and self.table_depth[x] <= maxdepth])
        discontinuities = np.where(depths[1:]-depths[:-1] == 0)[0]
        # Shift values at discontinities by 1 m to simplify evaluating values
        # around these.
        depths[discontinuities] = depths[discontinuities]-1.
        depths[discontinuities+1] = depths[discontinuities+1]+1.
        return depths



    def v_p(self, depth):

        return self._lookup(depth, self.table_vp)

    def v_s(self, depth):

        return self._lookup(depth, self.table_vs)


    def depth(self, pressure):
        if pressure > max(self.table_pressure) or pressure < min(self.table_pressure):
            raise ValueError("Pressure outside range of SeismicTable")

        depth = np.interp(pressure, self.table_pressure, self.table_depth)
        return depth

    def radius(self, pressure):

        radius = np.interp(pressure, self.table_pressure[
                           ::-1], self.earth_radius - self.table_depth[::-1])
        return radius

    def _lookup(self, depth):
        return np.interp(depth, self.table_depth, value_table)







