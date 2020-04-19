from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
from fe_si_phases import solutions, endmembers, child_solutions

"""
HERE IS WHERE WE PUT ALL THE AVAILABLE EQUILIBRIUM DATA!!
AT THE MOMENT, THIS IS SET UP FOR EQUILIBRIA INVOLVING ONLY SOLUTION PHASES.
ASK ME IF YOU NEED SOME EQUILIBRIA INVOLVING ENDMEMBERS
"""
experiments = [{'id': 'ozawa_3', 'P': 47.e9, 'T': 3160, 'P_unc': 4e9, 'T_unc': 170,
                'phases': ['liq_fe_si','fcc_fe_si'],
                'Si_mol_percents': [18.1, 15.7],
                'Si_mol_percents_unc': [0.360, 0.729]},
               {'id': 'ozawa_4', 'P': 56e9, 'T': 3020, 'P_unc': 4e9, 'T_unc': 40,
                'phases': ['liq_fe_si','fcc_fe_si'],
                'Si_mol_percents': [17.1, 14.1],
                'Si_mol_percents_unc': [0.362, 0.368]},
                {'id': 'ozawa_7', 'P': 58e9, 'T': 3150, 'P_unc': 7e9, 'T_unc': 100,
                    'phases': ['fcc_fe_si', 'liq_fe_si'],
                    'Si_mol_percents': [11.4, 12.3],
                    'Si_mol_percents_unc': [0, 0.371]},
                {'id': 'ozawa_8', 'P': 34e9, 'T': 2550, 'P_unc': 6e9, 'T_unc': 0,
                'phases': ['fcc_fe_si', 'liq_fe_si'],
                'Si_mol_percents': [16.1, 17.1],
                'Si_mol_percents_unc': [0.727, 0.362]},
                {'id': 'ozawa_16', 'P': 115e9, 'T': 4060, 'P_unc': 4e9, 'T_unc': 0,
                'phases': ['B2_fe_si', 'liq_fe_si'],
                'Si_mol_percents': [16.8, 14.1],
                'Si_mol_percents_unc': [0.362, 0.365]},
                {'id': 'ozawa_21', 'P': 120e9, 'T': 3910, 'P_unc': 4e9, 'T_unc': 150,
                'phases': ['B2_fe_si', 'liq_fe_si'],
                'Si_mol_percents': [6.54, 4.28],
                'Si_mol_percents_unc': [2.672, 1.545]},
                {'id': 'ozawa_KH', 'P': 127e9, 'T': 3910, 'P_unc': 4e9, 'T_unc': 20,
                'phases': ['B2_fe_si', 'liq_fe_si'],
                'Si_mol_percents': [7.28, 2.93],
                'Si_mol_percents_unc': [0, 0.389]},
                {'id': 'fischeut_1', 'P': 50e9, 'T': 2750, 'P_unc': 0e9, 'T_unc': 50,
                'phases': ['fcc_fe_si','B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [9.47, 25.97, 21.33],
                'Si_mol_percents_unc': [1, 1, 1]},
                {'id': 'fischeut_2', 'P': 80e9, 'T': 3282, 'P_unc': 0e9, 'T_unc': 50,
                'phases': ['fcc_fe_si','B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [7.65, 25.97, 13.02],
                'Si_mol_percents_unc': [1, 1, 1]},
                {'id': 'fischeut_3', 'P': 125e9, 'T': 3813, 'P_unc': 0e9, 'T_unc': 50,
                'phases': ['hcp_fe_si','B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [7.65, 28.94, 11.26],
                'Si_mol_percents_unc': [1, 1, 1]},
                {'id': 'fischeut_4', 'P': 145e9, 'T': 3907, 'P_unc': 0e9, 'T_unc': 50,
                'phases': ['hcp_fe_si','B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [7.65, 28.94, 11.26],
                'Si_mol_percents_unc': [1, 1, 1]},
                {'id': 'mydata_fe-si_8', 'P': 64.5e9, 'T': 2640, 'P_unc': 1e9, 'T_unc': 56,
                'phases': ['liq_fe_si','fcc_fe_si','B2_fe_si'],
                'Si_mol_percents': [17.5, 9., 26],
                'Si_mol_percents_unc': [2, 2, 2]},
                {'id': 'mydata_fe-si_8B', 'P': 60.95e9, 'T': 2761, 'P_unc': 1e9, 'T_unc': 51,
                'phases': ['liq_fe_si','fcc_fe_si','B2_fe_si'],
                'Si_mol_percents': [18.5, 9, 25],
                'Si_mol_percents_unc': [2, 2, 2]},
                {'id': 'mydata_fe-si_9', 'P': 34.6e9, 'T': 2380, 'P_unc': 1e9, 'T_unc': 31,
                'phases': ['liq_fe_si','fcc_fe_si','B2_fe_si'],
                'Si_mol_percents': [24, 9.5, 28],
                'Si_mol_percents_unc': [2, 2, 2]},
                {'id': 'mydata_fe-si_9B', 'P': 35.2e9, 'T': 2371, 'P_unc': 1e9, 'T_unc': 19,
                'phases': ['liq_fe_si','fcc_fe_si','B2_fe_si'],
                'Si_mol_percents': [24, 9.5, 28],
                'Si_mol_percents_unc': [2, 2, 2]},
                {'id': 'mydata_fe-si_10', 'P': 48.3e9, 'T': 2491, 'P_unc': 1e9, 'T_unc': 14,
                'phases': ['liq_fe_si','fcc_fe_si','B2_fe_si'],
                'Si_mol_percents': [21, 9.5, 26],
                'Si_mol_percents_unc': [2, 2, 2]},
                {'id': 'fisch120210', 'P': 19.5e9, 'T': 1792, 'P_unc': 0.7e9, 'T_unc': 101,
                'phases': ['fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [9.47,41.13],
                'Si_mol_percents_unc': [1.5, 3]},
                {'id': 'fisch20080315_1', 'P': 36.8e9, 'T': 2630, 'P_unc': 3.6e9, 'T_unc': 150,
                'phases': ['fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [9.47, 37.92],
                'Si_mol_percents_unc': [1.5, 3]},
                {'id': 'fisch20080315_2', 'P': 37.0e9, 'T': 2538, 'P_unc': 2.7e9, 'T_unc': 104,
                'phases': ['fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [9.47, 37.92],
                'Si_mol_percents_unc': [1.5, 3]},
                {'id': 'fisch20080315_3', 'P': 36.3e9, 'T': 2248, 'P_unc': 2.4e9, 'T_unc': 100,
                'phases': ['fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [9.47, 37.92],
                'Si_mol_percents_unc': [1.5, 3]},
                {'id': 'fisch20080315_4', 'P': 44.4e9, 'T': 2864, 'P_unc': 2.8e9, 'T_unc': 150,
                'phases': ['fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [9.47, 21.33],
                'Si_mol_percents_unc': [1.5, 3]},
                {'id': 'fisch20080315_5', 'P': 45.5e9, 'T': 2750, 'P_unc': 3.8e9, 'T_unc': 150,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [9.47, 21.33],
                'Si_mol_percents_unc': [1.5, 3]},
                {'id': 'fisch20080711_1', 'P': 62.4e9, 'T': 3340, 'P_unc': 3.5e9, 'T_unc': 101,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [8.71, 18.50],
                'Si_mol_percents_unc': [1.5, 3]},
                {'id': 'fisch110814_1', 'P': 73.5e9, 'T': 3180, 'P_unc': 1.6e9, 'T_unc': 100,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [26, 15.70],
                'Si_mol_percents_unc': [3, 3]},
                {'id': 'fisch20080711_2', 'P': 80.8e9, 'T': 3165, 'P_unc': 3.9e9, 'T_unc': 150,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [26, 13],
                'Si_mol_percents_unc': [3, 3]},
                {'id': 'fisch110814_2', 'P': 88.4e9, 'T': 3629, 'P_unc': 3.9e9, 'T_unc': 113,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [26.58, 12.70],
                'Si_mol_percents_unc': [3, 3]},
                {'id': 'fisch110814_3', 'P': 100.4e9, 'T': 3712, 'P_unc': 1.8e9, 'T_unc': 100,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [27.37, 11],
                'Si_mol_percents_unc': [3, 3]},
                {'id': 'Fe-FeSi01A', 'P': 10e9, 'T': 1746, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [28.9, 9.5, 23.10],
                'Si_mol_percents_unc': [7.4, 7.4, 7.4]},
                {'id': 'Fe-FeSi01B', 'P': 9e9, 'T': 1765, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [28.9, 9.5, 22.80],
                'Si_mol_percents_unc': [7.4, 7.4, 7.4]},
                {'id': 'Fe-FeSi_7', 'P': 88.77e9, 'T': 3327, 'P_unc': 2e9, 'T_unc': 174,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [26.5, 7.6, 12.6],
                'Si_mol_percents_unc': [7.4, 7.4, 7.4]},
                {'id': 'Fe-FeSi_6B', 'P': 70.67e9, 'T': 3096, 'P_unc': 2e9, 'T_unc': 108,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [25.9, 8.2, 15.7],
                'Si_mol_percents_unc': [7.4, 7.4, 7.4]},
                {'id': 'Fe-FeSi_1A', 'P': 20e9, 'T': 2093, 'P_unc': 1e9, 'T_unc': 131,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [45, 9.5, 26],
                'Si_mol_percents_unc': [7.4, 7.4, 7.4]},
                {'id': 'FSi-61', 'P': 23e9, 'T': 2028, 'P_unc': 1e9, 'T_unc': 100,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [48, 9.5, 38.05],
                'Si_mol_percents_unc': [7.4, 7.4, 6.4]},
                {'id': 'FSi-60', 'P': 11e9, 'T': 1850, 'P_unc': 1e9, 'T_unc': 100,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [47, 9.5, 37.13],
                'Si_mol_percents_unc': [7.4, 7.4, 7.4]},
                {'id': 'FSi-59_6', 'P': 18e9, 'T': 2050, 'P_unc': 1e9, 'T_unc': 100,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [41, 9.5, 31.7],
                'Si_mol_percents_unc': [7.4, 7.4, 7.4]},
                {'id': 'FSi-59_21', 'P': 36e9, 'T': 2200, 'P_unc': 1e9, 'T_unc': 100,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [47, 9.5, 42.6],
                'Si_mol_percents_unc': [7.4, 7.4, 7.4]},
                {'id': 'FSi61_09', 'P': 23e9, 'T': 2028, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [48, 38.05],
                'Si_mol_percents_unc': [8.3, 6.4]},
                {'id': 'FSi61_11', 'P': 23e9, 'T': 2198, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [46, 35.93],
                'Si_mol_percents_unc': [8.3, 8.3]},
                {'id': 'FSi61_11', 'P': 23e9, 'T': 2198, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [46, 35.93],
                'Si_mol_percents_unc': [8.3, 8.3]},
                {'id': 'FSi61_12', 'P': 23e9, 'T': 2260, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [42, 31.8],
                'Si_mol_percents_unc': [8.3, 8.3]},
                {'id': 'FSi61_13', 'P': 23e9, 'T': 2319, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [39, 28.9],
                'Si_mol_percents_unc': [8.3, 6.7]},
                {'id': 'FSi61_14', 'P': 23e9, 'T': 2346, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [39, 28.9],
                'Si_mol_percents_unc': [8.3, 4.4]},
                {'id': 'FSi61_15', 'P': 23e9, 'T': 2470, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [33, 22.9],
                'Si_mol_percents_unc': [8.3, 5.2]},
                {'id': 'K&H2004', 'P': 21e9, 'T': 2093, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [50, 10, 41.1],
                'Si_mol_percents_unc': [10, 10, 2]},
                {'id': 'Lac&sund', 'P': 0e9, 'T': 1500, 'P_unc': 0e9, 'T_unc': 50,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [50, 10, 35.3],
                'Si_mol_percents_unc': [10, 10, 2]},
                {'id': 'K&H2004', 'P': 21e9, 'T': 2093, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','fcc_fe_si','liq_fe_si'],
                'Si_mol_percents': [50, 10, 41.1],
                'Si_mol_percents_unc': [10, 10, 2]},
                {'id': 'asanuma2010_FESI16', 'P': 22e9, 'T': 2210, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [30.39, 25.97],
                'Si_mol_percents_unc': [1,5]},
                {'id': 'asanuma2010_FESI04', 'P': 28e9, 'T': 2290, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [30.39, 25.97],
                'Si_mol_percents_unc': [1,5]},
                {'id': 'asanuma2010_FESI05', 'P': 34e9, 'T': 2610, 'P_unc': 6e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [30.39, 25.97],
                'Si_mol_percents_unc': [1,5]},
                {'id': 'asanuma2010_FESI06', 'P': 49e9, 'T': 2730, 'P_unc': 1e9, 'T_unc': 100,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [30.39, 25.97],
                'Si_mol_percents_unc': [1,5]},
                {'id': 'asanuma2010_FESI12', 'P': 58e9, 'T': 2800, 'P_unc': 1e9, 'T_unc': 100,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [30.39, 25.97],
                'Si_mol_percents_unc': [1,5]},
                {'id': 'asanuma2010_FESI09', 'P': 68e9, 'T': 2880, 'P_unc': 2e9, 'T_unc': 50,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [30.39, 25.97],
                'Si_mol_percents_unc': [1,5]},
                {'id': 'asanuma2010_FESI14', 'P': 104e9, 'T': 3060, 'P_unc': 1e9, 'T_unc': 100,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [30.39, 21.33],
                'Si_mol_percents_unc': [1,5]},
                {'id': 'asanuma2010_FESI15', 'P': 119e9, 'T': 3240, 'P_unc': 1e9, 'T_unc': 100,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [30.39, 11.26],
                'Si_mol_percents_unc': [1,5]},
                {'id': 'asanuma2010_FESI15', 'P': 119e9, 'T': 3240, 'P_unc': 1e9, 'T_unc': 100,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [30.39, 11.26],
                'Si_mol_percents_unc': [1,5]},
                {'id': 'fischer2013_59', 'P': 80.3e9, 'T': 3615, 'P_unc': 1.6e9, 'T_unc': 121,
                'phases': ['B2_fe_si','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi23A', 'P': 2e9, 'T': 1999, 'P_unc': 0.5e9, 'T_unc': 69,
                'phases': ['B20_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi23A', 'P': 2e9, 'T': 1999, 'P_unc': 0.5e9, 'T_unc': 69,
                'phases': ['B20_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi21A', 'P': 5e9, 'T': 2121, 'P_unc': 1e9, 'T_unc': 82,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi21B', 'P': 7e9, 'T': 2235, 'P_unc': 1e9, 'T_unc': 103,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi02C', 'P': 10e9, 'T': 2322, 'P_unc': 1e9, 'T_unc': 58,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi19B', 'P': 14e9, 'T': 2474, 'P_unc': 1e9, 'T_unc': 62,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi19A', 'P': 15e9, 'T': 2415, 'P_unc': 1e9, 'T_unc': 68,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi03A', 'P': 20e9, 'T': 2628, 'P_unc': 2e9, 'T_unc': 81,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi20A', 'P': 22e9, 'T': 2697, 'P_unc': 1e9, 'T_unc': 50,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi19C', 'P': 26e9, 'T': 2788, 'P_unc': 1e9, 'T_unc': 71,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi20B', 'P': 35e9, 'T': 2966, 'P_unc': 4e9, 'T_unc': 74,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi19E', 'P': 44e9, 'T': 3033, 'P_unc': 3.8e9, 'T_unc': 96,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi19E', 'P': 52e9, 'T': 3204, 'P_unc': 3e9, 'T_unc': 77,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi22A', 'P': 52e9, 'T': 3204, 'P_unc': 3e9, 'T_unc': 77,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi22B', 'P': 55e9, 'T': 3175, 'P_unc': 2e9, 'T_unc': 72,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi20C', 'P': 59e9, 'T': 3356, 'P_unc': 2e9, 'T_unc': 86,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi19F', 'P': 68e9, 'T': 3461, 'P_unc': 1.4e9, 'T_unc': 180,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi27A', 'P': 99e9, 'T': 3860, 'P_unc': 4e9, 'T_unc': 138,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi26B', 'P': 105e9, 'T': 3859, 'P_unc': 4e9, 'T_unc': 139,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi25C', 'P': 139e9, 'T': 3951, 'P_unc': 4e9, 'T_unc': 123,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]},
                {'id': 'lord2010_FeSi25C', 'P': 152e9, 'T': 4069, 'P_unc': 4e9, 'T_unc': 98,
                'phases': ['B2_FeSi','liq_fe_si'],
                'Si_mol_percents': [50, 50],
                'Si_mol_percents_unc': [0.1,0.1]}]



#this is all the data I have, lots of the compositions are estimated from Fischer, but where I've had to estimate I've used large errors. The only data not here is some of the FIscher data, three of the pieces of melting data have three coextisting solid phases, and im not sure how to estimate their compositions.


"""
This is where we load the state and compositional information into assemblages.
Unless you want to add new phases, you shouldn't need to do anything here.
"""
experimental_assemblages = []
for i, expt in enumerate(experiments):

    n_phases = len(expt['phases'])

    phases = []
    # Fill the assemblage with the correct phases in the correct order
    for k, phase_name in enumerate(expt['phases']):
        try:
            phases.append(solutions[phase_name])
        except KeyError:
            try:
                phases.append(endmembers[phase_name])
            except KeyError:
                raise Exception('Phase not recognised')

    assemblage = burnman.Composite(phases)

    # Give the assemblage a name, nominal P, T state and
    # associated uncertainties as a covariance matrix
    assemblage.experiment_id = expt['id']
    assemblage.nominal_state = np.array([expt['P'], expt['T']])
    assemblage.state_covariances = np.array([[np.power(expt['P_unc'], 2.), 0.],
                                              [0., np.power(expt['T_unc'], 2.)]])

    # Assign *elemental* compositions and compositional uncertainties to the phases
    for k, phase in enumerate(expt['phases']):

        # We only need to do this for solution phases
        if assemblage.phases[k] in solutions.values():
            assemblage.phases[k].fitted_elements = ['Fe', 'Si']

            x_Si = expt['Si_mol_percents'][k]/100.
            x_Fe = 1. - x_Si
            x_Si_unc = max(expt['Si_mol_percents_unc'][k]/100., 0.001)/np.sqrt((1. - 2.*x_Fe*(1. - x_Fe)))
            x_Fe_unc = x_Si_unc

            assemblage.phases[k].composition = np.array([x_Fe, x_Si])
            assemblage.phases[k].compositional_uncertainties = np.array([x_Fe_unc, x_Si_unc])

    # Compute the *phase proportions* and associated uncertainties
    burnman.processanalyses.compute_and_set_phase_compositions(assemblage)

    # Store the compositions as attributes of the *assemblage*
    # rather than attributes of each phase so that they don't get overwritten
    assemblage.stored_compositions = ['composition not assigned']*len(assemblage.phases)
    for k in range(len(phases)):
        try:
            assemblage.stored_compositions[k] = (assemblage.phases[k].molar_fractions,
                                                 assemblage.phases[k].molar_fraction_covariances)
        except AttributeError:
            pass

    # Append the assemblage to the list of experimental assemblages
    experimental_assemblages.append(assemblage)