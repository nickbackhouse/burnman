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
experiments = [{'id': 'E01a', 'P': 20.e9, 'T': 2000., 'P_unc': 1.e9, 'T_unc': 20.,
                'phases': ['fcc_fe_si', 'hcp_fe_si'],
                'Si_mol_percents': [10., 20.],
                'Si_mol_percents_unc': [1., 1.]},
               {'id': 'E02a', 'P': 20.e9, 'T': 2000., 'P_unc': 1.e9, 'T_unc': 20.,
                'phases': ['fcc_fe_si', 'hcp_fe_si'],
                'Si_mol_percents': [11., 19.],
                'Si_mol_percents_unc': [1., 1.]}]


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
            raise Exception('This file is not set up to handle endmember reactions yet')
            #phases.append(endmembers[phase_name])

    assemblage = burnman.Composite(phases)

    # Give the assemblage a name, nominal P, T state and
    # associated uncertainties as a covariance matrix
    assemblage.experiment_id = expt['id']
    assemblage.nominal_state = np.array([expt['P'], expt['T']])
    assemblage.state_covariances = np.array([[np.power(expt['P_unc'], 2.), 0.],
                                              [0., np.power(expt['T_unc'], 2.)]])

    # Assign *elemental* compositions and compositional uncertainties to the phases
    for k, phase in enumerate(expt['phases']):
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
    assemblage.stored_compositions = [(assemblage.phases[k].molar_fractions,
                                       assemblage.phases[k].molar_fraction_covariances)
                                      for k in range(n_phases)]

    # Append the assemblage to the list of experimental assemblages
    experimental_assemblages.append(assemblage)
