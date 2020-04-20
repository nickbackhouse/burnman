from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from scipy.optimize import minimize, fsolve
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))
import burnman
from burnman.solidsolution import SolidSolution as Solution
from burnman.processanalyses import compute_and_set_phase_compositions, assemblage_affinity_misfit
from fe_si_experimental_equilibria import experimental_assemblages, solutions, endmembers, child_solutions
from parameter_fitting_functions import minimize_func, get_params, set_params, Storage

if len(sys.argv) == 2:
    if sys.argv[1] == '--fit':
        run_inversion = True
        print('Running inversion')
    else:
        run_inversion = False
        print('Not running inversion. Use --fit as command line argument to invert parameters')
else:
    run_inversion = False
    print('Not running inversion. Use --fit as command line argument to invert parameters')


dataset = {'endmembers': endmembers,
           'solutions': solutions,
           'child_solutions': {},
           'assemblages': experimental_assemblages}


"""
BEGIN USER INPUTS

Here are lists of the parameters we're trying to fit

- endmember_args should be a list of lists with the format
[<endmember name in dictionary>, <parameter name to change>, <starting value>, <normalization: expected size of change>]
For example, adding the standard Helmholtz free energy of fcc_silicon as a parameter would look like this:
['fcc_silicon', 'F_0', endmembers['fcc_silicon'].params['F_0'], 1.e3]
"""

endmember_args = [] # nothing here yet

"""
- solution_args is similar, but for solution phases.
[<solution name in dictionary>, <excess property to change (E, S or V)>, <endmember a #>, <endmember b # - endmember a # -1>, <starting value>, <normalization: expected size of change>]
For example, adding the interaction energy (E) in the binary HCP phase as a parameter would look like this:
['hcp_fe_si', 'E', 0, 0, solutions['hcp_fe_si'].energy_interaction[0][0], 1.e3]
"""

solution_args = [['B2_fe_si', 'E', 0, 0, solutions['B2_fe_si'].energy_interaction[0][0], 1.e3],
                 ['B2_fe_si', 'V', 0, 0, solutions['B2_fe_si'].volume_interaction[0][0], 1.e-8],
                 ['fcc_fe_si', 'E', 0, 0, solutions['fcc_fe_si'].energy_interaction[0][0], 1.e3],
                 ['fcc_fe_si', 'V', 0, 0, solutions['fcc_fe_si'].volume_interaction[0][0], 1.e-8],
                 ['hcp_fe_si', 'E', 0, 0, solutions['hcp_fe_si'].energy_interaction[0][0], 1.e3],
                 ['hcp_fe_si', 'V', 0, 0, solutions['hcp_fe_si'].volume_interaction[0][0], 1.e-8]]

"""
Here are lists of Gaussian priors for the parameters we're trying to fit
Leave them empty if you don't have any strong constraints on the parameter values
"""

endmember_priors = [] # nothing here right now
solution_priors = [] # nothing here right now

"""
Here are lists of experiment uncertainties for when
there are multiple sample chambers per experiment.
Leave them empty if you don't have any multichamber experiments.
"""

experiment_uncertainties = [] # nothing here right now


"""
Finally, you can declare a function allowing you to
impose special constraints on the dataset.
"""
def special_constraints(dataset, storage):
    pass # nothing here right now

"""
END USER INPUTS
"""

# Create storage object of all the priors and uncertainties
storage = Storage({'endmember_args': endmember_args,
                   'solution_args': solution_args,
                   'endmember_priors': endmember_priors,
                   'solution_priors': solution_priors,
                   'experiment_uncertainties': experiment_uncertainties})

# Create labels for each parameter
labels = [a[0]+'_'+a[1] for a in endmember_args]
labels.extend(['{0}_{1}[{2},{3}]'.format(a[0], a[1], a[2], a[3])
               for a in solution_args])
labels.extend(['{0}_{1}'.format(a[0], a[1]) for a in experiment_uncertainties])

normalizations = [a[-1] for a in endmember_args]
normalizations.extend([a[-1] for a in solution_args])
normalizations.extend([1. for a in experiment_uncertainties])

"""
Run the minimization
"""
verbose = True
if run_inversion:
    sol = minimize(minimize_func, get_params(storage),
                   args=(dataset, storage, special_constraints, verbose),
                   method='BFGS', tol=1.e-2) # , options={'eps': 1.e-02}))

    if not sol.success:
        if sol.status == 2:
            print('THE INVERSION MAY HAVE BEEN SUCCESSFUL, BUT THERE WAS A PRECISION LOSS WARNING')
            print('This could be because you have a lot of data which is not constraining any parameters.')
            print(sol)
        else:
            print('UNFORTUNATELY, THE INVERSION FAILED.')
            print('This could be because you have bad data, a lot of data which is not constraining any parameters, or you have parameters not constrained by the data.')
            print(sol)
    else:
        print('INVERSION SUCCESSFUL!')
        print('Final misfit: {0}'.format(sol.fun))

# Print the current parameters
prms = get_params(storage)
print('\nOptimised parameters:')
for i in range(len(labels)):
    print('{0}: {1}'.format(labels[i], prms[i]*normalizations[i]))

print('\nIf the dataset is used later in this file, these parameter values will be used automatically.')
print('For example, the FCC energy interaction parameter is now {0} J/mol'.format(solutions['fcc_fe_si'].energy_interaction[0][0]))
