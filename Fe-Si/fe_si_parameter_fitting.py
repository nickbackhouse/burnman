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


# Here are lists of the parameters we're trying to fit
endmember_args = [] # nothing here yet ['wus', 'H_0', wus.params['H_0'], 1.e3]
solution_args = [['hcp_fe_si', 'E', 0, 0,
                  solutions['hcp_fe_si'].energy_interaction[0][0], 1.e3]]

# Here are lists of Gaussian priors for the parameters we're trying to fit
endmember_priors = [] # nothing here right now
solution_priors = [] # nothing here right now

# Here are lists of experiment uncertainties for when
# there are multiple sample chambers per experiment
experiment_uncertainties = [] # nothing here right now


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


def special_constraints(dataset, storage):
    pass # nothing here right now

"""
Run the minimization
"""
verbose = False
if run_inversion:
    sol = minimize(minimize_func, get_params(storage),
                   args=(dataset, storage, special_constraints, verbose),
                   method='BFGS') # , options={'eps': 1.e-02}))
    print('Finished inversion. It is *{0}* that this was successful'.format(sol.success))

# Print the current parameters
prms = get_params(storage)
print('\nNormalised parameters:')
for i in range(len(labels)):
    print('{0}: {1}'.format(labels[i], prms[i]))
print('\nTo get to the real parameters, you must multiply by the user-defined constants.')
print('You can also interrogate the dataset itself, whose properties are now updated.')
