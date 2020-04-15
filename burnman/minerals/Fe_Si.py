# BurnMan - a lower mantle toolkit
# Copyright (C) 2012, 2013, Heister, T., Unterborn, C., Rose, I. and Cottaar, S.
# Released under GPL v2 or later.

"""
Fe-Si database
"""

from burnman.mineral import Mineral
from burnman.solidsolution import SolidSolution
from burnman.solutionmodel import *
from burnman.processchemistry import read_masses, dictionarize_formula, formula_mass
from burnman.constants import Avogadro

# Lacaze and Sundman (1991) suggest  0.5*Fe(nonmag) + 0.5*Si - 36380.6 + 2.22T
class FeSi_B20 (Mineral): # WARNING, no magnetic properties to avoid screwing up Barin
    def __init__(self):
        formula='Fe1.0Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeSi B20',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -78852. , # Barin
            'S_0': 44.685 , # Barin
            'V_0': 1.359e-05 ,
            'Cp': [38.6770, 0.0217569, -159.151, 0.00586],
            'a_0': 3.057e-05 ,
            'K_0': 2.057e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.057e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)

class FeSi_B2 (Mineral): # enthalpy and entropy calculated from A2-structured Fe and Si
    def __init__(self):
        formula='Fe0.5Si0.5'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Fe0.5Si0.5 B2',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': (9149.0 + 47000.)/2. -81000./2., # -59650./2. includes enthalpy of ordering
            'S_0': (36.868 + 18.820 + 22.5)/2. -9.5050/2., # includes entropy of ordering
            'V_0': 1.298e-05/2. ,
            'Cp': [(21.09 + 22.826)/2., (0.0101455 + 0.003856857)/2., (-221508.+-353888.416)/2., (47.1947 + -0.0596068)/2.],
            'a_0': 3.580e-05 ,
            'K_0': 2.208e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.199e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula),
            'curie_temperature': [521.5, 0.0] ,
            'magnetic_moment': [1.11, 0.0] ,
            'magnetic_structural_parameter': 0.4 }
        Mineral.__init__(self)

'''
class FeSi_B2 (Mineral):
    def __init__(self):
        formula='Fe1.0Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeSi B2',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -78852. , # to fit
            'S_0': 44.685 , # to fit
            'V_0': 1.300e-05 ,
            'Cp': [38.6770e+01, 0.0217569, -159.151, 0.0060376],
            'a_0': 3.064e-05 ,
            'K_0': 2.199e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.199e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula),
            'curie_temperature': [521.5, 0.0] ,
            'magnetic_moment': [1.11, 0.0] ,
            'magnetic_structural_parameter': 0.4 }
        Mineral.__init__(self)
'''

class Si_diamond_A4 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si A4',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 0. , # Barin
            'S_0': 18.820 , # Barin
            'V_0': 1.2057e-05 , # Refit from Anzellini (2019)
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 7.757256e-06 , # Fit to Roberts, 1981
            'K_0': 102.e9 , # Refit from Anzellini (2019)
            'Kprime_0': 3.3 , # [fixed]
            'Kdprime_0': -3.3/102e+9 , # [heuristic]
            'T_einstein': 764., # Fit to Roberts, 1981 (would be 516. from 0.8*Tdebye (645 K); see wiki)
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)

class Si_hcp_A3 (Mineral):

    def __init__(self):
        formula = 'Si'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'HCP Si',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': 53280.,# - 7000., # make 7kJ/mol more stable to get closer to melting curve
            'V_0': 8.61166e-6, # 8.658e-6,
            'K_0': 100.e9,
            'Kprime_0': 4.0,
            'Debye_0': 600., # A4 is 645 K
            'grueneisen_0': 1.0,
            'q_0': 1.,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)

class Si_fcc_A1 (Mineral):

    def __init__(self):
        formula = 'Si'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FCC Si',
            'formula': formula,
            'equation_of_state': 'slb3',
            'F_0': 86540. - 1000., # make 1kJ/mol more stable to fit transition
            'V_0': 7.665e-6,
            'K_0': 159.e9,
            'Kprime_0': 4.,
            'Debye_0': 600., # A4 is 645 K
            'grueneisen_0': 1.0,
            'q_0': 1.,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)

class Si_liquid( Mineral ):
    """
    Liquid silicon using the Brosh-Calphad equation of state
    """
    def __init__(self):
        formula={'Si': 1.}
        m = formula_mass(formula)
        self.params = {
            'name': 'Si liquid (BroshEoS)',
            'formula': formula,
            'equation_of_state': 'brosh_calphad',
            'molar_mass': m,
            'n': sum(formula.values()),
            'gibbs_coefficients': [[12000., [42046., 107.4096, -22.826, 176944.208,
                                             0., 0., 0., -1.92843e-3, 0., 0.,
                                             0., -0.236272, 0.]]],
            'V_0': 10.1e-06 , # 9.84e-6 gives a decent Fit to Watanabe et al. (2007) at 1 bar
            'theta_0': 764./0.806, # Fit to Roberts, 1981 for A4
            'K_0': 45.e9 , # 40.e9 is a decent fit to Watanabe et al. (2007) at 1 bar
            'Kprime_0': 5.4 , # 5. is Anzellini for phase V
            'grueneisen_0': 1.0, # 1. is a decent fit to Watanabe et al. (2007) at 1 bar
            'delta': [2., 0.], # b5, b7
            'b': [1., 1.] # b4, b6
        }
        Mineral.__init__(self)


# Half; Lacaze and Sundman (1991) suggest  0.5*Fe(nonmag) + 0.5*Si - 36380.6 + 2.22T
class half_FeSi_B20 (Mineral): # WARNING, no magnetic properties to avoid screwing up Barin
    def __init__(self):
        formula='Fe0.5Si0.5'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'FeSi B20',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': -78852./2. , # Barin
            'S_0': 44.685/2. , # Barin
            'V_0': 1.359e-05/2. ,
            'Cp': [38.6770/2., 0.0217569/2., -159.151/2., 0.00586/2.],
            'a_0': 3.057e-05 ,
            'K_0': 2.057e+11 ,
            'Kprime_0': 4.0 ,
            'Kdprime_0': -4.0/2.057e+11 ,
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)

"""
class Si_bcc_A2 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si bcc A2',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 47000., # Saunders et al. 1988
            'S_0': 18.820 + 22.5, # Barin, Saunders et al. 1988 suggest dGrxn = 47000 - 22.5T
            'V_0': 9.1e-06 , # Hallstedt, 2007
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 7.757256e-06 , # Fit to Roberts, 1981 for A4
            'T_einstein': 764., # Fit to Roberts, 1981 for A4
            'K_0': 50.e+9 , # ? Must destabilise BCC relative to HCP, FCC
            'Kprime_0': 6.0 , # Similar to HCP, FCC
            'Kdprime_0': -6.0/50.e+9 , # ?
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)

class Si_hcp_A3 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si hcp A3',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 67331. -1847. -10000. - 0.*300., # Fit to Anzellini curves
            'S_0': 18.820 + 20.8 - 0., # Barin, Saunders et al. 1988 suggest dGrxn = 49200 - 20.8T
            'V_0': 8.51e-06 , # Refit from Anzellini (2019)
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 15.e-06 , # 7.757256e-06 is Fit to Roberts, 1981 for A4
            'T_einstein': 764., # Fit to Roberts, 1981 for A4
            'K_0': 111.e9 , # Refit from Anzellini (2019)
            'Kprime_0': 4. , # [fixed]
            'Kdprime_0': -4./111.e9 , # [heuristic]
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)

class Si_fcc_A1 (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si fcc A1',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 99850. -102., # Fit to Anzellini curves
            'S_0': 18.820 + 21.8, # Barin, Saunders et al. 1988 suggest dGrxn = 51000 - 21.8T
            'V_0': 7.6e-06 , # Refit to Anzellini, 2019 (Hallstedt, 2007 is 9.2!!)
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin
            'a_0': 15.e-06 , # 7.757256e-06 Fit to Roberts, 1981 for A4
            'T_einstein': 764., # Fit to Roberts, 1981 for A4
            'K_0': 170.e9 , # Refit from Anzellini (2019)
            'Kprime_0': 4.0 , # [fixed]
            'Kdprime_0': -4.0/170.e9 , # [heuristic]
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)

class Si_liquid (Mineral):
    def __init__(self):
        formula='Si1.0'
        formula = dictionarize_formula(formula)
        self.params = {
            'name': 'Si liquid',
            'formula': formula,
            'equation_of_state': 'hp_tmt',
            'H_0': 50208. , # SGTE data
            'S_0': 18.820 + 29.797, # Barin, Saunders et al. 1988 suggest dGrxn = 50208 - 29.762T
            'V_0': 10.07e-06 , # Fit to Watanabe et al. (2007); 9.154e-06 is Anzellini for phase V
            'Cp': [22.826, 0.003856857, -353888.416, -0.0596068], # Barin for A4, assume the same (in Barin, is constant 27.196, but this is probably not that accurate). Agrees with Kobatake et al. (2008): 30+/-5 J mol−1K−1 from 1750–2050K.
            'a_0': 29.e-06 , # Fit to Watanabe et al. (2007)
            'T_einstein': 764., # Fit to Roberts, 1981 for A4
            'K_0': 60.e9 , # 99.e9 is Anzellini for phase V
            'Kprime_0': 6. , # 5. is Anzellini for phase V
            'Kdprime_0': -6./60.e9 , # [heuristic]
            'n': sum(formula.values()),
            'molar_mass': formula_mass(formula)}
        Mineral.__init__(self)
"""
