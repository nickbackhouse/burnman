import os
import sys
# hack to allow scripts to be placed in subdirectories next to burnman:
if not os.path.exists('burnman') and os.path.exists('../burnman'):
    sys.path.insert(1, os.path.abspath('..'))

import burnman
import burnman.minerals as minerals

"""
We create instances of all the required phases and solutions in dictionaries
to allow them to be called by string
"""

endmembers = {'bcc_iron': minerals.SE_2015.bcc_iron(), # iron polymorphs
              'fcc_iron': minerals.SE_2015.fcc_iron(),
              'hcp_iron': minerals.SE_2015.hcp_iron(),
              'liq_iron': minerals.SE_2015.liquid_iron(),
              'fcc_silicon': minerals.Fe_Si.Si_fcc_A1(),
              'hcp_silicon': minerals.Fe_Si.Si_hcp_A3(),
              'liq_silicon': minerals.Fe_Si.Si_liquid(),
              'B2_FeSi': minerals.Fe_Si.FeSi_B2(),
              'B20_FeSi': minerals.Fe_Si.FeSi_B20()}


solutions = {'B2_fe_si': burnman.SolidSolution(name = 'B2-ordered bcc Fe-Si',
                                               solution_type = 'symmetric',
                                               endmembers = [[endmembers['bcc_iron'], 'Fe0.5[Fe]0.5'],
                                                             [endmembers['B2_FeSi'], 'Fe0.5[Si]0.5']],
                                               energy_interaction = [[50074.99313971981]],
                                               entropy_interaction = [[0.0e3]],
                                               volume_interaction = [[6.803233058958188e-07]]),
             'fcc_fe_si': burnman.SolidSolution(name = 'disordered fcc Fe-Si',
                                                solution_type = 'symmetric',
                                                endmembers = [[endmembers['fcc_iron'],    '[Fe]'],
                                                              [endmembers['fcc_silicon'], '[Si]']],
                                                energy_interaction = [[-99591.43339056121]],
                                                entropy_interaction = [[0.0e3]],
                                                volume_interaction = [[-5.766641001970873e-07]]),
             'hcp_fe_si': burnman.SolidSolution(name = 'disordered hcp Fe-Si',
                                                solution_type = 'symmetric',
                                                endmembers = [[endmembers['hcp_iron'],    '[Fe]'],
                                                              [endmembers['hcp_silicon'], '[Si]']],
                                                energy_interaction = [[-100.0e3]],
                                                entropy_interaction = [[0.0e3]],
                                                volume_interaction = [[0.0e3]]),
             'liq_fe_si': burnman.SolidSolution(name = 'disordered liq Fe-Si',
                                                solution_type = 'symmetric',
                                                endmembers = [[endmembers['liq_iron'],    '[Fe]'],
                                                              [endmembers['liq_silicon'], '[Si]']],
                                                energy_interaction = [[-188385.74173569347]],
                                                entropy_interaction = [[0.0e3]],
                                                volume_interaction = [[2.3518970898094444e-07]])}

child_solutions = {}
