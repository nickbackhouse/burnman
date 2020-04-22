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
                                               energy_interaction = [[37608.3866743928]],
                                               entropy_interaction = [[0.0e3]],
                                               volume_interaction = [[-1.3183370865265117e-06]]),
             'fcc_fe_si': burnman.SolidSolution(name = 'disordered fcc Fe-Si',
                                                solution_type = 'symmetric',
                                                endmembers = [[endmembers['fcc_iron'],    '[Fe]'],
                                                              [endmembers['fcc_silicon'], '[Si]']],
                                                energy_interaction = [[ -108036.88253048982]],
                                                entropy_interaction = [[0.0e3]],
                                                volume_interaction = [[-3.514522148863687e-06]]),
             'hcp_fe_si': burnman.SolidSolution(name = 'disordered hcp Fe-Si',
                                                solution_type = 'symmetric',
                                                endmembers = [[endmembers['hcp_iron'],    '[Fe]'],
                                                              [endmembers['hcp_silicon'], '[Si]']],
                                                energy_interaction = [[-296760.620588609]],
                                                entropy_interaction = [[0.0e3]],
                                                volume_interaction = [[-1.2127395249052055e-06]]),
             'liq_fe_si': burnman.SolidSolution(name = 'disordered liq Fe-Si',
                                                solution_type = 'symmetric',
                                                endmembers = [[endmembers['liq_iron'],    '[Fe]'],
                                                              [endmembers['liq_silicon'], '[Si]']],
                                                energy_interaction = [[-104209.67164712933]],
                                                entropy_interaction = [[0.0e3]],
                                                volume_interaction = [[-3.1293761576923608e-06]])}

child_solutions = {}
