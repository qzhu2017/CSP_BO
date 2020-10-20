import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes#, PropertyNotImplementedError
eV2GPa = 160.21766

class GPR(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'var_e', 'var_f']
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.results = {}

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):

        Calculator.calculate(self, atoms, properties, system_changes)
        stress = self.parameters.stress
        return_std=self.parameters.return_std
        if return_std:
            res = self.parameters.ff.predict_structure(atoms, stress, True)
            self.results['var_e'] = res[3]
            self.results['var_f'] = res[4]
        else:
            res = self.parameters.ff.predict_structure(atoms, stress, False)

        self.results['energy'] = res[0]
        self.results['free_energy'] = res[0]
        self.results['forces'] = res[1]
        if stress:
            self.results['stress'] = res[2].sum(axis=0)*eV2GPa
        else:
            self.results['stress'] = None

    def get_var_e(self, total=False):
        if total:
            return self.results["var_e"]
        else:
            return self.results["var_e"]/(len(self.results["forces"])**2)

    def get_var_f(self):
        return self.results["var_f"]

    def get_e(self, peratom=True):
        if peratom:
            return self.results["energy"]/len(self.results["forces"])
        else:
            return self.results["energy"]


