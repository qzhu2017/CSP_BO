import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes#, PropertyNotImplementedError
np.set_printoptions(formatter={'float': '{: 8.4f}'.format})

class GPR(Calculator):
    implemented_properties = ['energy', 'forces', 'var_e', 'var_f']
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.results = {}

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):

        Calculator.calculate(self, atoms, properties, system_changes)

        d = self.parameters.descriptor.calculate(atoms)

        #
        energy_data = []
        force_data = []
        energy_data.append(d['x'])
        for i in range(len(d['x'])):
            ids = np.argwhere(d['seq'][:,1]==i).flatten()
            _i = d['seq'][ids, 0] 
            force_data.append((d['x'][_i,:], d['dxdr'][ids]))
        test_data = {"energy": energy_data, "force": force_data} 
        res, std = self.parameters.ff.predict(test_data, total_E=True, return_std=True)

        self.results['energy'] = res[0]
        self.results['free_energy'] = res[0]
        self.results['forces'] = res[1:].reshape([len(atoms), 3])
        self.results['var_e'] = std[0]
        self.results['var_f'] = std[1:].reshape([len(atoms), 3])

    def get_var_e(self, total=False):
        if total:
            return self.results["var_e"]
        else:
            return self.results["var_e"]/(len(self.results["forces"])**2)

    def get_var_f(self):
        return self.results["var_f"]


