import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes#, PropertyNotImplementedError
from ase.neighborlist import NeighborList
from ase.constraints import full_3x3_to_voigt_6_stress
from cspbo.utilities import metric_single

eV2GPa = 160.21766

class GPR(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'var_e', 'var_f']
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        #self.base_calculator = base_calculator # temperorally
        #self.e_tol = e_tol
        #self.f_tol = f_tol
        self.results = {}

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        self._calculate(atoms, properties)

        e_tol = 1.2 * self.parameters.ff.noise_e
        f_tol = 1.2 * self.parameters.ff.noise_f
        E_std, F_std = self.results['var_e'], self.results['var_f'].max()

        if E_std > e_tol or F_std > f_tol:
            # update model
            E = self.results['energy']
            atoms.calc = self.parameters.base_calculator
            data = (atoms, atoms.get_potential_energy(), atoms.get_forces())
            print("Switch to base calculator, E_std: {:.3f}/{:.3f}/{:.3f}, F_std: {:.3f}".format(E_std, E, data[1], F_std))
            pts, N_pts, _ = self.parameters.ff.add_structure(data)
            if N_pts > 0:
                self.parameters.ff.set_train_pts(pts, mode='a+')
                self.parameters.ff.fit(opt=True, show=False)
                #train_E, train_E1, train_F, train_F1 = self.parameters.ff.validate_data()
                #l1 = metric_single(train_E, train_E1, "Train Energy")
                #l2 = metric_single(train_F, train_F1, "Train Forces")
                #self.parameters.ff.sparsify()
                #print(self.parameters.ff)
                #atoms.write('test.cif', format='cif')
                #import sys; sys.exit()
            self._calculate(atoms, properties)
            atoms.calc = self
        else:
            print("Using the surrogate model, E_std: {:.3f}, F_std: {:.3f}".format(E_std, F_std))

    def _calculate(self, atoms, properties,
                  system_changes=all_changes):

        Calculator.calculate(self, atoms, properties, system_changes)
        if hasattr(self.parameters, 'stress'):
            stress = self.parameters.stress
        else:
            stress = False
        if hasattr(self.parameters, 'f_tol'):
            f_tol = self.parameters.f_tol
        else:
            f_tol = 1e-12

        if hasattr(self.parameters, 'return_std'):
            return_std=self.parameters.return_std
        else:
            return_std=False

        if return_std:
            #print(atoms)
            res = self.parameters.ff.predict_structure(atoms, stress, True, f_tol=f_tol)
            self.results['var_e'] = res[3]
            self.results['var_f'] = res[4]

        else:
            res = self.parameters.ff.predict_structure(atoms, stress, False, f_tol=f_tol)

        self.results['energy'] = res[0]
        self.results['free_energy'] = res[0]
        self.results['forces'] = res[1]
        if stress:
            self.results['stress'] = res[2].sum(axis=0) #*eV2GPa
        else:
            self.results['stress'] = None

    def get_var_e(self, total=False):
        if total:
            return self.results["var_e"]*len(self.results["forces"]) # eV
        else:
            return self.results["var_e"] # eV/atom

    def get_var_f(self):
        return self.results["var_f"]

    def get_e(self, peratom=True):
        if peratom:
            return self.results["energy"]/len(self.results["forces"])
        else:
            return self.results["energy"]



class LJ():
    """
    Pairwise LJ model (mostly copied from `ase.calculators.lj`)
    https://gitlab.com/ase/ase/-/blob/master/ase/calculators/lj.py

    Args:
        atoms: ASE atoms object
        parameters: dictionary to store the LJ parameters

    Returns:
        energy, force, stress
    """
    def __init__(self, parameters=None):
        # Set up default descriptors parameters
        keywords = ['rc', 'sigma', 'epsilon']
        _parameters = {
                       'name': 'LJ',
                       'rc': 5.0,
                       'sigma': 1.0, 
                       'epsilon': 1.0,
                      }
    
        if parameters is not None:
            _parameters.update(parameters)

        self.load_from_dict(_parameters)

    def __str__(self):
        return "LJ(eps: {:.3f}, sigma: {:.3f}, cutoff: {:.3f})".format(\
        self.epsilon, self.sigma, self.rc)

    def load_from_dict(self, dict0):
        self._parameters = dict0
        self.name = self._parameters["name"]
        self.epsilon = self._parameters["epsilon"]
        self.sigma = self._parameters["sigma"]
        self.rc = self._parameters["rc"]

       
    def save_dict(self):
        """
        save the model as a dictionary in json
        """
        return self._parameters

    def calculate(self, atoms):
        """
        Compute the E/F/S
        
        Args:
            atom: ASE atoms object
        """

        sigma, epsilon, rc = self.sigma, self.epsilon, self.rc

        natoms = len(atoms)
        positions = atoms.positions
        cell = atoms.cell

        e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        nl = NeighborList([rc / 2] * natoms, self_interaction=False)
        nl.update(atoms)

        for ii in range(natoms):
            neighbors, offsets = nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)

            # pointing *towards* neighbours
            distance_vectors = positions[neighbors] + cells - positions[ii]

            r2 = (distance_vectors ** 2).sum(1)
            c6 = (sigma ** 2 / r2) ** 3
            c6[r2 > rc ** 2] = 0.0
            c12 = c6 ** 2

            pairwise_energies = 4 * epsilon * (c12 - c6) - e0 * (c6 != 0.0)
            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies

            pairwise_forces = (-24 * epsilon * (2 * c12 - c6) / r2)[
                :, np.newaxis
            ] * distance_vectors

            forces[ii] += pairwise_forces.sum(axis=0)
            stresses[ii] += 0.5 * np.dot(
                pairwise_forces.T, distance_vectors
            )  # equivalent to outer product

            # add j < i contributions
            for jj, atom_j in enumerate(neighbors):
                energies[atom_j] += 0.5 * pairwise_energies[jj]
                forces[atom_j] += -pairwise_forces[jj]  # f_ji = - f_ij
                stresses[atom_j] += 0.5 * np.outer(
                    pairwise_forces[jj], distance_vectors[jj]
                )

        # whether or not output stress
        if atoms.number_of_lattice_vectors == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            stress = stresses / atoms.get_volume()
        else:
            stress = None

        energy = energies.sum()
        #print(energy)
        return energy, forces, stress