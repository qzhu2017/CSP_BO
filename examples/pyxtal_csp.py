from pyxtal import pyxtal
from pyxtal.interface.vasp import optimize as vasp_opt
from pyxtal.interface.gulp import optimize as gulp_opt
import pymatgen.analysis.structure_matcher as sm
from ase.db import connect
from copy import deepcopy
import numpy as np
import os
from random import randint, choice, sample
from time import time
import warnings

warnings.filterwarnings("ignore")

def new_struc(struc, ref_strucs):
    """
    check if this is a new structure

    Args:
        struc: input structure
        ref_strucs: reference structure

    Return:
        id: `None` or the id (int) of matched structure
    """
    lat1 = struc.lattice
    vol1, eng1 = lat1.volume, struc.energy
    pmg_s1 = struc.to_pymatgen()

    for i, ref in enumerate(ref_strucs):
        lat2 = ref.lattice
        vol2, eng2 = lat2.volume, ref.energy
        abc2 = np.sort(np.array([lat2.a, lat2.b, lat2.c]))
        if abs(vol1-vol2)/vol1<5e-2 and abs(eng1-eng2)<1e-2:
            pmg_s2 = ref.to_pymatgen()
            if sm.StructureMatcher().fit(pmg_s1, pmg_s2):
                return i
    return None


def detail(calc, header=None, cputime=None, count=None, origin=None):
    """
    dummy function to gather the output information for each structure
    Args:
        calc: PyXtal object
    """

    eng = calc.energy
    l1 = calc.lattice
    a, b, c, alpha, beta, gamma = l1.get_para(degree=True)

    group = calc.group
    spg = group.symbol
    if group.number in [7, 14, 15]:
        if hasattr(calc, 'diag') and calc.diag and group.alias:
            spg = group.alias
        
    string = '{:10s} ['.format(spg)
    for sp, numIon in zip(calc.species, calc.numIons):
        string += '{:>2s}{:<3d}'.format(sp, numIon)
    string += ']'

    string += '{:8.3f} '.format(eng)

    if cputime is None:
        if count is not None:
            string += '({:2d}) '.format(count)
    else:
        string += '{:6.2f}'.format(cputime)

    string += ' {:6.3f} {:6.3f} {:6.3f}'.format(a, b, c)
    string += ' {:6.2f} {:6.2f} {:6.2f}'.format(alpha, beta, gamma)
    string += ' {:8.3f}'.format(l1.volume)

    if origin is not None:
        string += ' {:10s}'.format(origin)
        if calc.p_energy is not None:
            string += ' {:6.3f}'.format(eng-calc.p_energy)

    return string

def from_random(elements={"C":4}, sgs=[4], lat=None):
    while True:
        np.random.RandomState()
        sg = choice(sgs)
        species = []
        numIons = []
        for ele in elements.keys():
            species.append(ele)
            if len(elements[ele]) == 2:
                num = randint(elements[ele][0], elements[ele][1])
                numIons.append(num)
            else:
                numIons.append(elements[ele])
        if lat is not None:
            lat = lat.swap_axis(random=True)
            lat = lat.swap_angle()
        struc = pyxtal()
        #print(sg, species, numIons, lat)
        struc.from_random(3, sg, species, numIons, lattice=lat, force_pass=True)
        if struc.valid:
            return struc


def opt_struc(model, sgs=[4], elements={"C":4}, lat=None, 
            calculator='gulp', ff='tersoff', max_num=36):
    """
    Prepare and perform the structural relaxation for each individual
    Args:
        model: PyXtal object
        opt_lat: whether or not relax the lattice during optimization
        sgs: list of allows space group numbers, e.g., `[4, 7, 14]`
        lat: cell parameters for random structure generation, (default: `None`)
    """
    origin = model.tag

    if origin == "Random":
        struc = from_random(elements, sgs, lat)
    elif origin == "Heredity":
        raise NotImplementedError("Heredity is not supported yet") 
    else: #mutation
        try:
            g_type = choice(['k', 't'])
            max_cell = max([1, int(max_num/sum(model.numIons))])
            struc = model.subgroup(once=True, group_type=g_type, max_cell=max_cell)
        except:
            struc = from_random(elements, sgs, lat)
            origin = 'Random'

    #try:
    if calculator == 'gulp':
        strucs, energies, times, error = gulp_opt(struc, ff, path='pyxtal_calc')
    else:
        strucs, energies, times, error = vasp_opt(struc, path='pyxtal_calc')

    if not error:
        if calculator == 'gulp': 
            struc = strucs
            struc.energy = energies/sum(struc.numIons)
            cputime = times
        else:
            struc = strucs[-1]
            struc.energy = energies[-1]
            cputime = sum(times)
        struc.tag = origin
        if hasattr(model, 'p_energy'):
            struc.p_energy = model.p_energy
        else:
            struc.p_energy = None
        print(detail(struc, cputime=cputime, origin=origin))
    return struc
    #except:
    #    print("calculation is wrong")
    #    return model

class EA():
    """
    A simple EA class to perform global structure search
    Random search is also possible
    """
    def __init__(self, elements, sgs=range(2,231),
                 lat = None, N_gen=10, N_pop=10, fracs=[0.3, 0.3, 0.3, 0.1], 
                 random=True, ref=None, opt_lat=True, calculator='gulp',
                 ff='tersoff'):
        self.elements = elements
        self.lat = lat
        self.sgs = sgs
        self.N_gen = N_gen
        self.N_pop = N_pop
        self.fracs = fracs
        self.all_strucs = []
        self.opt_lat = opt_lat
        self.calculator = calculator
        self.dump_file = "pyxtal.db"
        self.ff = ff

        if random:
            self.fracs = [0.7, 0.3, 0.0, 0.0]
        if ref is not None:
            self.ref_pmg = self.parse_ref(ref)
            self.EA_ref()
        else:
            self.ref = None

        self.predict()

    def parse_ref(self, ref_path):
        """
        parse the reference structure
        Expect to return a pymatgen structure object
        """
        if isinstance(ref_path, str):
            return mg.Structure.from_file(ref_path)
        else:
            return ref_path

    def EA_ref(self):
        """
        convert the ref_struct to pyxtal format
        """
        model = pyxtal()
        model.from_seed(seed=self.ref_pmg)
        model.tag = 'Reference'
        model.p_energy = None
        calc = opt_struc(model, calculator=self.calculator, ff=self.ff)
        self.ref = calc  # pyxtal format

    def predict(self):
        """
        The main code to run EA prediction
        """
        for gen in range(self.N_gen):
            print('\nGeneration {:d} starts'.format(gen))
            current_strucs = {'models': [], 'rank': [], 'engs':[]}
            model = pyxtal()
            if gen == 0: # or self.random:
                for pop in range(self.N_pop):
                    model.tag = 'Random'
                    current_strucs['models'].append(model)
            else:
                N_pops = [int(self.N_pop*i) for i in self.fracs]
                for sub_pop in range(N_pops[0]):
                    id = prev_strucs['rank'][0]
                    model.tag = 'Random'
                    current_strucs['models'].append(model)

                for sub_pop in range(N_pops[1]):
                    id = self.selTournament(prev_strucs['engs'])
                    model = prev_strucs['models'][id].copy()
                    model.tag = 'Mutation'
                    model.p_energy = model.energy
                    current_strucs['models'].append(model)

                for sub_pop in range(N_pops[2]):
                    id = self.selTournament(prev_strucs['engs'])
                    model = prev_strucs['models'][id].copy()
                    model.tag = 'Heredity'
                    model.p_energy = model.energy
                    current_strucs['models'].append(model)

                for sub_pop in range(N_pops[3]):
                    id = prev_strucs['rank'][sub_pop]
                    model = prev_strucs['models'][id].copy()
                    model.tag = 'KeptBest'
                    model.p_energy = model.energy
                    current_strucs['models'].append(model)

            # Local optimization
            opt_models = []
            for model in current_strucs['models']:
                opt_model = opt_struc(model, self.sgs, self.elements, calculator=self.calculator, ff=self.ff)
                opt_models.append(opt_model)

            # Summary and Ranking
            for i, model in enumerate(opt_models):
                if hasattr(model, 'energy'):
                    current_strucs['engs'].append(model.energy)
                    self.all_strucs.append(model.copy())
                    current_strucs['models'][i] = model.copy()

            print('Generation {:d} finishes'.format(gen))
            current_strucs['engs'] = self.discard_similar(current_strucs)
            current_strucs['rank'] = np.argsort(current_strucs['engs'])
            id = current_strucs['rank'][0]
            model = current_strucs['models'][id]
            eng = model.energy
            origin = model.tag
            header = 'Best: {:4d} '.format(gen)
            print(detail(model, header=header, origin=origin))
            prev_strucs = deepcopy(current_strucs)

        print('\n The low energy structures are :')
        self.good_structures = self.extract_good_structures()
        self.dump()

    def selTournament(self, fitness, factor=0.4):
        """
        Select the best individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        """

        IDs = sample(set(range(len(fitness))), int(len(fitness)*factor))
        min_fit = np.argmin(fitness[IDs])
        return IDs[min_fit]

    def extract_good_structures(self):
        """
        extract the low-enenrgy structures from the global search
        """
        print("=====extract_good_structures======")
        engs = []
        for struc in self.all_strucs:
            engs.append(struc.energy)
        engs = np.array(engs)
        med_eng = np.median(engs)
        ids = np.argsort(engs)
        good_structures = [self.all_strucs[ids[0]]]
        counts = [1]
        for i in ids:
            if engs[i] > med_eng + 5.0:
                break
            else:
                struc = self.all_strucs[i]
                id = new_struc(struc, good_structures)
                if id is None:
                    good_structures.append(struc)
                    counts.append(1)
                else:
                    counts[id] += 1

        for struc, count in zip(good_structures, counts):
            print(detail(struc, count=count))
        return good_structures

    def discard_similar(self, current_strucs):
        """
        remove duplicate structures
        """
        engs = np.array(current_strucs['engs'])
        ids = np.argsort(engs)
        good_structures = [current_strucs['models'][ids[0]]]
        for i in ids[1:]:
            struc = current_strucs['models'][i]
            if hasattr(struc, "energy"): #
                id = new_struc(struc, good_structures)
                if id is None:
                    good_structures.append(struc)
                else:
                    engs[i] = 100000
            else:
                engs[i] = 100000
        return engs

    def dump(self):
        """
        save the low-energy structures to pkl format
        """
        with connect(self.dump_file) as db:
            for struc in self.good_structures:
                kvp = {"spg": struc.group.symbol,
                       "dft_energy": struc.energy, 
                      }
                db.write(struc.to_ase(), key_value_pairs=kvp)
 


elements = {"C": [4,6]}
#EA(elements, calculator='vasp')
EA(elements, calculator='gulp')
