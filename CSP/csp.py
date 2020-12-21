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
    vol1, eng1 = lat1.volume, struc.energy_per_atom
    pmg_s1 = struc.to_pymatgen()

    for i, ref in enumerate(ref_strucs):
        lat2 = ref.lattice
        vol2, eng2 = lat2.volume, ref.energy_per_atom
        abc2 = np.sort(np.array([lat2.a, lat2.b, lat2.c]))
        if abs(vol1-vol2)/vol1<5e-2 and abs(eng1-eng2)<1e-2:
            pmg_s2 = ref.to_pymatgen()
            if sm.StructureMatcher().fit(pmg_s1, pmg_s2):
                return i
    return None


def detail(calc, header=None, cputime=None, count=None, origin=None, fitness=None):
    """
    dummy function to gather the output information for each structure
    Args:
        calc: PyXtal object
    """
    if header is None:
        string = ''
    else:
        string = header
    eng = calc.energy_per_atom
    l1 = calc.lattice
    a, b, c, alpha, beta, gamma = l1.get_para(degree=True)

    group = calc.group
    spg = group.symbol
    if group.number in [7, 14, 15]:
        if hasattr(calc, 'diag') and calc.diag and group.alias:
            spg = group.alias
        
    string += '{:10s} ['.format(spg)
    for sp, numIon in zip(calc.species, calc.numIons):
        string += '{:>2s}{:<3d}'.format(sp, numIon)
    string += ']'

    string += '{:8.3f} '.format(eng)

    if fitness is not None:
        string += ' {:6.3f}'.format(fitness)


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
        #np.random.RandomState()
        np.random.RandomState(1)
        sg = choice(sgs)
        species = []
        numIons = []
        for ele in elements.keys():
            species.append(ele)
            if isinstance(elements[ele], list):
                num = randint(elements[ele][0], elements[ele][1])
                numIons.append(num)
            else: #fixed composition
                numIons.append(elements[ele])
        if lat is not None:
            lat = lat.swap_axis(random=True)
            lat = lat.swap_angle()
        struc = pyxtal()
        #print(sg, species, numIons, lat)
        struc.from_random(3, sg, species, numIons, lattice=lat, force_pass=True)
        if struc.valid:
            return struc


def opt_struc(model, sgs=[4], elements={"C":4}, lat=None, pressure=0.0,
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
    elif origin == "Seeds":
        struc = model
    else: #mutation
        try:
            max_cell = max([1, int(max_num/sum(model.numIons))])
            if len(model.species) == 1:
                struc = model.subgroup_with_substitution({"Si":"C", "C":"Si"}, once=True, max_cell=max_cell)
            else:
                g_type = choice(['k', 't'])
                struc = model.subgroup(once=True, eps=0.1, group_type=g_type, max_cell=max_cell)
        except:
            struc = from_random(elements, sgs, lat)
            origin = 'Random'

    #try:
    t0 = time()
    if calculator == 'gulp':
        struc, energy, cputime, error = gulp_opt(struc, ff, path='pyxtal_calc', pstress=pressure, adjust=True, clean=False)
    else:
        struc, energy, cputime, error = vasp_opt(struc, path='pyxtal_calc', pstress=pressure, levels=[0,2,3], clean=False)
    #print("from main: ---------------", t0-time())
    #print("from calc: ---------------", cputime)
    if not error:
        struc.tag = origin
        if hasattr(model, 'p_energy'):
            struc.p_energy = model.p_energy
        else:
            struc.p_energy = None
        struc.energy_per_atom = energy
        struc.energy = energy*sum(struc.numIons)
        struc.time = cputime#/60
        #import sys; sys.exit()
    else:
        struc = model 
        struc.energy = None
        struc.energy_per_atom = None

    return struc

class EA():
    """
    A simple EA class to perform global structure search
    Random search is also possible

    Args:
        elements: {"C": [4, 6]}
        sgs: list of allowed space groups
        lat: lattice for initial structure
        N_gen: number of generations
        N_pop: number of populations
        fracs: []
        random: switch to random search? 
        opt_lat: do not optimize the lattice?
        calculator: gulp, vasp, or pyxtal_ff later
        ff: force field for gulp
        max_num: max_number of atoms in a structure
        seeds: None or string to store the ase database file
    """
    def __init__(self, elements, sgs=range(2,231), pressure=None,
                 lat = None, N_gen=10, N_pop=10, fracs=[0.3, 0.3, 0.3, 0.1], 
                 random=True, ref=None, seeds=None, opt_lat=True, 
                 calculator='gulp', ff='tersoff', max_num=36):
        self.elements = elements
        self.elements_set = set(elements.keys())
        self.varcomp = self.parse_varcomp()
        self.seeds = seeds
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
        self.max_num=max_num
        self.pressure = pressure
        if random:
            self.fracs = [0.7, 0.3, 0.0, 0.0]
        if ref is not None:
            self.ref_pmg = self.parse_ref(ref)
            self.EA_ref()
        else:
            self.ref = None

        self.predict()

    def parse_varcomp(self):
        if len(self.elements.keys())>1:
            for key in self.elements.keys():
                if isinstance(self.elements[key], list):
                    return True
        return False

    def parse_ref(self, ref_path):
        """
        parse the reference structure
        Expect to return a pymatgen structure object
        """
        if isinstance(ref_path, str):
            return mg.Structure.from_file(ref_path)
        else:
            return ref_path

    def get_seeds(self):
        from ase.db import connect
        print("Extracting the seed structures from ", self.seeds)
        models = []
        with connect(self.seeds) as db:
            for row in db.select():
                struc = db.get_atoms(id=row.id)
                eles = set(struc.get_chemical_symbols())
                if len(struc)<=self.max_num and eles.issubset(self.elements_set):
                    model = pyxtal()
                    model.from_seed(struc)
                    model.tag = 'Seeds'
                    model.p_energy = None
                    models.append(model)
        print("{:d} structures have been loaded".format(len(models)))
        return models

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
            current_strucs = []
            if gen == 0: # or self.random:
                if self.seeds is not None:
                    current_strucs.extend(self.get_seeds())
                for pop in range(self.N_pop):
                    model = pyxtal()
                    model.tag = 'Random'
                    model.p_energy = None
                    current_strucs.append(model)
                #print(len(current_strucs), "SSSSSSSSSS")
                #import sys; sys.exit()
            else:
                N_pops = [int(self.N_pop*i) for i in self.fracs]
                for sub_pop in range(N_pops[0]):
                    model = pyxtal()
                    model.tag = 'Random'
                    model.p_energy = None
                    current_strucs.append(model)

                for sub_pop in range(N_pops[1]):
                    id = self.selTournament(fitnesses)
                    model = prev_strucs[id].copy()
                    model.tag = 'Mutation'
                    model.p_energy = model.energy_per_atom
                    current_strucs.append(model)

                for sub_pop in range(N_pops[2]):
                    id = self.selTournament(fitnesses)
                    model = prev_strucs[id].copy()
                    model.tag = 'Heredity'
                    model.p_energy = model.energy_per_atom
                    current_strucs.append(model)

                for sub_pop in range(N_pops[3]):
                    id = rank[sub_pop]
                    model = prev_strucs[id].copy()
                    model.tag = 'KeptBest'
                    model.p_energy = model.energy_per_atom
                    current_strucs.append(model)

            # Local optimization
            opt_models = []
            for i, model in enumerate(current_strucs):
                opt_model = opt_struc(model, self.sgs, self.elements, \
                pressure=self.pressure, calculator=self.calculator, ff=self.ff)
                # Keep only well relaxed structures
                if opt_model.energy is not None:
                    id = new_struc(opt_model, opt_models)
                    if id is None:
                        opt_models.append(opt_model)
                        print(detail(opt_model, header='{:<4d}'.format(i), cputime=opt_model.time, origin=opt_model.tag))
                        self.all_strucs.append(opt_model.copy())
            
            # compute the fitness for the left structures
            fitnesses = self.compute_fitness(opt_models)
            rank = np.argsort(fitnesses)

            # print generation summary
            print('Generation {:d} finishes'.format(gen))
            if self.varcomp:
                print('The current convex hull')
                #print(self.end_members)
                print(self.ref_hull)
                for i in range(len(rank)):
                    model = opt_models[rank[i]]
                    fitness=fitnesses[rank[i]]
                    print(detail(model, fitness=fitness, origin=model.tag))
            else:
                id = rank[0]
                model = opt_models[id]
                origin = model.tag
                header = 'Best: {:4d} '.format(gen)
                print(detail(model, header=header, origin=origin))

            prev_strucs = opt_models

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
            engs.append(struc.energy_per_atom)
        engs = np.array(engs)
        med_eng = np.median(engs)
        ids = np.argsort(engs)
        good_structures = [self.all_strucs[ids[0]]]
        counts = [1]
        for i in ids:
            if engs[i] > med_eng + 2.0:
                break
            else:
                struc = self.all_strucs[i]
                id = new_struc(struc, good_structures)
                if id is None:
                    good_structures.append(struc)
                    counts.append(1)
                else:
                    counts[id] += 1
        if self.varcomp:
            fitnesses = self.compute_fitness(good_structures)
            ranks = np.argsort(fitnesses)
            for i in range(len(ranks)):
                ID = ranks[i]
                print(detail(good_structures[ID], count=counts[ID], fitness=fitnesses[ID]))
        else:
            for struc, count in zip(good_structures, counts):
                print(detail(struc, count=count))
        return good_structures

    def dump(self):
        """
        save the low-energy structures to database format
        """
        with connect(self.dump_file) as db:
            for struc in self.good_structures:
                kvp = {"spg": struc.group.symbol,
                       "ave_energy": struc.energy_per_atom, 
                       "tot_energy": struc.energy, 
                       "pressure": self.pressure,
                      }

                db.write(struc.to_ase(), key_value_pairs=kvp)

    def compute_fitness(self, models):
        """
        compute the fitness 
        """
        from ase.phasediagram import PhaseDiagram
    
        fitnesses = []
        if self.varcomp:
            if hasattr(self, 'ref_hull'):
                refs = self.ref_hull
                base = len(refs)
            else:
                refs = []
                base = 0
            for model in models:
                refs.append((model.to_ase().get_chemical_formula(), model.energy))
            #print(refs)
            pd = PhaseDiagram(refs, verbose=False)
            fitness = []
            for i, pt in enumerate(pd.points[base:]):
                E_ref, _, _ = pd.decompose(refs[i+base][0])
                model = models[i]
                fitness = (model.energy - E_ref)/sum(model.numIons)
                fitnesses.append(fitness)
            
            self.ref_hull = [refs[i] for i, r in enumerate(pd.hull) if r]
        else:
            fitnesses = [model.energy_per_atom for model in models]

        return np.array(fitnesses)


#elements = {"B": 28}
#EA(elements, sgs=[229, 225], N_pop=20, calculator='vasp')
elements = {"C": [1,8], "Si": [1,8]}
EA(elements, seeds='my.db', N_pop=30, N_gen=1, pressure=10.0, calculator='gulp', max_num=18)
#EA(elements, seeds='my.db', N_pop=20, pressure=10.0, calculator='vasp')
