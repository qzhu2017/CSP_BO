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
import json

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
    vol1, eng1 = lat1.volume/sum(struc.numIons), struc.energy_per_atom
    pmg_s1 = struc.to_pymatgen()

    for i, ref in enumerate(ref_strucs):
        if ref.selected:
            lat2 = ref.lattice
            vol2, eng2 = lat2.volume/sum(struc.numIons), ref.energy_per_atom
            if abs(vol1-vol2)/vol1<5e-2 and abs(eng1-eng2)<2e-3:
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
    string += ' {:8.3f}'.format(l1.volume/sum(calc.numIons))

    if origin is not None:
        string += ' {:10s}'.format(origin)
        if calc.p_energy is not None:
            string += ' {:6.3f}'.format(eng-calc.p_energy)

    #if fitness is not None and fitness < 0.2 and l1.volume/sum(calc.numIons) > 30 or l1.volume/sum(calc.numIons) < 5:
    #if spg == 'F-43m' and l1.volume/sum(calc.numIons) > 30 or l1.volume/sum(calc.numIons) < 5:
    #    print(calc)
    #    print("===================", string)
    #    calc.to_ase().write('bug.vasp', format='vasp', vasp5=True, direct=True)
    #    import sys; sys.exit()

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
    t0 = time()
    if calculator == 'gulp':
        struc, energy, cputime, error = gulp_opt(model, ff, path='pyxtal_calc', pstress=pressure, adjust=True, clean=False)
    else:
        struc, energy, cputime, error = vasp_opt(model, path='pyxtal_calc', pstress=pressure, levels=[0,2,3], clean=False)
    #print("from main: ---------------", t0-time())
    #print("from calc: ---------------", cputime)
    if not error:
        struc.tag = origin
        if hasattr(model, 'p_energy'):
            struc.p_energy = model.p_energy
        else:
            struc.p_energy = np.nan
        struc.energy_per_atom = energy
        struc.energy = energy*sum(struc.numIons)
        struc.time = cputime#/60
    else:
        #import sys; sys.exit()
        struc = model 
        struc.energy = np.nan
        struc.energy_per_atom = np.nan

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
    def __init__(self, elements, sgs=range(2,231), pressure=0,
                 N_gen=10, N_pop=10, fracs=[0.3, 0.3, 0.3, 0.1], 
                 calculator='gulp', ff='tersoff', max_num=36, 
                 out_folder='results', permutation=None, ref=None, 
                 seeds=None, lat=None, opt_lat=True):

        self.elements = elements
        self.elements_set = set(elements.keys())
        self.varcomp = self.parse_varcomp()
        self.permutation = None
        if self.varcomp:
            if permutation is None:
                self.get_permutation()
            else:
                self.permutation = permutation
        self.seeds = seeds
        self.lat = lat
        if isinstance(sgs, range):
            sgs = list(sgs)
        self.sgs = sgs
        self.N_gen = N_gen
        self.N_pop = N_pop
        self.fracs = fracs
        self.all_strucs = []
        self.opt_lat = opt_lat
        self.calculator = calculator
        self.ff = ff
        self.max_num=max_num
        self.pressure = pressure
        self.out_folder = out_folder
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        if ref is not None:
            self.ref_pmg = self.parse_ref(ref)
            self.EA_ref()
        else:
            self.ref = None
        self.save(self.out_folder + '/default.json')
        #self.predict()

    @classmethod
    def load(cls, filename=None):
        """
        load the calculations from a dictionary
        """
        with open(filename, "r") as fp:
            dicts = json.load(fp)

        elements = dicts["elements"]
        sgs = dicts["sgs"]
        pressure = dicts["pressure"]
        N_gen = dicts["N_gen"]
        N_pop = dicts["N_pop"]
        fracs = dicts["fracs"]
        calculator = dicts["calculator"]
        ff = dicts['ff']
        max_num = dicts["max_num"]
        out_folder = dicts["out_folder"]
        permutation = dicts["permutation"]
        return cls(elements, sgs, pressure, N_gen, N_pop, fracs,
            calculator, ff, max_num, out_folder, permutation)

    def _save_dict(self):
        dict0 = {"elements": self.elements,
                 "sgs": self.sgs,
                 "pressure": self.pressure,
                 "N_gen": self.N_gen,
                 "N_pop": self.N_pop,
                 "fracs": self.fracs,
                 "calculator": self.calculator,
                 "ff": self.ff,
                 "max_num": self.max_num,
                 "out_folder": self.out_folder,
                 "permutation": self.permutation,
                }
        return dict0

    def save(self, filename):
        dict0 = self._save_dict()
        with open(filename, "w") as fp:
            json.dump(dict0, fp)
        print("save the calculation to", filename)


    def get_permutation(self):
        permutation = {}
        for ele1 in self.elements_set:
            for ele2 in self.elements_set:
                if ele1 != ele2:
                    permutation[ele1] = ele2
        self.permutation = permutation

       
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
                    model.p_energy = np.nan
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
        model.p_energy = np.nan
        calc = opt_struc(model, calculator=self.calculator, ff=self.ff)
        self.ref = calc  # pyxtal format

    def EA_random(self):
        model = from_random(elements, self.sgs, self.lat)
        model.tag = 'Random'
        model.p_energy = np.nan
        print("Generate structure from random: ", model.formula)
        return model

    def EA_heredity(self):
        raise NotImplementedError("Heredity is not supported yet") 

    def EA_mutation(self):
        while True:
            id = self.selTournament()
            model = self.prev_strucs[id].copy()
            if model.group.number >=16:
                break
        #print(model)
        model.to_ase().write('bug.vasp', format='vasp', vasp5=True, direct=True)
        max_cell = int(self.max_num/sum(model.numIons))
        if max_cell < 1:
            max_cell = 1
        elif max_cell > 4:
            max_cell = 4
        g_type = 'k+t'
        #try:
        if not self.varcomp:
            struc = model.subgroup_once(0.2, None, None, g_type, max_cell)
        else:
            if len(model.species) == 1:
                permutation = self.permutation
                while True:
                    struc = model.subgroup_once(0.01, None, permutation, max_cell=max_cell)
                    if len(struc.numIons) > 1:
                        break
            else:
                while True:
                    if np.random.random()>0.25:
                        permutation = self.permutation
                        struc = model.subgroup_once(0.01, None, permutation, max_cell=max_cell)
                    else:
                        struc = model.subgroup_once(0.2, None, None, g_type, max_cell)
                    if len(struc.numIons) > 1:
                        break
        struc.tag = 'Mutation'
        struc.p_energy = model.energy_per_atom
        print("Generate structure from mutation: ", struc.formula, "with parent", model.formula)
        return struc
        #except:
        #    print("cannot do the subgroup mutation, switch to random")
        #    model.to_ase().write('bug.vasp', format='vasp', vasp5=True, direct=True)
        #    print(model)
        #    print(max_cell)
        #    struc = model.subgroup(once=True, eps=0.1, group_type=g_type, max_cell=max_cell)
        #    import sys; sys.exit()
        #    return self.EA_random()

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
                    model = self.EA_random()
                    current_strucs.append(model)
            else:
                N_pops = [int(self.N_pop*i) for i in self.fracs]
                for sub_pop in range(N_pops[0]):
                    model = self.EA_random()
                    current_strucs.append(model)

                for sub_pop in range(N_pops[1]):
                    model = self.EA_mutation()
                    current_strucs.append(model)

                for sub_pop in range(N_pops[2]):
                    #id = self.selTournament(fitnesses)
                    #model = prev_strucs[id].copy()
                    #model.tag = 'Heredity'
                    #model.p_energy = model.energy_per_atom
                    current_strucs.append(model)

                for sub_pop in range(N_pops[3]):
                    id = rank[sub_pop]
                    model = self.prev_strucs[id].copy()
                    model.tag = 'KeptBest'
                    model.p_energy = model.energy_per_atom
                    current_strucs.append(model)

            for model in current_strucs:
                model.done = False
                model.selected = False
                model.generation = gen
                model.energy = np.nan
                model.energy_per_atom = np.nan
                model.fitness = np.nan
                model.count = 1

            self.dump(current_strucs, self.out_folder+'/working.db', 'w')

            # Local optimization
            for i, model in enumerate(current_strucs):
                if not model.done:
                    opt_model = opt_struc(model, self.sgs, self.elements, \
                    pressure=self.pressure, calculator=self.calculator, ff=self.ff)
                    opt_model.done = True
                    opt_model.selected = False
                    opt_model.generation = model.generation
                    opt_model.fitness = np.nan
                    opt_model.count = 1
                    # Keep only well relaxed structures
                    if opt_model.energy is not np.nan:
                        id = new_struc(opt_model, current_strucs)
                        if id is None:
                            opt_model.selected = True
                            current_strucs[i] = opt_model
                            print(detail(opt_model, header='{:<4d}'.format(i), \
                            cputime=opt_model.time, origin=opt_model.tag))
                        else:
                            current_strucs[id].count += 1
                            print(i, "skip the duplicate structure", opt_model.formula)

                    # save this for every update
                    self.update(i, opt_model, self.out_folder+'/working.db')
            
            opt_models = [model for model in current_strucs if model.selected]

            # compute the fitness for the left structures
            self.fitnesses = self.compute_fitness(opt_models)
            rank = np.argsort(self.fitnesses)

            # print generation summary
            print('Generation {:d} finishes'.format(gen))
            if self.varcomp:
                print('The current convex hull')
                #print(self.end_members)
                print(self.ref_hull)
                for i in range(len(rank)):
                    model = opt_models[rank[i]]
                    fitness = self.fitnesses[rank[i]]
                    print(detail(model, fitness=fitness, origin=model.tag))
            else:
                id = rank[0]
                model = opt_models[id]
                origin = model.tag
                header = 'Best: {:4d} '.format(gen)
                print(detail(model, header=header, origin=origin))

            self.prev_strucs = opt_models

            # save all_strucs every generation
            new_models = []
            for opt_model in opt_models:
                id = new_struc(opt_model, self.all_strucs)
                if id is None:
                    new_models.append(opt_model)
                else:
                    self.all_strucs[id].count += 1

            if len(self.all_strucs) == 0:
                self.dump(new_models, self.out_folder+'/all.db', 'w')
            else:
                self.dump(new_models, self.out_folder+'/all.db', 'a+')
            self.all_strucs.extend(new_models)

        print('\n The low energy structures are :')
        self.extract_good_structures()

    def selTournament(self, factor=0.4):
        """
        Select the best individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        """
        fitness = self.fitnesses
        IDs = sample(set(range(len(fitness))), int(len(fitness)*factor))
        min_fit = np.argmin(fitness[IDs])
        return IDs[min_fit]

    def extract_good_structures(self, all_strucs=None):
        """
        extract the low-enenrgy structures from the global search
        """
        print("=====extract_good_structures======")

        if all_strucs is None:
            all_strucs = self.all_strucs

        fits = self.compute_fitness(all_strucs)
        med_fit = np.median(fits)
        ranks = np.argsort(fits)

        for i in range(len(ranks)):
            ID = ranks[i]
            all_strucs[ID].fitness = fits[ID]
            count = all_strucs[ID].count
            print(detail(all_strucs[ID], count=count, fitness=fits[ID]))
            #print(detail(all_strucs[ID], fitness=fits[ID]))
            if fits[ID] >= med_fit + 0.5:
                break

    def update(self, id, struc, filename):
        """
        overwrite the existing atoms entry
        """
        kvp = {"spg": struc.group.symbol,
               "eng_per_at": struc.energy_per_atom,
               "eng": struc.energy,
               "done": struc.done,
               "selected": struc.selected,
               "origin": struc.tag,
               "pressure": self.pressure if self.pressure is not None else np.nan,
               "fitness": struc.fitness,
               "generation": struc.generation,
               "count": struc.count,
              }

        # the index starts from 1 in ase db 
        with connect(filename) as db:
            db.write(struc.to_ase(), id=id+1, key_value_pairs=kvp)

    def dump(self, strucs, filename, permission):
        """
        save the low-energy structures to database format
        Args: 
            strucs:
            filename:
            permission:
        """
        if permission == 'w' and os.path.exists(filename):
            os.remove(filename)
        with connect(filename) as db:
            for i, struc in enumerate(strucs):
                kvp = {"spg": struc.group.symbol,
                       "eng_per_at": struc.energy_per_atom,
                       "eng": struc.energy,
                       "done": struc.done,
                       "selected": struc.selected,
                       "origin": struc.tag,
                       "pressure": self.pressure if self.pressure is not None else np.nan,
                       "fitness": struc.fitness,
                       "generation": struc.generation,
                       "count": struc.count,
                      }
                db.write(struc.to_ase(), key_value_pairs=kvp)

    def load_all_strucs(self):
        filename = self.out_folder + "/all.db"
        print("load the structures from ", filename)

        all_strucs = []
        with connect(filename) as db:
            for row in db.select():
                s = db.get_atoms(id=row.id)
                struc = pyxtal()
                struc.from_seed(s)
                struc.energy = row.eng
                struc.energy_per_atom = row.eng_per_at
                struc.selected = row.selected
                struc.origin = row.origin
                struc.fitness = row.fitness
                struc.count = row.count
                all_strucs.append(struc)
                if row.id % 20 == 0:
                    print(row.id, row.spg)
        return all_strucs


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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        dest="run",
        action = 'store_true',
        default = False,
        help="run real calculation"
    )
    parser.add_argument(
        "-e",
        dest="extract",
        action = 'store_true',
        default = False,
        help="extract good structures",
    )
    parser.add_argument(
        "-t",
        dest="test",
        action = 'store_true',
        default = False,
        help="test",
    )
    parser.add_argument(
        "-f",
        dest="file",
        help="file to load the calculation",
    )

    
    options = parser.parse_args()
    if options.run:
        elements = {"C": [1,8], "Si": [1,8]}
        #calc = EA(elements, seeds='my.db', N_pop=30, N_gen=2, max_num=20, fracs=[0.1, 0.8, 0.0, 0.1])
        calc = EA(elements, seeds='my.db', N_pop=5, N_gen=2, max_num=10, fracs=[0.5, 0.4, 0.0, 0.1], calculator='vasp')
        #EA(elements, seeds='my.db', N_pop=20, pressure=10.0, calculator='vasp')
        calc.predict()
    elif options.test:
        # test element
        elements = {"C": 4}
        calc = EA(elements, N_pop=30, N_gen=5)
        calc.predict()

        # test element with seeds
        elements = {"Si": 8}
        calc = EA(elements, N_pop=10, N_gen=5, seeds='my.db')
        calc.predict()

        # test element with pressure
        calc = EA(elements, N_pop=10, N_gen=5, seeds='my.db', pressure=10.0)
        calc.predict()

        # test binary with pressure
        elements = {"C": [1,8], "Si": [1,8]}
        calc = EA(elements, seeds='my.db', N_pop=10, N_gen=5, max_num=10, fracs=[0.4, 0.5, 0.0, 0.1])
        calc.predict()

    elif options.extract:
        if options.file is None:
            raise RuntimeError("Needs to provide the calculation file")

        calc = EA.load(options.file)
        all_strucs = calc.load_all_strucs()
        calc.extract_good_structures(all_strucs)
