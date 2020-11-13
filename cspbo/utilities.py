import numpy as np
import cupy as cp
from ase.neighborlist import neighbor_list
from functools import partial
from multiprocessing import Pool, cpu_count
from ase.db import connect
#from .descriptors.rdf import RDF
#from pymatgen.io.ase import AseAtomsAdaptor
from pyxtal.database.element import Element
from pyxtal.crystal import random_crystal
from random import choice
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def PyXtal(sgs, species, numIons):
    """ 
    PyXtal interface for the followings,

    Parameters
    ----------
        sg: a list of allowed space groups, e.g., range(2, 231)
        species: a list of chemical species, e.g., ["Na", "Cl"]
        numIons: a list to denote the number of atoms for each speice, [4, 4]
    Return:
        the pyxtal structure
    """
    while True:
        struc = random_crystal(choice(sgs), species, numIons)
        if struc.valid:
            return struc.to_ase()
 

def new_pt(data, Refs, d_tol=1e-1, eps=1e-8):
    (X, ele) = data
    X = X/(np.linalg.norm(X)+eps)
    for Ref in Refs:
        (X1, ele1) = Ref
        if ele1 == ele:
            X1 = X1/np.linalg.norm(X1+eps)
            d = X@X1.T
            if 1-d**2 < d_tol:
                return False
    return True
 
def Cosine(Rij, Rc):
    # Rij is the norm 
    ids = (Rij > Rc)
    result = 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)
    result[ids] = 0
    return result

def rmse(true, predicted):
    """ Calculate root mean square error of energy or force. """
    true, predicted = np.array(true), np.array(predicted)
    return np.sqrt(sum((true-predicted) ** 2 /len(true)))

def mae(true, predicted):
    """ Calculate mean absolute error of energy or force. """
    true, predicted = np.array(true), np.array(predicted)
    return sum(abs(true-predicted)/len(true))

def r2(true, predicted):
    """ Calculate the r square of energy or force. """
    true, predicted = np.array(true), np.array(predicted)
    t_bar = sum(true)/len(true)
    square_error = sum((true-predicted) ** 2)
    true_variance = sum((true-t_bar) ** 2) + 1e-8
    return 1 - square_error / true_variance

def metrics(y_train, y_test, y_train_pred, y_test_pred, header):
    r2_train = 'R2 {:6.4f}'.format(r2(y_train, y_train_pred))
    r2_test  = 'R2 {:6.4f}'.format(r2(y_test, y_test_pred))
    mae_train  = 'MAE {:6.3f}'.format(mae(y_train, y_train_pred))
    mae_test   = 'MAE {:6.3f}'.format(mae(y_test, y_test_pred))
    rmse_train = 'RMSE {:6.3f}'.format(rmse(y_train, y_train_pred))
    rmse_test  = 'RMSE {:6.3f}'.format(rmse(y_test, y_test_pred))
    str1 = "{:s} Train[{:4d}]: {:s} {:s} {:s}".format(\
            header, len(y_train), r2_train, mae_train, rmse_train)
    str2 = "{:s} Test [{:4d}]: {:s} {:s} {:s}".format(\
            header, len(y_test), r2_test, mae_test, rmse_test)
    print(str1)
    print(str2)
    return (str1, str2)

def metric_single(y_train, y_train_pred, header, show_max=False):
    r2_train = 'R2 {:6.4f}'.format(r2(y_train, y_train_pred))
    mae_train  = 'MAE {:6.3f}'.format(mae(y_train, y_train_pred))
    rmse_train = 'RMSE {:6.3f}'.format(rmse(y_train, y_train_pred))
    str1 = "{:s} [{:4d}]: {:s} {:s} {:s}".format(\
            header, len(y_train), r2_train, mae_train, rmse_train)
    if show_max:
        max_diff = np.max(np.abs(y_train_pred-y_train))
        str1 += '  Max {:6.4f}'.format(max_diff)
    print(str1)
    return str1

def build_desc(method='SO3', rcut=5.0, lmax=4, nmax=4, alpha=2.0):
    #from pyxtal_ff.descriptors.ACSF import ACSF
    #from pyxtal_ff.descriptors.EAMD import EAMD
    if method == "SO3":
        from pyxtal_ff.descriptors.SO3 import SO3
        des = SO3(nmax=nmax, lmax=lmax, rcut=rcut, alpha=alpha, derivative=True, stress=True)
    elif method == "SO4":
        from pyxtal_ff.descriptors.SO4 import SO4_Bispectrum
        des = SO4_Bispectrum(lmax=lmax, rcut=rcut, derivative=True, stress=True) 

    return des

def convert_train_data(data, des,  N_force=100000):
    """
    Nmax: Maximum number of force data
    """
    energy_data = []
    force_data = []
    db_data = []
    xs_added = []

    for _data in data:
        (struc, energy, forces) = _data
        d = des.calculate(struc) 
        ele = [Element(ele).z for ele in d['elements']]
        ele = np.array(ele)
        f_ids = []
        for i in range(len(struc)):
            if len(force_data) < N_force:
                ids = np.argwhere(d['seq'][:,1]==i).flatten()
                _i = d['seq'][ids, 0] 
                if len(xs_added) == 0:
                    force_data.append((d['x'][_i,:], d['dxdr'][ids], forces[i], ele[_i]))
                    f_ids.append(i)
                else:
                    if new_pt((X, ele[i]), xs_added):
                        force_data.append((d['x'][_i,:], d['dxdr'][ids], forces[i], ele[_i]))
                        f_ids.append(i)
                        xs_added.append((X, ele[i]))

        energy_data.append((d['x'], energy/len(struc), ele)) 
        db_data.append((struc, energy, forces, True, f_ids))

    train_data = {"energy": energy_data, "force": force_data, "db": db_data}
    return train_data


def get_data(db_name, des, N_force=100000, lists=None, select=False, ncpu=1):
    """
    Nmax: Maximum number of force data
    """
    X, Y, structures = convert_struc(db_name, des, lists, ncpu=ncpu)
    print('\n')
    energy_data = []
    force_data = []
    db_data = []

    for id in range(len(X)):
        ele = [Element(ele).z for ele in X[id]['elements']]
        ele = np.array(ele)
        energy_data.append((X[id]['x'], Y["energy"][id]/len(X[id]['x']), ele)) 
        if select:
            ids = [0] #[choice(range(len(X[id]['x'])))]
        else:
            ids = range(len(X[id]['x']))
        f_ids = []
        for i in ids:
            if len(force_data) < N_force:
                ids = np.argwhere(X[id]['seq'][:,1]==i).flatten()
                _i = X[id]['seq'][ids, 0] 
                force_data.append((X[id]['x'][_i,:], X[id]['dxdr'][ids], Y['forces'][id][i], ele[_i]))
                f_ids.append(i)

        db_data.append((structures[id], Y['energy'][id], Y['forces'][id], True, f_ids))
    train_data = {"energy": energy_data, "force": force_data, "db": db_data}
    return train_data



#def convert_rdf(db_file, N=None):
#
#    train_Y, ds = [], []
#    with connect(db_file) as db:
#        for row in db.select():
#            s = db.get_atoms(id=row.id)
#            if hasattr(row, 'ff_energy'):
#                eng = row.ff_energy
#            else:
#                eng = row.data.energy/len(s)
#            train_Y.append(eng)
#            pmg_struc = AseAtomsAdaptor().get_structure(s)
#            ds.append(RDF(pmg_struc, R_max=10).RDF[1])
#            if N is not None and len(train_Y) == N:
#                break
#    return ds, np.array(train_Y)

#def smear(data, sigma=0.2):
#    """
#    Apply Gaussian smearing to spectrum y value.
#    Args:
#        sigma: Std dev for Gaussian smear function
#    """
#    diff = [data[0, i + 1] - data[0, i] for i in range(np.shape(data)[0] - 1)]
#    avg_x_per_step = np.sum(diff) / len(diff)
#    data[1, :] = gaussian_filter1d(data[1, :], sigma / avg_x_per_step)
#    return data
#
#def get_rdf(s, r_min=0.5, r_max=8.0, N_bins=40, sigma=0.2):
#    # plot atomic RDF
#    # needs a cutoff
#    rdf = np.zeros([len(s)+2, N_bins]) 
#    _is, _js, _ds = neighbor_list('ijd', s, rcut)
#    dr = (r_max-r_min)/(N_bins-1)
#    bins = np.arange(r_min, r_max, R_bin)
#    cutoff = Cosine(neighbors, bins)
#
#    for i in range(len(s)):
#        neighbors = _ds[_is == i]
#        des = np.histogram(neighbors, bins=bins)
#        rdf[i, :] = smear(np.vstack(bins, des), sigma)[1, :]
#    rdf /= s.get_volume()
#    rdf[-2,:] = bins
#    rdf[-1,:] = Cosine(bins, r_max)
#    return rdf

def get_2b(s, rcut=4.0, kernel='all'):
    
    _is, _js, _ds = neighbor_list('ijd', s, rcut)
    if kernel == 'atom':
        des_2b = np.zeros([len(s), 30, 2])
        for i in range(len(s)):
            neighbors = _ds[_is == i]
            cutoff = Cosine(neighbors, rcut)
            des_2b[i, :len(neighbors), 0] = neighbors 
            des_2b[i, :len(neighbors), 1] = cutoff
        return des_2b
    else:
        return (np.vstack((_ds, Cosine(_ds, rcut))), len(s))

def get_train_data(db_file, include_stress=False):
    strucs = []
    energies = []
    forces = []
    stresses = []
    with connect(db_file) as db:
        for row in db.select():
            s = db.get_atoms(id=row.id)
            strucs.append(s)
            energies.append(row.data.energy)
            forces.append(np.array(row.data.force))
            if include_stress:
                stress.append(np.array(row.data.stress))

    if include_stress:
        return (strucs, energies, forces, stress)
    else:
        return (strucs, energies, forces)


def convert_struc(db_file, des, ids=None, N=None, ncpu=1, stress=False):
    structures, train_Y = [], {"energy":[], "forces": [], "stress": []}
    with connect(db_file) as db:
        for row in db.select():
            include = True
            if (ids is not None) and (row.id-1 not in ids):
                include = False
            if include:
                s = db.get_atoms(id=row.id)
                train_Y['energy'].append(row.data.energy)
                train_Y['forces'].append(row.data.force)
                if stress:
                    train_Y['stress'].append(row.data.stress)
                structures.append(s)

                if N is not None and len(structures) == N:
                    break

    if (ncpu == 1) or (len(structures)==1):
        xs = []
        for i, struc in enumerate(structures):
            strs = '\rDescriptor calculation: {:4d} out of {:4d}'.format(i+1, len(structures))
            print(strs, flush=False, end='')
            d = des.calculate(struc) 
            xs.append(d)
    else:
        print('---Parallel mode is on, {} cores with be used'.format(ncpu))
        import tqdm
        tqdm_func = tqdm.tqdm
        structures0 = tqdm_func(list(structures), desc='running')

        with Pool(ncpu) as p:
            func = partial(fea, des)
            xs = p.map(func, structures0)
            p.close()
            p.join()
    train_x = xs 

    return train_x, train_Y, structures

def get_strucs(db_file, N_max=None):
    structures = []
    values = []
    with connect(db_file) as db:
        for row in db.select():
            s = db.get_atoms(id=row.id)
            structures.append(s)
            E = row.data.energy
            F = np.array(row.data.force)
            if "stress" in row.data.keys():
                S = np.array(row.data.stress)
            else:
                S = None
            values.append((E, F, S))
            if (N_max is not None) and (len(values) == N_max):
                break
    return structures, values


def fea(des, struc):
    #return des.calculate(struc)['x']
    return des.calculate(struc)

def normalize(x_train, x_test):
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)

def write_db_from_dict(data, db_filename='viz.db', permission='w'):
    if permission=='w' and os.path.exists(db_filename):
        os.remove(db_filename)

    N = len(data["atoms"])
    with connect(db_filename) as db:
        for i in range(N):
            kvp = {}
            for key in data.keys():
                if key == "atoms":
                    struc = data["atoms"][i]
                else:
                    kvp[key] = data[key][i]
            db.write(struc, key_value_pairs=kvp)
    print("Saved all structure to", db_filename)

def write_db(data, db_filename='viz.db', permission='w'):
    from ase.db import connect
    import os
    if permission=='w' and os.path.exists(db_filename):
        os.remove(db_filename)

    (structures, y_qm, y_ml) = data
    with connect(db_filename) as db:
        print("writing data to db: ", len(structures))
        for i, x in enumerate(structures):
            kvp = {"QM_energy": y_qm[i], 
                   "ML_energy": y_ml[i], 
                   "diff_energy": abs(y_qm[i]-y_ml[i])}
            db.write(x, key_value_pairs=kvp)
    
def plot(Xs, Ys, labels, figname='results.png'):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    for x, y, label in zip(Xs, Ys, labels):
        plt.scatter(x, y, alpha=0.8, label=label, s=5)
    xs = np.linspace(np.min(x)-0.1, np.max(x)+0.1, 100)
    plt.plot(xs, xs, 'b')
    plt.plot(xs, xs+0.10, 'g--')
    plt.plot(xs, xs-0.10, 'g--')
    plt.xlabel('QM (eV)') 
    plt.ylabel('ML (eV)')
    plt.legend()
    plt.tight_layout() 
    plt.savefig(figname)
    plt.close()
    print("save the figure to ", figname)

def regression(method, data, layers):
    (x_train0, y_train, x_test0, y_test) = data

    if method == "GPR":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, WhiteKernel
        para = (0.5, 0.5)
        print("\nGPR with Matern: ", para)
        kernel = Matern(length_scale=para[0], nu=para[1]) #+ WhiteKernel()
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp.fit(x_train0, y_train)
        y_train_pred = gp.predict(x_train0)
        y_test_pred = gp.predict(x_test0)
        labels = metrics(y_train, y_test, y_train_pred, y_test_pred, "GP")
    else:
        from sklearn.neural_network import MLPRegressor
        h = []
        for item in layers.split(','):
            h.append(int(item))
        print("\nNN with hidden layer size", h)
        mlp = MLPRegressor(hidden_layer_sizes=h, max_iter=20000, solver="lbfgs", alpha=0)
        mlp.fit(x_train0, y_train)
        y_train_pred = mlp.predict(x_train0)
        y_test_pred = mlp.predict(x_test0)
        labels=  metrics(y_train, y_test, y_train_pred, y_test_pred, "NN")
    return y_train_pred, y_test_pred, labels

def plot(Xs, Ys, labels, figname='results.png', draw_line=True, type='Energy'):
    x_mins, x_maxs = [], []
    for x, y, label in zip(Xs, Ys, labels):
        plt.scatter(x, y, alpha=0.8, label=label, s=5)
        x_mins.append(np.min(x))
        x_maxs.append(np.max(x))
    xs = np.linspace(min(x_mins)-0.1, max(x_maxs)+0.1, 100)
    if draw_line:
        plt.plot(xs, xs, 'g--', alpha=0.5)
        #plt.plot(xs, xs+0.10, 'g--')
        #plt.plot(xs, xs-0.10, 'g--')
        plt.xlim(min(x_mins)-0.1, max(x_maxs)+0.1)
        plt.ylim(min(x_mins)-0.1, max(x_maxs)+0.1)
    if type == "Energy":
        unit = "(eV/atom)"
    elif type == 'Force':
        unit = "(eV/A)"
    elif type == 'Stress':
        unit = "GPa"
    plt.xlabel('True' + unit) 
    plt.ylabel('Prediction' + unit)
    plt.legend(loc=2)
    plt.tight_layout() 
    plt.savefig(figname)
    plt.close()
    print("save the figure to ", figname)


def write_db(data, db_filename='viz.db', permission='w'):
    import os
    if permission=='w' and os.path.exists(db_filename):
        os.remove(db_filename)

    (structures, y_qm, y_ml) = data
    with connect(db_filename) as db:
        print("writing data to db: ", len(structures))
        for i, x in enumerate(structures):
            kvp = {"QM_energy": y_qm[i], 
                   "ML_energy": y_ml[i], 
                   "diff_energy": abs(y_qm[i]-y_ml[i])}
            db.write(x, key_value_pairs=kvp)
 
def plot_two_body(model, figname, rs=[1.0, 5.0]):
    from ase import Atoms
    from cspbo.calculator import GPR
    
    rs = np.linspace(rs[0], rs[1], 50)
    cell = 10*np.eye(3)
    dimers = [Atoms("2Si", positions=[[0,0,0], [r,0,0]], cell=cell) for r in rs]
    calc = GPR(ff=model, return_std=False)
    engs = []
    for dimer in dimers:
        dimer.set_calculator(calc)
        engs.append(dimer.get_potential_energy())

    plt.plot(rs, engs, '-d', label='2-body')
    plt.legend()
    plt.xlabel('R (Angstrom)')
    plt.ylabel('Energy (eV)')
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()
    print("save the figure to ", figname)

def list_to_tuple(force_data, stress=False, include_force=False):
    icol = 0
    for fd in force_data:
        icol += fd[0].shape[0]
    jcol = fd[0].shape[1]
    
    ELE = []
    indices = []
    F = []
    X = np.zeros([icol, jcol])
    if stress:
        length = 9
    else:
        length = 3

    if isinstance(force_data[0][1], np.ndarray):
        dXdR = np.zeros([icol, jcol, length])
    else: #on cuda
        dXdR = cp.zeros([icol, jcol, length])

    count = 0
    for fd in force_data:
        if include_force:
            (x, dxdr, f, ele) = fd
            F.append(f)
        else:
            (x, dxdr, ele) = fd
        shp = x.shape[0]
        indices.append(shp)
        X[count:count+shp, :jcol] = x
        dXdR[count:count+shp, :jcol, :] = dxdr
        ELE.extend(ele)
        count += shp
    ELE = np.ravel(ELE)
    if include_force:
        return (X, dXdR, ELE, indices, F)
    else:
        return (X, dXdR, ELE, indices)

def tuple_to_list(force_data):
    X, dXdR, ELE, indices = force_data
    X1 = []
    c = 0
    for ind in indices:
        X1.append((X[c:c+ind], dXdR[c:c+ind], ELE[c:c+ind])) 
        c += ind
    return X1


