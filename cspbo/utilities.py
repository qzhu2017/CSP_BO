import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
from ase.db import connect
from .descriptors.rdf import RDF
from pymatgen.io.ase import AseAtomsAdaptor
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def rmse(true, predicted):
    """ Calculate root mean square error of energy or force. """
    return np.sqrt(sum((true-predicted) ** 2 /len(true)))

def mae(true, predicted):
    """ Calculate mean absolute error of energy or force. """
    return sum(abs(true-predicted)/len(true))

def r2(true, predicted):
    """ Calculate the r square of energy or force. """
    t_bar = sum(true)/len(true)
    square_error = sum((true-predicted) ** 2)
    true_variance = sum((true-t_bar) ** 2)
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

def build_desc(method='SO3', rcut=5.0, lmax=4, nmax=4, alpha=2.0):
    #from pyxtal_ff.descriptors.ACSF import ACSF
    #from pyxtal_ff.descriptors.EAMD import EAMD
    if method == "SO3":
        from pyxtal_ff.descriptors.SO3 import SO3
        des = SO3(nmax=nmax, lmax=lmax, rcut=rcut, alpha=alpha, derivative=False)
    elif method == "SO4":
        from pyxtal_ff.descriptors.SO4 import SO4_Bispectrum
        des = SO4_Bispectrum(lmax=lmax, rcut=rcut, derivative=False) 

    return des

def convert_rdf(db_file, N=None):

    train_Y, ds = [], []
    with connect(db_file) as db:
        for row in db.select():
            s = db.get_atoms(id=row.id)
            if hasattr(row, 'ff_energy'):
                eng = row.ff_energy
            else:
                eng = row.data.energy/len(s)
            train_Y.append(eng)
            pmg_struc = AseAtomsAdaptor().get_structure(s)
            ds.append(RDF(pmg_struc, R_max=10).RDF[1])
            if N is not None and len(train_Y) == N:
                break
    return ds, np.array(train_Y)


def convert_struc(db_file, des, N=None, ncpu=1):

    structures, train_Y, ds = [], [], []
    with connect(db_file) as db:
        for row in db.select():
            s = db.get_atoms(id=row.id)
            if hasattr(row, 'ff_energy'):
                eng = row.ff_energy
            else:
                eng = row.data.energy/len(s)
            train_Y.append(eng)
            structures.append(s)
            pmg_struc = AseAtomsAdaptor().get_structure(s)
            ds.append(RDF(pmg_struc, R_max=10).RDF[1])
            if N is not None and len(train_Y) == N:
                break

    if ncpu == 1:
        xs = []
        for i, struc in enumerate(structures):
            strs = '\rDescriptor calculation: {:4d} out of {:4d}'.format(i+1, len(structures))
            print(strs, flush=False, end='')
            
            d = des.calculate(struc) 
            xs.append(d['x'])
    else:
        print('---Parallel mode is on, {} cores with be used'.format(ncpu))
        import tqdm
        tqdm_func = tqdm.tqdm
        structures = tqdm_func(list(structures), desc='running')

        with Pool(ncpu) as p:
            func = partial(fea, des)
            xs = p.map(func, structures)
            p.close()
            p.join()
    train_x = (xs, ds) 

    return train_x, np.array(train_Y)

def fea(des, struc):
    return des.calculate(struc)['x']

def normalize(x_train, x_test):
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)

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

def plot(Xs, Ys, labels, figname='results.png'):
    x_mins, x_maxs = [], []
    for x, y, label in zip(Xs, Ys, labels):
        plt.scatter(x, y, alpha=0.8, label=label, s=5)
        x_mins.append(np.min(x))
        x_maxs.append(np.max(x))
    xs = np.linspace(min(x_mins)-0.1, max(x_maxs)+0.1, 100)
    plt.plot(xs, xs, 'b')
    plt.plot(xs, xs+0.10, 'g--')
    plt.plot(xs, xs-0.10, 'g--')
    plt.xlabel('True (eV/atom)') 
    plt.ylabel('Prediction (eV/atom)')
    plt.legend()
    plt.xlim(min(x_mins)-0.1, max(x_maxs)+0.1)
    plt.ylim(min(x_mins)-0.1, max(x_maxs)+0.1)
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
 
def plot_two_body(model, des, figname):
    from ase import Atoms
    rs = np.linspace(1.0, 8.0, 50)
    cell = 10*np.eye(3)
    dimers = [Atoms("2Si", positions=[[0,0,0], [r,0,0]], cell=cell) for r in rs]
    
    xs = []
    for dimer in dimers:
        d = des.calculate(dimer) 
        xs.append(d['x'])
    ds = ([None]*len(dimers), xs)
    energies = model.predict(ds)
    plt.plot(rs, 2*energies)
    plt.xlabel('R (Angstrom)')
    plt.ylabel('Energy (eV)')
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()
    print("save the figure to ", figname)


