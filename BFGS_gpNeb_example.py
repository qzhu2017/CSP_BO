import numpy as np
from cspbo.utilities import metric_single, build_desc, get_data, get_train_data, rmse
from ase.calculators.emt import EMT
from GPRANEB_Hybrid import GP_NEB
from ase.mep import NEB
from cspbo.calculator_hybrid import GPR
from ase.optimize import BFGS
from ase.optimize import MDMin

initial_state = 'initial.traj'
final_state = 'final.traj'
num_images = 5
k_spring = 0.1
iterMax = 100
Emax_std = 0.005
fmax_std = 0.05

for kernel in ['Dot', 'RBF']:
#for kernel in ['RBF']:
    neb_gp = GP_NEB(initial_state, 
                    final_state, 
                    num_images=num_images, 
                    k_spring=k_spring, 
                    iterMax=iterMax, 
                    kernel=kernel)

    images = neb_gp.generate_images(IDPP = False)#[:1]

    for i, image in enumerate(images):
        image.calc = neb_gp.useCalc
        data = (image, image.get_potential_energy(), image.get_forces())
        pts, N_pts, _ = neb_gp.model.add_structure(data)
        if N_pts > 0:
            neb_gp.model.set_train_pts(pts, mode="a+")

    neb_gp.model.fit()
    #train_E, train_E1, E_std, train_F, train_F1, F_std = neb_gp.model.validate_data(return_std=True); print(E_std)
    train_E, train_E1, train_F, train_F1 = neb_gp.model.validate_data()
    l1 = metric_single(train_E, train_E1, "Train Energy")
    l2 = metric_single(train_F, train_F1, "Train Forces")
    print(neb_gp.model)

    for image in images:
        image.calc = GPR(base_calculator=neb_gp.useCalc,
                         ff=neb_gp.model,
                         return_std=True)
        #image.calc.freeze()
    for image in images:
        print('test', image.positions[-1])

    ## Now we can use the BFGS optimizer to optimize the images
    neb = NEB(images)
    opt = BFGS(neb, trajectory='neb.traj') ###
    ##opt = MDMin(neb, trajectory='neb.traj') ###
    opt.run(fmax=0.01)
    ##pass
