import numpy as np
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

for kernel in ['RBF']:
    neb_gp = GP_NEB(initial_state, 
                    final_state, 
                    num_images=num_images, 
                    k_spring=k_spring, 
                    iterMax=iterMax, 
                    kernel=kernel)

    images = neb_gp.generate_images(IDPP=False)

    for image in images:
        image.calc = neb_gp.useCalc
        data = (image, image.get_potential_energy(), image.get_forces())
        pts, N_pts, _ = neb_gp.model.add_structure(data)
        if N_pts > 0:
            neb_gp.model.set_train_pts(pts, mode="a+")
    neb_gp.model.fit()

    for image in images:
        image.calc = GPR(base_calculator=neb_gp.useCalc,
                         ff=neb_gp.model,
                         return_std=True)
    # Now we can use the BFGS optimizer to optimize the images
    neb = NEB(images)
    #opt = BFGS(neb, trajectory='neb.traj') ###
    opt = MDMin(neb, trajectory='neb.traj') ###
    opt.run(fmax=0.1)
    #pass

for image in images:
    print(image.get_potential_energy())
