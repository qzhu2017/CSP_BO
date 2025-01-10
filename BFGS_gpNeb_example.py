import numpy as np
#from GPRANEB_v4 import GP_NEB
#from GPRANEB import GP_NEB
from GPRANEB_Hybrid import GP_NEB
from ase.io import read, write
import os
#from NEB_GP import GP_NEB

initial_state = 'initial.traj'
final_state = 'final.traj'
num_images = 5
k_spring = 0.1
iterMax = 100
Emax_std = 0.01
fmax_std = 0.01
neb_gp = GP_NEB(initial_state, final_state, num_images=num_images, k_spring=k_spring, iterMax=iterMax, kernel='Dot')
#neb_gp.usage_instructions()
images = neb_gp.generate_images(IDPP=False)

#neb_gp.use_BFGS(images)

#neb_gp.train_gpr_model(images)
#neb_gp.calc_assign(images[1])
neb_gp.useBFGS(images) # use BFGS to optimize the images
"""
for i in range(num_images):
    neb_gp.calc_assign(images[i])
""" 


#velocity_vec = np.zeros((num_images-2)*neb_gp.num_atoms*3)
#model = neb_gp.train_gpr_model(images)
#refined_images = neb_gp.run_neb(IDPP = True, SD=False, Emax_std=Emax_std, fmax_std=fmax_std, velocity_vec=velocity_vec, n_reset=0, alpha=0.1)
#refined_images = neb_gp.run_neb_calc_assign(IDPP = False, SD=True, Emax_std=Emax_std, fmax_std=fmax_std, velocity_vec=velocity_vec, n_reset=0, alpha=0.1)
# convert the images to cif files for visualization

#neb_gp.plot_neb_path(refined_images)

