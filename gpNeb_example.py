import numpy as np
from GPRANEB import GP_NEB
from ase.io import read, write
import os
#from NEB_GP import GP_NEB

initial_state = 'N_initial.traj'
final_state = 'N_final.traj'
num_images = 5
k_spring = 0.1
iterMax = 80
Emax_std = 0.05
fmax_std = 0.05
neb_gp = GP_NEB(initial_state, final_state, num_images=num_images, k_spring=k_spring, iterMax=iterMax)
#neb_gp.usage_instructions()
images = neb_gp.generate_images()
velocity_vec = np.zeros((num_images-2)*neb_gp.num_atoms*3)
#model = neb_gp.train_gpr_model(images)
refined_images = neb_gp.run_neb(IDPP = True, SD=False, Emax_std=Emax_std, fmax_std=fmax_std, velocity_vec=velocity_vec, n_reset=0, alpha=0.1)
# convert the images to cif files for visualization
# using ase built-in function
# Convert each refined image to a CIF file and save it in a directory
# make a directory to store the cif files
"""
os.mkdir('N_Diffgpr_images')
for i, image in enumerate(refined_images):
    filename = f'gpr_image_{i}.cif'
    write(filename, image)
    # add the file file to directory
    os.rename(filename, f'N_Diffgpr_images/{filename}')
"""
neb_gp.plot_neb_path(refined_images)

# If neb.traj file is available for comparison
# Read the trajectory file
"""
images_cf = read('neb.traj', index=':')

# Convert each image in the trajectory to a CIF file
for i, image in enumerate(images_cf):
    filename = f'cf_neb_image_{i}.cif'
    write(filename, image)
    os.rename(filename, f'gpr_images/{filename}')
"""
"""
Tsets if succefully switching the calculators
for i in range(1,10):
    neb_gp.gpr_calculator(images, Emax_std=Emax_std, fmax_std=fmax_std)
    print(i)
"""
