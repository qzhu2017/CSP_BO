import numpy as np
from cspbo.utilities import metric_single
from ase.calculators.emt import EMT
from GPRANEB import GP_NEB
from ase.mep import NEB
from cspbo.calculator_hybrid import GPR
from ase.optimize import BFGS
from ase.optimize import MDMin

initial_state = 'database/initial.traj'
final_state = 'database/final.traj'
num_images = 5
k_spring = 0.1
iterMax = 100
Emax_std = 0.005
fmax_std = 0.05

for kernel in ['Dot', 'RBF']:
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
    train_E, train_E1, train_F, train_F1 = neb_gp.model.validate_data()
    l1 = metric_single(train_E, train_E1, "Train Energy")
    l2 = metric_single(train_F, train_F1, "Train Forces")
    print(neb_gp.model)

    for image in images:
        image.calc = GPR(base_calculator=neb_gp.useCalc,
                         ff=neb_gp.model,
                         return_std=True)
        #image.calc.freeze()

    ## Now we can use the BFGS optimizer to optimize the images
    neb = NEB(images)
    opt = BFGS(neb, trajectory='neb.traj') ###
    ##opt = MDMin(neb, trajectory='neb.traj') ###
    opt.run(fmax=0.01)


for kernel in ['Dot', 'RBF']:
    neb_gp = GP_NEB(initial_state, final_state, 
                    num_images=num_images, 
                    k_spring=k_spring, 
                    iterMax=iterMax,
                    kernel = kernel,
                    )
    
    images = neb_gp.generate_images()
    velocity_vec = np.zeros((num_images-2)*neb_gp.num_atoms*3)
    images = neb_gp.run_neb(IDPP = True, SD=False, 
                            Emax_std=Emax_std, 
                            fmax_std=fmax_std, 
                            velocity_vec=velocity_vec, 
                            n_reset=0, 
                            alpha=0.1)
    neb_gp.plot_neb_path(images, figname=kernel+'onthefly.png')


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
