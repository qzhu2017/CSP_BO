import sys # this may not be necessary
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.io import read
from ase.mep import NEB
from ase.db import connect

from cspbo.utilities import metric_single, build_desc, get_data, get_train_data, rmse
from cspbo.gaussianprocess import GaussianProcess as gpr
from cspbo.calculator import GPR
from cspbo.kernels.RBF_mb import RBF_mb
from cspbo.kernels.Dot_mb import Dot_mb # recommended at the moment
from scipy.interpolate import CubicSpline, make_interp_spline
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF

"""
Perform a Gaussian Process Regression (GPR) aided NEB calculation.
The module is designed to be used in the ASE package.
The module will use SO3 as the descriptor for the GPR model.
Can opt for RBF or Dot product kernel for the GPR model.
Dot product kernel is the default kernel used if not specified.

"""
print(".............GPRANEB module loaded.............\n")
print("You're welcome to modify the code to suit your needs\n")
print("..........Good luck with your calculations..........\n")

class GP_NEB:
    def __init__(self, initial_state, final_state, 
                 num_images = 5, 
                 k_spring = 0.1, 
                 iterMax = 100, 
                 k_Dot = True, 
                 f_cov = 0.05, 
                 useCalc=EMT()):

        self.initial_state = read(initial_state)
        self.final_state = read(final_state)
        self.num_atoms = len(self.initial_state)
        self.num_images = num_images
        self.k_spring = k_spring
        self.iterMax = iterMax
        self.k_Dot = k_Dot
        self.f_cov = f_cov
        self.model = None
        self.gpr_json = 'gpr_model.json'
        self.gpr_db = 'gpr_model.db'
        self.set_GPR()
        self.calc = GPR(ff=self.model, return_std=True)
        self.useCalc = useCalc
        #self.add_to_db = add_to_db
        # The initial and final states are ase trajectory files
        # For example, initial.traj and final.traj

    def set_GPR(self, lm=3, nm=3, rcut=4.0, zeta=2.0, device='cpu'):
        """
        Setup GPR model
        """
        if os.path.exists(self.gpr_json):
            self.model = gpr()
            self.model.load(self.gpr_json)
        else:
            des = build_desc("SO3", lmax=lm, nmax=nm, rcut=rcut)
            if self.k_Dot:
                kernel = Dot_mb(para=[2, 0.5], zeta=zeta, device=device)
            else:
                kernel = RBF_mb(para=[1, 0.5], zeta=zeta, device=device)

            self.model = gpr(kernel=kernel, 
                             descriptor=des, 
                             noise_e=[0.01, 0.01, 0.03], 
                             f_coef=20)

    # Start with a function to generate the images
    # This function will take advantage of the ASE NEB module
    # The number of images generated is n-2 where n is the number of images specified
    def generate_images(self, IDPP = False):
        """
        Generate initial images from ASE's NEB module
        """
        initial = self.initial_state
        final = self.final_state
        num_images = self.num_images

        # Add the initial image
        images = [initial]
        imgCount = 0
        for i in range(1,num_images-1):
            images.append(initial.copy())
            imgCount += 1
        # Add the final image
        images.append(final)
        # Use the NEB module to interpolate the images
        neb = NEB(images)
        if IDPP:
            neb.interpolate(method='idpp')
        else:
            neb.interpolate()
        # Linear interpolation is the default but IDPP can be used
        # We definitely implement our own interpolation method
        return images
    
    # Function to train the GPR model
    def train_gpr_model(self, pts):
        self.model.set_train_pts(pts, mode='a+')
        self.model.fit(opt=True)
        train_E, train_E1, train_F, train_F1 = self.model.validate_data()
        l1 = metric_single(train_E, train_E1, "Train Energy")
        l2 = metric_single(train_F, train_F1, "Train Forces")
        print(self.model)

   
    # Now we write the NEB algorithm to get the neb forces
    # These forces will be used to optimize the images
    """
    Here on out we could use the ASE NEB module to optimize the images
    For example, we could use the BFGS optimizer to optimize the images
    Issue is this would do a check to see if to switch between the GPR model and the ab initio calculator
    This would slow down the optimization process. 
    We want to check if to switch between the GPR model and the ab initio calculator after every 10 steps

    There is also the issue of assing each image a calculator in a way that ASE recognizes

    """
    def calculate_neb_forces(self, images, ase_nebLib = False):
        # Ideally, we would use the ASE NEB module to get the neb forces
        # The images should have been assigned the a calculator
        if ase_nebLib:
            neb = NEB(images)
            neb_forces = neb.get_forces()
            # The forces are usually a 2D array (num_images-2)*num_atoms by 3
            # We want it to be  of the shape shape as those from neb_force_calc
            # THis to make sure we can use them later in the path_update function
            neb_forces = neb_forces.reshape((self.num_images-2, self.num_atoms, 3))
            return neb_forces
        else:
            # We write our own NEB algorithm here
            # We shall write a function for this.
            neb_forces = self.neb_force_calc(images)
            return neb_forces

    
    def neb_force_calc(self, images):
        """ 
        We shall use the original NEB implementation
        Find the Improved Tangent Method in the articles below
        Henkelman et al. J. Chem. Phys. 113, 9901 (2000)
        Herbol et al. J. Chem. Theory Comput. 2017, 13, 3250-3259

        We shall implement the ITM method later
        """
        # Use the real forces. This makes the first and last image forces zero
        forces_atoms = []
        fin_neb_forces = []
        E_images = []
        for image in images:
            ###image.calc = self.gpr_calculator(image)
            forces_atoms.append(image.get_forces())
            E_images.append(image.get_potential_energy())
        
        forces_atoms = np.array(forces_atoms)
        real_forces = np.zeros(forces_atoms.shape)
        real_forces[1:-1] = forces_atoms[1:-1]
        E_images = np.array(E_images)

        # NEB algorithm
        """
        J. Chem. Phys. 113, 9978-9985 (2000) DOI: 10.1063/1.1323224
        """
        for i in range(1, self.num_images-1):
            # Calculate the spring forces
            forces = real_forces[i]
            forces = forces.flatten()
            dR_right = images[i+1].get_positions().flatten() - images[i].get_positions().flatten()
            dR_left = images[i].get_positions().flatten() - images[i-1].get_positions().flatten()
            dR_right_norm = np.linalg.norm(dR_right)
            dR_left_norm = np.linalg.norm(dR_left)
            tangent = dR_right/dR_right_norm + dR_left/dR_left_norm
            tangent = tangent/np.linalg.norm(tangent) # Normalize the tangent
            # Now we are ready to calculate the spring forces. # Should the spring constant be negative?
            spring_force = self.k_spring*(dR_right_norm - dR_left_norm)*tangent
            # Now calculate the true forces
            true_force = forces - np.dot(forces, tangent)*tangent
            # Now calculate the total forces
            total_perp_force = spring_force - true_force
            total_perp_force = total_perp_force.reshape((self.num_atoms, 3))
            fin_neb_forces.append(total_perp_force)
        
        """
        The ITM method will be implemented here with if statement to check if called upon
        
        """
        return fin_neb_forces

    # Path update function. Either steepest descent or Quick-min
    def path_update(self, images, neb_forces, velocity_vec, n_reset = 0, alpha = 0.1, SD=True, step_size=0.1):
        # The path update method is either steepest descent or Quick-min
        # We shall implement the Quick-min method later
        positions = [images[i].get_positions() for i in range(1, self.num_images-1)]
        forces = [neb_forces[i-1] for i in range(1, self.num_images-1)]
        positions = np.array(positions).flatten()
        forces = np.array(forces).flatten()

        if SD:
            """
            for i in range(1, self.num_images-1):
                # Calculate the new positions
                # Sometimes the forces are rescaled by max force
                new_positions = images[i].get_positions() + step_size*neb_forces[i-1]
                images[i].set_positions(new_positions)
                # maybe roll out the positions and forces and do this at once instead of looping through each image
            """
            # get all the positions and forces (neb_forces) and flatten them
            # The forces and positions should be of the same shape
            """
            positions = [images[i].get_positions() for i in range(1, self.num_images-1)]
            forces = [neb_forces[i-1] for i in range(1, self.num_images-1)]
            positions = np.array(positions).flatten()
            forces = np.array(forces).flatten()
            """
            new_positions = positions - step_size*forces
            new_positions = new_positions.reshape((self.num_images-2, self.num_atoms, 3))
            for i in range(1, self.num_images-1):
                images[i].set_positions(new_positions[i-1])
            
            return images
        else:
            
            """
             J. Chem. Phys. 128, 134106 2008; DOI: 10.1063/1.2841941
             # Implement simple Quick-min update function
            
            # velocity_vec is same length as positions and forces
            # Project the velocity in the direction of the force
            velocity_vec = np.array(velocity_vec).flatten() # This is precaution incase the velocity_vec is multidimensional
            forces_tild = forces/np.linalg.norm(forces)
            velocity_vec = np.dot(velocity_vec, forces)*forces_tild
            # If the dot product is negative, then zero vector for the velocity_vec
            if np.dot(velocity_vec, forces) < 0:
                velocity_vec = np.zeros(velocity_vec.shape)
            else:
                pass # This line may not be necessary but it's here for clarity and safety
            # Now update the positions
            new_positions = positions - step_size*velocity_vec # + or -?
            velocity_vec = velocity_vec + step_size*forces
            new_positions = new_positions.reshape((self.num_images-2, self.num_atoms, 3))
            # Update the positions of the images
            for i in range(1, self.num_images-1):
                images[i].set_positions(new_positions[i-1])
            """
            """
            # Implement the FIRE algorithm as update function
            Phys. Rev. Lett. 97, 170201 (2006); DOI: 10.1103/PhysRevLett.97.170201
            J. Chem. Phys. 128, 134106 2008; DOI: 10.1063/1.2841941

            """
            # Project the velocity in the direction of the force
            velocity_vec = np.array(velocity_vec).flatten() # This is precaution incase the velocity_vec is multidimensional
            # Parameters needed
            # dt is the time step, lets make it equal to the step_size
            dt = step_size # Redundant but here for clarity
            alpha_initial = alpha
            f_inc = 1.1
            f_dec = 0.5
            f_acc = 0.99
            #n_reset = 0 # variable to keep track of and return
            N_min = 5
            max_dt = 10*dt
            #alpha = alpha_initial # variable to keep track of and return
            # Calculate the dot products of the velocities and the force
            v_dot_f = np.dot(velocity_vec, forces)
            v_dot_v = np.dot(velocity_vec, velocity_vec)
            f_dot_f = np.dot(forces, forces)
            velocity_prime = (1.0 - alpha)*velocity_vec + alpha*np.sqrt(v_dot_v/f_dot_f)*forces
            # If the dot product is negative, then zero vector for the velocity_vec
            if v_dot_f > 0:
                if n_reset > N_min:
                    dt = min(dt*f_inc, max_dt)
                    alpha *= f_acc
                else:
                    pass # This line may not be necessary but it's here for clarity and safety
                n_reset += 1
            else:
                velocity_prime = np.zeros(velocity_vec.shape)
                alpha = alpha_initial
                dt *= f_dec
                n_reset = 0
            # Now update the positions
            new_positions = positions - dt*velocity_prime # + or -?
            velocity_vec = velocity_prime + dt*forces
            new_positions = new_positions.reshape((self.num_images-2, self.num_atoms, 3))
            # Update the positions of the images
            for i in range(1, self.num_images-1):
                images[i].set_positions(new_positions[i-1])
            
            return images, velocity_vec, n_reset, alpha

    # Now let's the driver function to run the NEB calculation
    # call it run_neb
    def run_neb(self, IDPP = False, SD=True, 
                step_size=0.1, 
                Emax_std = 0.05, 
                fmax_std = 0.05, 
                velocity_vec = None, 
                n_reset = 0, 
                alpha = 0.1,
                on_the_fly=True):

        images = self.generate_images(IDPP)
        log_file = open('neb_log.txt', 'w')
        log_file.write('..................NEB log file..................\n')
        log_file.write('....................Welcome.....................\n')
        log_file.write('............Starting the NEB calculation........\n')

        num_useCalc = 0
        for image in images: 
            image.calc = self.useCalc
            num_useCalc += 1
            if on_the_fly:
                data = (image, image.get_potential_energy(), image.get_forces())
                pts, N_pts, _ = self.model.add_structure(data)
                if N_pts > 0:
                    self.model.set_train_pts(pts, mode="a+") 
        if on_the_fly:
            self.model.fit()
            for image in images:
                image.calc = self.calc
 
        for i in range(self.iterMax):
            neb_forces = self.calculate_neb_forces(images)
            if SD:
                images = self.path_update(images, neb_forces, velocity_vec, SD, step_size)
            else:
                images, velocity_vec, n_reset, alpha = self.path_update(images, neb_forces, velocity_vec, n_reset, alpha, SD, step_size)
            log_file.write(f'Iteration {i+1} completed\n')
            log_file.write(f'Images updated\n')
            for j, image in enumerate(images):
                if on_the_fly:
                    E, F, _, E_std, F_std = self.model.predict_structure(image, stress=False, return_std=True)
                    log_file.write(f'Image {j} E: {E}/{E_std}; forces: {F} F_std_max {F_std.max()}\n')
                    #if E_std > Emax_std or F_std.max() > fmax_std:
                    if E_std > 1.2*self.model.noise_e or F_std.max() > 1.2*self.model.noise_f:
                        num_useCalc += 1
                        image.calc = self.useCalc
                        data = (image, image.get_potential_energy(), image.get_forces())
                        pts, N_pts, _ = self.model.add_structure(data)
                        if N_pts > 0:
                            self.train_gpr_model(pts)
                        image.calc = self.calc
                        log_file.write(f'Image {j} E: {E}/{E_std}; forces: {F} F_std_max {F_std.max()}\n')
                else:
                    num_useCalc += 1
                    image_calc = self.useCalc

            # Use max of the l2 norm of the force on each image
            #nf = neb_forces.reshape((self.num_images-2, self.num_atoms*3))
            #fnorm = np.linalg.norm(nf, axis=1)
            #fmax_c = fnorm.max()
            neb_forces = np.array(neb_forces)
            fmax_c = np.sqrt((neb_forces ** 2).sum(axis=1).max())
            log_file.write(f'Number of iterations: {i+1}\n')
            log_file.write(f'Max force: {fmax_c}\n')
            log_file.write('....................................................\n')
            log_file.write('Force on the atoms\n')

            print(f'Iteration {i+1} completed, Max force: {fmax_c}')

            if fmax_c < self.f_cov:
                log_file.write('\n................NEB calculation converged..................\n\n')
                print(f"\nNEB calculation converged {fmax_c}/{self.f_cov}")
                break



        # Final log entries
        log_file.write(f'Number of iterations: {i}\n')
        log_file.write('Final image forces:\n')
        for j, image in enumerate(images):
            log_file.write(f'Image {j} forces: {image.get_forces()}\n')
            log_file.write(f'Image {j} energy: {image.get_potential_energy()}\n')

        log_file.write('.......................................................\n')
        #log_file.write('Final NEB forces ............\n')
        log_file.write('..................End of NEB log file..................\n')
        log_file.close()
        print(f"\nTotal Number of useCalc calls: {num_useCalc}")
            
        #neb_forces = self.calculate_neb_forces(images)
        #images = self.path_update(images, neb_forces, velocity_vec, method, step_size)
        return images
    
    # Function to plot the NEB path
    def plot_neb_path(self, images, unit='eV', figname='neb_path.png'):
        """
        This is just to show what it looks like
        There are more elegant ways to plot the path
        """
        posToDist = np.array([image.positions for image in images])
        imgDist = np.cumsum([np.linalg.norm(posToDist[i] - posToDist[i+1]) for i in range(len(posToDist)-1)])
        imgDist = [0] + imgDist.tolist()  # Add the initial point at distance 0

        # Calculate the energies of the images
        imgEnergies = [image.get_potential_energy() for image in images]

        # Spline interpolation for a smooth curve
        imgDist_smooth = np.linspace(min(imgDist), max(imgDist), 300)
        spline = make_interp_spline(imgDist, imgEnergies, k=3)
        imgEnergies_smooth = spline(imgDist_smooth)

        # Plot the NEB path
        plt.figure()
        plt.plot(imgDist_smooth, imgEnergies_smooth, marker='', linestyle='-', color='b', label='Interpolated Path')
        plt.plot(imgDist, imgEnergies, 'o', color='r', label='Images')
        plt.xlabel('Reaction path')
        plt.ylabel(f'Energy ({unit})')
        plt.title('NEB Path')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig(figname)

        #pass
        
    
    def usage_instructions(self):
        """
        Prints out instructions and examples on how to use the GP_NEB class.
        """
        instructions = """
            GP_NEB Class Usage Instructions:
    
            1. Initialize the GP_NEB object:
                neb_gp = GP_NEB(initial_state, final_state, num_images=5, k_spring=5, iterMax=100)
    
            2. Generate images:
                images = neb_gp.generate_images()
    
            3. Train the GPR model:
                model = neb_gp.train_gpr_model(images)
    
            4. Refine the images using the GPR model:
                refined_images = neb_gp.run_neb(Emax_std=0.05, fmax_std=0.1)
    
            5. Plot the NEB path:
                neb_gp.plot_neb_path(refined_images, filename='neb_path.png')
    
            6. Convert refined images to CIF files:
                for i, image in enumerate(refined_images):
                    filename = f'refined_image_{i}.cif'
                    write(filename, image)
    
            7. Convert images in neb.traj to individual CIF files:
                images_from_traj = read('neb.traj', index=':')
                for i, image in enumerate(images_from_traj):
                    filename = f'neb_image_{i}.cif'
                    write(filename, image)
    
            Example:
            -----------
            initial_state = 'initial.traj'
            final_state = 'final.traj'
            num_images = 5
            k_spring = 5
            iterMax = 100
            Emax_std = 0.05
            fmax_std = 0.1
    
            neb_gp = GP_NEB(initial_state, final_state, num_images=num_images, k_spring=k_spring, iterMax=iterMax)
            images = neb_gp.generate_images()
            model = neb_gp.train_gpr_model(images)
            refined_images = neb_gp.run_neb(Emax_std=Emax_std, fmax_std=fmax_std)
            neb_gp.plot_neb_path(refined_images, filename='neb_path.png')
    
            for i, image in enumerate(refined_images):
                filename = f'refined_image_{i}.cif'
                write(filename, image)
    
            images_from_traj = read('neb.traj', index=':')
            for i, image in enumerate(images_from_traj):
                filename = f'neb_image_{i}.cif'
                write(filename, image)
            """
        print(instructions)
    
    #print(".............You use the usage_instructions method for help.............\n")
