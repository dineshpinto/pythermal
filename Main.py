from __future__ import print_function, division

import math as mt
import time

import numpy as np
import scipy.linalg as la

import Output
import SubRoutine1
import SubRoutine2
import SubRoutine3

__author__ = "Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics"


class System:
    def __init__(self, val):
        # No of particles
        self.nop = int(val[0])
        # Shape of square 2D array i.e. nsa = 2(2x2), 3(3x3)
        self.nsa = int(val[1])
        # No. of sites in sub-lattice A
        self.nol_a = int(val[2])
        # Time Evolution - starting time, ending time and no. of time steps
        self.t_initial = float(val[3])
        self.t_final = float(val[4])
        self.t_steps = int(val[5])
        # Break symmetry
        self.break_symmetry = np.array([])

        # Check for number of particles in A [Failure is FATAL]
        if self.nop > self.nol_a:
            print('Too many particles [{}] for sub-lattice A [{}]'.format(self.nop, self.nol_a))
            raise Exception

        # Lattice sites to delete for particular values of nsa & nol_a
        self.lat_del_pos, self.lat_del_pos_a = self.lattice_generator()
        # No of  lattice sites before deletion eg. nsa = 3 => nol = 9
        self.nol = self.nsa * self.nsa
        # No. of sites in sub-lattice B
        self.nol_b = self.nol - (self.nol_a + len(self.lat_del_pos)) + 1
        # Site joining sub-lattice A and B (numbered after deleting sites)
        self.link_pos = mt.sqrt(self.nol_a) * self.nsa - (self.nsa - mt.sqrt(self.nol_a))
        # Lattice before deleting sites
        self.lat = np.arange(1, self.nol + 1, dtype=np.int32)
        # Time gap between successive time steps
        self.delta_t = (self.t_final - self.t_initial) / self.t_steps
        # Array storing various times
        self.timestep_array = np.arange(self.t_initial, self.t_final, self.delta_t)
        # Show images during execution (0=NO & 1=YES)
        self.checkbox = val[6]
        # Set initial state as an eigenvector of entire system (0=NO & 1=YES)
        self.checkbox2 = val[7]

    # Generates arrays of sites to be deleted using nsa and nol_a. If unsuccessful raises FATAL Exception
    def lattice_generator(self):
        if self.nsa == 4 and self.nol_a == 4:
            self.lat_del_pos = np.array([3, 4, 9, 13])
            self.lat_del_pos_a = np.array([3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        elif self.nsa == 4 and self.nol_a == 9:
            self.lat_del_pos = np.array([4, 8, 13, 14])
            self.lat_del_pos_a = np.array([4, 8, 12, 13, 14, 15, 16])
        elif self.nsa == 5 and self.nol_a == 4:
            self.lat_del_pos = np.array([3, 4, 5, 11, 16, 21])
            self.lat_del_pos_a = np.array([3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                           25])
        elif self.nsa == 5 and self.nol_a == 9:
            self.lat_del_pos = np.array([4, 5, 9, 10, 16, 17, 21, 22])
            self.lat_del_pos_a = np.array([4, 5, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        elif self.nsa == 5 and self.nol_a == 16:
            self.lat_del_pos = np.array([5, 10, 15, 21, 22, 23])
            self.lat_del_pos_a = np.array([5, 10, 15, 21, 21, 22, 23, 24, 25])
        else:
            print('Lattice shape not supported. Unable to delete sites.')
            raise Exception

        self.lat_del_pos = np.concatenate((self.lat_del_pos, self.break_symmetry))
        self.lat_del_pos_a = np.concatenate((self.lat_del_pos_a, self.break_symmetry))
        return self.lat_del_pos, self.lat_del_pos_a

    # Returns a unique folder path to output to. Path is: Output/Case-(nop)_(nsa)_(nol_a)_(t_final)/
    def folder_path(self):
        return 'Output_PyThermal/Case-{}_{}_{}_{}/'.format(self.nop, self.nsa, self.nol_a, self.t_final)

    # Stores metadata required for matplotlib plots
    @staticmethod
    def plotting_metadata():
        titles = [r'Von-Neumann entropy ($S_{VN}$) vs time ($\tau$)', r'Purity ($tr(\rho^2))$) vs time ($\tau$)',
                  r'Avg. particles in A vs time ($\tau$)', r'Avg. particles in B vs time ($\tau$)',
                  r'Avg. particles in A and B vs time ($\tau$)']
        y_labels = [r'Von-Neumann Entropy $[S_{VN} = - tr(\rho \ln(\rho))] \rightarrow$',
                    r'Purity $[tr(\rho^2))] \rightarrow$', r'Avg. particles in A', r'Avg. particles in B',
                    r'Avg. particles in A and B']
        x_labels = [r'Time $[\tau]\rightarrow$']
        y_limits = [(0.0, 3.0), (-1.0, 2.0), (0.0, 5.0)]
        return titles, y_labels, x_labels, y_limits


def main(initial_values):
    s = System(initial_values)
    path = s.folder_path()
    titles, y_labels, x_labels, y_limits = s.plotting_metadata()

    Output.warning('Symmetry may still be present in lattice')

    # -----Sub-Routine 1 (Eigenstates, Hamiltonian, Eigenvalues and Eigenvectors)-----

    # Eigenstates
    eigenstates, nos = SubRoutine1.eigenstates_lattice(s.lat, s.nop, s.lat_del_pos)
    eigenstates_a, nos_a = SubRoutine1.eigenstates_lattice(s.lat, s.nop, s.lat_del_pos_a)
    Output.status(1)
    Output.write_file(path, 'Eigenstates.csv', eigenstates, fmt='%1d')
    Output.write_file(path, 'Eigenstates_A.csv', eigenstates_a, fmt='%1d')

    # Hamiltonian
    h_time1 = time.time()
    hamiltonian = SubRoutine1.parallel_call_hamiltonian(eigenstates, nos, s.nsa, s.nop)
    hamiltonian_a = SubRoutine1.parallel_call_hamiltonian(eigenstates_a, nos_a, s.nsa, s.nop)
    h_time2 = time.time()
    Output.status(2, h_time2 - h_time1)
    Output.write_file(path, 'Hamiltonian.csv', hamiltonian, fmt='%1d')
    Output.write_file(path, 'Hamiltonian_A.csv', hamiltonian_a, fmt='%1d')

    # Eigenvalues and Eigenvectors
    e_time1 = time.time()
    eigenvalues, eigenvectors = SubRoutine1.eigenvalvec(hamiltonian)
    eigenvalues_a, eigenvectors_a = SubRoutine1.eigenvalvec(hamiltonian_a)
    e_time2 = time.time()
    Output.status(3, e_time2 - e_time1)
    Output.write_file(path, 'Eigenvalues.csv', eigenvalues)
    Output.write_file(path, 'Eigenvectors.csv', eigenvectors)
    Output.write_file(path, 'Eigenvalues_A.csv', eigenvalues_a)
    Output.write_file(path, 'Eigenvectors_A.csv', eigenvectors_a)

    # ------Sub-Routine 2 (State Relabelling and Initial State)-----

    # State Relabelling
    r_time1 = time.time()
    re_states = SubRoutine2.relabel(eigenstates, s.nop, s.link_pos, s.nol_b)
    r_time2 = time.time()
    Output.status(4, r_time2 - r_time1)
    Output.write_file(path, 'RelabelledStates.csv', re_states)

    # Initial State
    evo_time1 = time.time()
    if s.checkbox2:
        psi_initial = eigenvectors[:, 0] / la.norm(eigenvectors[:, 0])
    else:
        psi_initial = SubRoutine2.init_state(eigenvectors_a, re_states, nos, s.nop)

    # -----Sub-Routine 3 (Time Evolution and Von-Neumann Entropy)-----

    # Time Evolution
    psi_t, sum_a, sum_b = SubRoutine3.time_evolution(psi_initial, hamiltonian, nos, s.timestep_array, re_states, s.nop)
    evo_time2 = time.time()
    Output.status(5, evo_time2 - evo_time1)
    Output.write_file(path, 'Psi.csv', psi_t)

    # Von-Neumann Entropy
    vn_time1 = time.time()
    vn_entropy_b, vn_trace2_b = SubRoutine3.von_neumann_b(psi_t, re_states, nos, s.nol_b, s.nop)
    vn_time2 = time.time()
    Output.status(6, vn_time2 - vn_time1)
    Output.write_file(path, 'VN_Entropy_B.csv', vn_entropy_b)

    # -----Output-----
    Output.plot(s.timestep_array, vn_entropy_b, titles[0], y_labels[0], x_labels[0], y_limits[0])
    Output.write_image(path, 'VN_Entropy_B.png', s.checkbox)
    Output.plot(s.timestep_array, vn_trace2_b, titles[1], y_labels[1], x_labels[0], y_limits[0])
    Output.write_image(path, 'Trace2_B.png', s.checkbox)
    Output.plot(s.timestep_array, sum_a, titles[2], y_labels[2], x_labels[0], y_limits[2])
    Output.write_image(path, 'Avg_A.png', s.checkbox)
    Output.plot(s.timestep_array, sum_b, titles[3], y_labels[3], x_labels[0], y_limits[2])
    Output.write_image(path, 'Avg_B.png', s.checkbox)
    Output.plot(s.timestep_array, sum_b + sum_a, titles[4], y_labels[4], x_labels[0], y_limits[2])
    Output.write_image(path, 'Avg_AB.png', s.checkbox)

    # -----Terminate-----
    Output.status(7)
    return True


if __name__ == '__main__':
    # [nop, nsa, nol_a, t_initial, t_final, t_steps, Show images(1=YES), initial psi(1=eigenstate of entire system)]
    init_values = [2, 4, 9, 0.0, 50.0, 100, 0, 0]
    main(init_values)
