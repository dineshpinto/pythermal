# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics, New Delhi
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import print_function, division, absolute_import

import os
from datetime import datetime
from multiprocessing import cpu_count
from time import time

if cpu_count() > 4:
    if 'OPENBLAS_NUM_THREADS' not in os.environ:
        os.environ['OPENBLAS_NUM_THREADS'] = '16'

import numpy as np
from scipy.linalg import norm

from output import status, write_file, read_file
from routines import (position_states, hamiltonian_parallel, eigen, relabel,
                      rho_b_pbasis, transformation, naive_thermal,
                      h_block_diagonal)

__author__ = "Thermalization and Quantum Entanglement Project Group, SSCTP"
__version__ = "v1.8.0-states"


class System:
    def __init__(self, initial_value, lattice_a=None, lattice_b=None):
        """
        Stores metadata about the system.
        :param initial_value: List of initial state values of the system
        :param lattice_a: Lattice sites in A (only used when manually
        defining lattice)
        :param lattice_b: Lattice sites in B (only used when manually
        defining lattice)
        """
        # No of particles
        self.nop = int(initial_value[0])

        # Dimensionality of lattice
        self.n_dim = int(initial_value[1])

        # Eigenvector chosen
        self.state_num = int(initial_value[2])

        # Whole lattice
        self.lattice = np.concatenate((lattice_a, lattice_b))

        # No. of sites in sub-lattice A
        self.nol_a = len(lattice_a)

        # No. of sites in sub-lattice B
        self.nol_b = len(lattice_b)

        # Total no. of  lattice sites
        self.nol = len(self.lattice)

    @property
    def fold_path_ti(self):
        """
        :return: Path for storing time independent variables
        """
        return ('../pythermal_output/TI-{}_{}_{}/'
                .format(self.nop, self.n_dim, self.nol_a))

    @staticmethod
    def file_names():
        """
        :return: Names of variables for storing on hard disk
        """
        names = ['Hamiltonian_AB.csv', 'Hamiltonian_A.csv',
                 'Eigenvalues_AB.csv', 'Eigenvectors_AB.csv',
                 'Eigenvalues_A.csv', 'Eigenvectors_A.csv']
        return names

    @property
    def check_existence(self):
        """
        Checks whether variables exists on hard disk.
        Could be deprecated for try-except statements.
        :return: Boolean list whether files exists on hard disk
        """
        names = self.file_names()
        existence = [False] * len(names)

        for idx, name in enumerate(names):
            if os.path.isfile(self.fold_path_ti + name):
                existence[idx] = True
                print('{} exists at {}'.format(name, self.fold_path_ti))
            else:
                print('{} does not exist'.format(name))
        print('')

        return existence

    def check_system(self, lattice_a, lattice_b):
        """
        Runs checks to make sure all inputs are valid.
        Raises ValueError if not.
        :param lattice_a:
        :param lattice_b:

        """
        if lattice_a is None or lattice_b is None:
            raise ValueError('Enter both lattice A and lattice B')

        if not self.nop > 0:
            raise ValueError('No. of particles should be greater than 0')

        if self.nop > self.nol_a:
            raise ValueError('Too many particles [{}] for sub-lattice A [{}]'
                             .format(self.nop, self.nol_a))
    
        if not self.nop < self.n_dim ** 2:
            raise ValueError('Too many particles [{}] for lattice [{}]'
                             .format(self.nop, self.n_dim ** 2))
    
        if not self.n_dim > 3:
            raise ValueError('Shape of lattice must be at least 3')


def main_states(initial_values, lattice_a=None, lattice_b=None):
    """
    Contains calls to/control of all functions in program.
    :param initial_values: List of initial values for system
        initial_values = [
            Total no. of particles(nop),
            Dimension of lattice(ndims),
            eigenvector no.]
    :param lattice_a: List of sites in A
    :param lattice_b: List of sites in B
    :return: True if execution successful
    """
    s = System(initial_values, lattice_a, lattice_b)
    path_ti, names = s.fold_path_ti, s.file_names()
    s.check_system(lattice_a, lattice_b)

    tot_time1 = time()

    # Eigenstates
    pos_states_ab, nos_ab = position_states(s.lattice, s.nop)
    pos_states_a, nos_a = position_states(lattice_a, s.nop)
    write_file(path_ti, 'PositionStates_AB.csv', pos_states_ab, fmt='%1d')
    write_file(path_ti, 'PositionStates_A.csv', pos_states_a, fmt='%1d')

    # Eigenvalues and Eigenvectors
    e_time1 = time()
    try:
        eigenvalues = read_file(path_ti, names[2])
        eigenvectors = read_file(path_ti, names[3], dtype=complex)
    except IOError:
        try:
            hamiltonian = read_file(path_ti, names[0], dtype=int)
        except IOError:
            print("Hamiltonian...")
            hamiltonian = hamiltonian_parallel(s.lattice, s.n_dim, s.nop)
            write_file(path_ti, names[0], hamiltonian, fmt='%1d')
            print("Hamiltonian done!")
        print("Diagonalizing...")
        eigenvalues, eigenvectors = eigen(hamiltonian)
        write_file(path_ti, names[2], eigenvalues)
        write_file(path_ti, names[3], eigenvectors)
        print("Diagonalizing done!")
    e_time2 = time()
    status(e_time2 - e_time1)

    # Block diagonal
    bd_time1 = time()
    try:
        h_bd_evecs = read_file(path_ti, 'Hamiltonian_B_BD_Evecs.csv',
                               dtype=complex)
    except IOError:
        try:
            h_bd = read_file(path_ti, 'Hamiltonian_B_BD.csv')
        except IOError:
            print("Block diagonal...")
            # ham = []
            # for l in range(s.nop + 1):
            #     ham.append(hamiltonian_parallel(lattice_b, s.n_dim, l))
            # h_bd = la.block_diag(*ham)
            h_bd = h_block_diagonal(lattice_b, s.n_dim, s.nop)
            write_file(path_ti, 'Hamiltonian_B_BD.csv', h_bd, fmt='%1d')
            print("Block diagonal done!")
        print("Diagonalizing block diagonal...")
        _, h_bd_evecs = eigen(h_bd)
        write_file(path_ti, 'Hamiltonian_B_BD_Evecs.csv', h_bd_evecs)
    bd_time2 = time()
    status(bd_time2 - bd_time1)

    # State Relabelling
    print("Relabelling...")
    labels = relabel(pos_states_ab, s.nop, s.nol_b, lat_a=lattice_a)
    write_file(path_ti, 'RelabelledStates.csv', labels)

    print("Eigenstate = {}/{}".format(s.state_num, nos_ab))
    state = eigenvectors[s.state_num] / norm(eigenvectors[s.state_num])

    # Density Matrix
    dm_time1 = time()
    rho_fname = ('[{}]{}.csv'.format(s.state_num, eigenvalues[s.state_num]))
    try:
        rho_ebasis = read_file(path_ti + 'RhoStates(EnergyBasis)/', rho_fname,
                               dtype=complex)
    except IOError:
        try:
            rho_pbasis = read_file(path_ti + 'RhoStates(PositionBasis)/',
                                   rho_fname, dtype=complex)
        except IOError:
            print("DM in position basis...")
            rho_pbasis = rho_b_pbasis(labels, state, nos_ab, s.nol_b,
                                      s.nop)
            rho_fname = ('[{}]{}.csv'.format(s.state_num,
                                             eigenvalues[s.state_num]))
            write_file(path_ti + 'RhoStates(PositionBasis)/', rho_fname,
                       rho_pbasis)
        print("Transforming DM to Energy basis...")
        rho_ebasis = transformation(rho_pbasis, h_bd_evecs)
        write_file(path_ti + 'RhoStates(EnergyBasis)/', rho_fname, rho_ebasis)
    dm_time2 = time()
    status(dm_time2 - dm_time1)

    # Naive check for thermal DM
    max_diag, max_offdiag = naive_thermal(rho_ebasis)

    tot_time2 = time()
    status(tot_time2 - tot_time1)

    with open(path_ti + 'log.txt', mode='a') as f:
        string = ("{}\n{}"
                  "\nNo. of particles = {}"
                  "\nNo. of lattice sites in A = {}"
                  "\nNo. of lattice sites in B = {}"
                  "\nTotal no. of lattice sites = {}"
                  "\nEigenstate = {}/{}"
                  "\nMax. diagonal = {}"
                  "\nMax. off-diagonal = {}"
                  "\nRatio (Max diag/Max off-diag)= {}"

                  "\n"
                  .format('*' * 79, datetime.now(), s.nop, s.nol_a, s.nol_b,
                          s.nol, s.state_num, nos_ab, max_diag, max_offdiag,
                          max_diag / max_offdiag))
        f.write(string)
    print("Output log at {}".format(path_ti + 'log.txt'))
    return True


if __name__ == '__main__':
    """
    init_values = [
    Total no. of particles(nop),
    Dimension of lattice(ndims),
    Eigenvector no.,

    """
    # init_values = [5, 6, 26565]
    init_values = [2, 6, 2]

    # Lattices A and B
    main_states(init_values, lattice_a=np.genfromtxt('a.txt', dtype=int),
                lattice_b=np.genfromtxt('b.txt', dtype=int))
