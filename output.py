"""
This file is a part of PyThermal. https://github.com/dkpinto/PyThermal

PyThermal - Time evolving fermions on a 2D crystal lattice
Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics

Project Mentor: Dr. A. Gupta
Project Students: A. Kumar, D. Pinto and M. Ghosh
"""

from __future__ import print_function

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


# Returns status of program execution
def status(status_num, time_taken=0.0):
    # Differentiate between Windows and *nix systems
    if os.name is 'nt':
        t = time.strftime("%H%M%S", time.gmtime(time_taken))
    else:
        t = time.strftime("%T", time.gmtime(time_taken))

    if status_num is 1:
        print("Generated Eigenstates\tTime: {}".format(t))
    elif status_num is 2:
        print("\nGenerated Hamiltonian\nTime: {}".format(t))
    elif status_num is 3:
        print("\nGenerated Eigenvalues & Eigenvectors\nTime: {}".format(t))
    elif status_num is 4:
        print("\nGenerated Relabelled States\nTime: {}".format(t))
    elif status_num is 5:
        print("\nGenerated Psi(t)\nTime: {}".format(t))
    elif status_num is 6:
        print("\nGenerated Von-Neumannn Entropy\nTime: {}".format(t))
    elif status_num is 7:
        print("\nComplete!")
    else:
        warning('Invalid status')


# Handles non-fatal warnings
def warning(*objects):
    print("WARNING:", *objects, file=sys.stderr)


# Writes output data to hard disk, path from class System.folder_path()
def write_file(path, filename, data=None, fmt='%.18e'):
    # Check if Output directory exists, if not then create it (Note race condition)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

    print("Writing to {}".format(filename))
    np.savetxt(path + filename, data, delimiter=',', fmt=fmt)


# Writes output images to hard disk, similar to write_file()
def write_image(path, filename):
    # Check if Output directory exists, if not then create it (Note race condition)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

    print("Drawing to {}".format(filename))
    plt.savefig(path + filename, format='png', dpi=400)

    # Clears figure from plt (prevents multiple plots from interfering)
    plt.clf()


# Reads data file from hard disk
def read_file(path, filename, dtype=np.float64):
    if os.path.isfile(path + filename):
        print('Reading from {}'.format(filename))
        return np.genfromtxt(path + filename, delimiter=',', dtype=dtype)

    else:
        print('{} not found'.format(path + filename))
        raise IOError


# Generate graphs using matplotlib, uses metadata from class System.plotting_method()
def plot(x, y, title=None, y_label=None, x_label=None, y_limit=None, path=None, filename=None, checkbox=None):
    # Plot area formatting
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='0.50', linestyle='-')
    plt.minorticks_on()

    # Plot labels and limits
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.ylim(y_limit)

    plt.plot(x, y, 'bo', markersize=4)

    if checkbox:
        plt.show()

    write_image(path, filename)
