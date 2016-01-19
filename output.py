# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import print_function

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


def status(status_num, time_taken=0.0):
    """
    Prints current status of program execution
    :param status_num: Current status of program execution
    :param time_taken: Block execution time

    """
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


def warning(*objects):
    """
    Handles non-fatal warnings
    :param objects:
    """
    print("WARNING:", *objects, file=sys.stderr)


def write_file(path, filename, data=None, fmt='%.18e'):
    """
    Checks if output directory exists, if not, creates it. Then writes to disk.
    :param path: Folder path to write to
    :param filename: Name of file
    :param data: Data o be written
    :param fmt: Format specifier

    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

    print("Writing to {}".format(filename))
    np.savetxt(path + filename, data, delimiter=',', fmt=fmt)


def write_image(path, filename):
    """
    Checks if output directory exists, if not, creates it. Then writes to disk.
    :param path: Folder path to write to
    :param filename: Name of file

    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

    print("Drawing to {}".format(filename))
    plt.savefig(path + filename, format='png', dpi=400)

    # Clears figure from plt (prevents multiple plots from interfering)
    plt.clf()


def read_file(path, filename, dtype=np.float64):
    """
    :param path: Path to folder
    :param filename: Name of file
    :param dtype: Data type of file
    :return: Array

    """
    if os.path.isfile(path + filename):
        print('Reading from {}'.format(filename))
        return np.genfromtxt(path + filename, delimiter=',', dtype=dtype)

    else:
        print('{} not found'.format(path + filename))
        raise IOError


def plot(x, y, title=None, y_label=None, x_label=None, y_limit=None, path=None, filename=None, checkbox=None):
    # Plot area formatting
    """
    Generate graphs using matplotlib, uses metadata from class System.plotting_method()
    :param x: x axis
    :param y: y axis
    :param title: Graph title
    :param y_label: y axis label
    :param x_label: x axis label
    :param y_limit: y axis limits
    :param path: Path for saving file
    :param filename: Name of file
    :param checkbox: Show images during execution(1) or not(0)

    """
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
