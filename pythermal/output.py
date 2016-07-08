# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Thermal equilibrium of hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics, St. Stephen's College, Delhi
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import print_function, division, absolute_import

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

__all__ = ['status', 'warning', 'write_file', 'write_image', 'read_file',
           'plot_write']


def status(time_taken=0.0):
    """
    Prints current status (execution time) of program execution.
    Note: Times returned from here are not true measures of algorithm speed.
    For rigorous function time testing use the timeit module.

    :param time_taken: Block execution time
    """
    # Differentiate between Windows and *nix systems for time
    if os.name is 'nt':
        t = time.strftime("%H%M%S", time.gmtime(time_taken))
    else:
        t = time.strftime("%T", time.gmtime(time_taken))

    print("Time: {}".format(t))


def warning(*objects):
    """
    Handles non-fatal warnings. Output to stderr.

    :param objects: Objects
    """
    print("WARNING:", *objects, file=sys.stderr)


def write_file(path, filename, data=None, fmt='%.18e'):
    """
    Checks if output directory exists, if not, creates it.
    Writes arrays/lists to the disk. Performs IO using NumPy.
    Possible switch in future to the more robust Pandas (dependant code must be unaffected).

    :param path: Folder path to write to
    :param filename: Name of file (include extension)
    :param data: Data to be written
    :param fmt: Format specifier
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        # Catches race condition
        except OSError:
            pass

    print("Writing to {}".format(path + filename))
    np.savetxt(path + filename, data, delimiter=',', fmt=fmt)


def write_image(path, filename):
    """
    Checks if output directory exists, if not, creates it. Then writes to disk.
    Writes images to the disk. Performs IO using MatPlotLib.

    :param path: Folder path to write to
    :param filename: Name of file (including extension)
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

    print("Drawing to {}.png".format(path + filename))
    plt.savefig(path + filename, format='png', dpi=400)

    # Clears figure from plt (prevents multiple plots from interfering)
    plt.clf()


def read_file(path, filename, dtype=np.float64):
    """
    Reads data from files stored locally.Performs IO using NumPy.
    Possible switch in future to the more robust Pandas (dependant code must be unaffected).

    :param path: Path to folder
    :param filename: Name of file
    :param dtype: Data type of file
    :return: Array of read data
    :raise: IOError if file not found
    """
    if os.path.isfile(path + filename):
        print('Reading from {}'.format(path + filename))
        out = np.genfromtxt(path + filename, delimiter=',', dtype=dtype)
        if dtype is complex:
            return np.nan_to_num(out)
        else:
            return out
    else:
        print('{} not found'.format(path + filename))
        raise IOError


def plot_write(x, y, title=None, y_label=None, x_label=None, y_limit=None,
               path=None, filename=None, checkbox=None):
    """
    Generate graphs using MatPlotLib. Uses write_image() to save to hard disk.
    Checkbox can be used to control plot display during execution.

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

    # Choose same folder if path not specified
    if path is None:
        path = ''

    write_image(path, filename)

    # Checkbox to display images after execution
    if checkbox:
        plt.show()
