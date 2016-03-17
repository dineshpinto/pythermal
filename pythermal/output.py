# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Thermal equilibrium of hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics, New Delhi
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
           'plotting_metadata', 'plotting', 'plot_write']


def status(time_taken=0.0):
    """
    Prints current status (execution time) of program execution.
    Note: Times returned from here are not true measures of algorithm speed.
    For rigorous testing use timeit.

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
    Handles non-fatal warnings.

    :param objects: Objects
    """
    print("WARNING:", *objects, file=sys.stderr)


def write_file(path, filename, data=None, fmt='%.18e'):
    """
    Checks if output directory exists, if not, creates it.
    Writes arrays/lists to the disk. Performs IO using NumPy.

    Possible switch in future to pandas (dependant code must be unaffected).

    :param path: Folder path to write to
    :param filename: Name of file (include extension)
    :param data: Data to be written
    :param fmt: Format specifier
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            # Catches race condition
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

    Possible switch in future to pandas (dependant code must be unaffected).

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


def plotting_metadata():
    """
    Stores metadata for MatPlotLib plots.

    :return: Filename of images
    :return: Image titles
    :return: y axis labels
    :return: x axis labels
    :return: y axis limits
    """
    filenames = ['VN_Entropy_B', 'Trace2_B', 'Avg_A', 'Avg_B', 'Avg_AB']

    titles = [r'Von-Neumann entropy ($S_{VN}$) vs time ($\tau$)',
              r'Purity ($tr(\rho^2))$) vs time ($\tau$)',
              r'Avg. particles in A vs time ($\tau$)',
              r'Avg. particles in B vs time ($\tau$)',
              r'Avg. particles in A and B vs time ($\tau$)']

    y_labels = [r'Von-Neumann Entropy $[S_{VN} = - tr(\rho \ln(\rho))]$',
                r'Purity $[tr(\rho^2))] \rightarrow$',
                r'Avg. particles in A', r'Avg. particles in B',
                r'Avg. particles in A and B']

    x_labels = [r'Time $[\tau]\rightarrow$']

    y_limits = [(0.0, 3.0), (-1.0, 2.0), (0.0, 5.0)]

    return filenames, titles, y_labels, x_labels, y_limits


def plotting(ent_b, tr_sqr_b, avg_part_a, avg_part_b, path_td, t, chk=None):
    """
    Call to plot_write() with x and y data. Extracts metadata for
    MatPlotLib plots from plotting_metadata().

    :param ent_b: Entropy of B
    :param tr_sqr_b: Trace of square of B
    :param avg_part_a: Avg. particles in A
    :param avg_part_b: Avg. particles in B
    :param path_td: Time dependent variables file path
    :param t: Array of times
    :param chk: Checkbox for showing images
    """
    image_name, titles, y_labels, x_labels, y_limits = plotting_metadata()

    plot_write(t, ent_b, titles[0], y_labels[0], x_labels[0],
               y_limits[0], path_td, image_name[0], chk)
    plot_write(t, tr_sqr_b, titles[1], y_labels[1], x_labels[0],
               y_limits[0], path_td, image_name[1], chk)
    plot_write(t, avg_part_a, titles[2], y_labels[2], x_labels[0],
               y_limits[2], path_td, image_name[2], chk)
    plot_write(t, avg_part_b, titles[3], y_labels[3], x_labels[0],
               y_limits[2], path_td, image_name[3], chk)
    plot_write(t, avg_part_b + avg_part_a, titles[4], y_labels[4],
               x_labels[0], y_limits[2], path_td, image_name[4], chk)


def plot_write(x, y, title=None, y_label=None, x_label=None, y_limit=None,
               path=None, filename=None, checkbox=None):
    """
    Generate graphs using MatPlotLib and save to hard disk.
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

    if checkbox:
        plt.show()

    write_image(path, filename)
