from __future__ import print_function

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


def status(status_num, time_taken=0.0):
    # Differentiate between Windows and *nix systems
    if os.name is 'nt':
        t = time.strftime("%H%M%S", time.gmtime(time_taken))
    else:
        t = time.strftime("%T", time.gmtime(time_taken))

    if status_num is 1:
        print("Generated Eigenstates\nTime: {}".format(t))
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
    return


# Writes output images to hard disk, similar to write_file()
def write_image(path, filename, checkbox):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

    print("Drawing to {}".format(filename))
    plt.savefig(path + filename, format='png', dpi=400)
    # Shows image n execution if checkbox value is 1
    if checkbox:
        plt.show()
    # Clears figure from plt (prevents multiple plots from interfering)
    plt.clf()
    return


# Reads data file from hard disk
def read_file(filename, dtype=np.float64):
    np.genfromtxt(filename, delimiter=',', dtype=dtype)
    return


# Generate graphs using matplotlib, uses metadata from class System.plotting_method()
def plot(x, y, title, y_label, x_label, y_limit=None):
    # Plot area formatting
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='0.50', linestyle='-')
    plt.minorticks_on()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.ylim(y_limit)

    # Plot formatting
    plt.plot(x, y, 'bo', markersize=4)
    return
