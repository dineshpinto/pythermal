from __future__ import print_function

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import Main


def status(status_num, time_taken=0.0):
    # Differentiate between Windows and *Nix systems
    if os.name == 'nt':
        t = time.strftime("%H%M%S", time.gmtime(time_taken))
    else:
        t = time.strftime("%T", time.gmtime(time_taken))
        
    if status_num == 1:
        print("Generated Eigenstates\nTime: {}".format(t))
    elif status_num == 2:
        print("\nGenerated Hamiltonian\nTime: {}".format(t))
    elif status_num == 3:
        print("\nGenerated Eigenvalues & Eigenvectors\nTime: {}".format(t))
    elif status_num == 4:
        print("\nGenerated Relabelled States\nTime: {}".format(t))
    elif status_num == 5:
        print("\nGenerated Psi(t)\nTime: {}".format(t))
    elif status_num == 6:
        print("\nGenerated Von-Neumannn Entropy\nTime: {}".format(t))
    elif status_num == 7:
        print("Complete!")
    else:
        warning('Invalid status')


# Handles non-fatal warnings
def warning(*objects):
    print("WARNING:", *objects, file=sys.stderr)


# Writes output to hard disk
def write_file(filename, data=None, fmt='%.18e'):
    s = Main.System()
    # Check if Output directory exists (Note: Race condition)
    path = s.folder_path()
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

    print("Writing to {}".format(filename))

    if '.png' in filename:
        plt.savefig(path + filename, format='png', dpi=400)
    else:
        np.savetxt(path + filename, data, delimiter=',', fmt=fmt)


# Reads from hard disk
def read_file(filename, dtype=np.float64):
    np.genfromtxt(filename, delimiter=',', dtype=dtype)


def plot_entropy(x, y):
    # Plot area formatting
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='0.50', linestyle='-')
    plt.minorticks_on()
    plt.ylabel(r'Von-Neumann Entropy $[S_{VN} = - tr(\rho \ln(\rho))] \rightarrow$')
    plt.xlabel(r'Time $[\tau]\rightarrow$')
    plt.title(r'Von-Neumann entropy ($S_{VN}$) vs time ($\tau$) for a 2D sub-lattice')

    # Entropy Plot
    plt.plot(x, y, 'bo', markersize=5, label=r'$S_{VN} = - tr(\rho \ln(\rho))$')
    write_file('Entropy_B.png')
    plt.show()


def plot_trace2(x, y):
    # Plot area formatting
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='0.50', linestyle='-')
    plt.minorticks_on()
    plt.ylabel(r'Trace of sqare of DM $[tr(\rho^2))] \rightarrow$')
    plt.xlabel(r'Time $[\tau]\rightarrow$')
    plt.title(r'Trace of sqare of DM vs time ($\tau$) for a 2D sub-lattice')

    # Trace squared plot 
    plt.plot(x, y, 'bo', markersize=5, label=r'$S_{VN} = - tr(\rho \ln(\rho))$')
    write_file('Trace2_B.png')
    plt.show()
