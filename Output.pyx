from __future__ import print_function

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


def status(status_num, time_taken=0.0):
    if status_num == 1:
        print("\n\tSub-Routine 1\nEigenstates........Complete!")
        print("\nHamiltonian........Initiated")
    elif status_num == 2:
        h_time = time.strftime("%T", time.gmtime(time_taken))
        print("Hamiltonian........Complete!\tExecution time = ", h_time)
        print("\nDiagonalization....Initiated")
    elif status_num == 3:
        e_time = time.strftime("%T", time.gmtime(time_taken))
        print("Diagonalization....Complete!\tExecution time = ", e_time)
        print("\n\tSub-Routine 2\nRecursion.....Initiated")
    elif status_num == 4:
        print("\nRecursion.....Complete!\tApproximate recursion time =", time_taken, "units")
        print("\nRelabelling.....Initiated")
        # time.strftime("%T", time.gmtime(time_taken))
    elif status_num == 5:
        r_time = time.strftime("%T", time.gmtime(time_taken))
        print("Relabelling....Complete!\tExecution time = ", r_time)
        print("\n\tSub-Routine 3\nTime Evolution....Initiated")
    elif status_num == 6:
        evo_time = time.strftime("%T", time.gmtime(time_taken))
        print("\nTime Evolution....Complete!\tExecution time = ", evo_time)
        print("\nVon-Neumannn Entropy.....Initiated")
    elif status_num == 7:
        vn_time = time.strftime("%T", time.gmtime(time_taken))
        print("\nVon-Neumannn Entropy....Complete!\tExecution time = ", vn_time)
    elif status_num == 8:
        print("Complete!")
    else:
        warning('Invalid status')


# Handles non-fatal warnings
def warning(*objects):
    print("WARNING:", *objects, file=sys.stderr)


# Writes output to hard disk
def write_file(filename, data, fmt='%.18e'):
    # Check if Output directory exists
    if not os.path.exists('Output'):
        try:
            os.makedirs('Output')
        except OSError:
            pass

    print("Writing to {0}".format(filename))
    np.savetxt('Output/' + filename, data, delimiter=',', fmt=fmt)


# Reads from hard disk
def read_file(filename, dtype=np.float64):
    np.genfromtxt(filename, delimiter=',', dtype=dtype)


def plotting(x, y):
    # Plot area formatting
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='0.50', linestyle='-')
    plt.minorticks_on()
    plt.ylabel(r'Von-Neumann Entropy $[S_{VN} = - tr(\rho \ln(\rho))] \rightarrow$')
    plt.xlabel(r'Time* $[\tau]\rightarrow$')
    plt.title(r'Von-Neumann entropy ($S_{VN}$) vs time ($\tau$) for a 2D sub-lattice')

    # Entropy Plot
    plt.plot(x, y, 'bo', markersize=5, label=r'$S_{VN} = - tr(\rho \ln(\rho))$')
    plt.savefig('Output/Entropy.png', format='png', dpi=400)
    # plt.show()
    return
