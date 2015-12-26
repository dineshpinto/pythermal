from __future__ import print_function

import sys
import time

import matplotlib.pyplot as plt


def status(status_num, time_taken=0.0):
    # global h_time, e_time, r_time, evo_time
    
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
        exit('Invalid status')
    return


# Warning handling
def warning(*objects):
    print("WARNING:", *objects, file=sys.stderr)


# Output when writing to hard disk
def write():
    print("Writing changes to file")


def plotting(x, y):
    # Plot Area
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='0.50', linestyle='-')
    plt.minorticks_on()
    plt.ylabel(r'Von-Neumann Entropy $[S_{VN} = - tr(\rho \ln(\rho))] \rightarrow$')
    plt.xlabel(r'Computer Time $[\tau]\rightarrow$')
    plt.title(r'Von-Neumann entropy vs time for a 2D sub-lattice')

    # Entropy Plot
    plt.plot(x, y, 'b-', markersize=5, label=r'$S_{VN} = - tr(\rho \ln(\rho))$')
    plt.savefig('Output/Entropy.png', format='png', dpi=400)
    plt.show()
    return
