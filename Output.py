import time

import humanize as hu
import tabulate
import numpy as np
import matplotlib.pyplot as plt

import Main


def status_output(status, time_taken=0.0):
    global h_time, e_time, r_time, evo_time

    if status == 1:
        print "\n\tInitialization\nEigenstates........Complete!\n\tSub-Routine 1\nHamiltonian........Initiated\n"

    elif status == 2:
        h_time = time.strftime("%T", time.gmtime(time_taken))
        print "\nHamiltonian........Complete!\tExecution time = ", h_time
        print "\nDiagonalization....Initiated"

    elif status == 3:
        e_time = time.strftime("%T", time.gmtime(time_taken))
        print "\nDiagonalization....Complete!\tExecution time = ", e_time
        print "\n\tSub-Routine 2\nRelabelling.....Initiated"

    elif status == 4:
        print "\nMinimum recursion time =", time_taken, "seconds or ", time_taken / 86400, "days"
        # time.strftime("%T", time.gmtime(time_taken))

    elif status == 5:
        r_time = time.strftime("%T", time.gmtime(time_taken))
        print "\nRelabelling....Complete!\tExecution time = ", r_time
        print "\n\tSub-Routine 3\nTime Evolution.....Initiated"

    elif status == 6:
        evo_time = time.strftime("%T", time.gmtime(time_taken))
        print "\nTime Evolution....Complete!\tExecution time = ", evo_time
        print "\nVon-Neumannn Entropy.....Initiated"

    else:
        exit()

    return


def printout(nos, eigenstates, hamiltonian, eigenvalues, eigenvectors):
    print 'nos =', len(eigenstates), '\neigenstates shape =', eigenstates.shape, ' eigenstates size =', \
        eigenstates.size, '\n\neigenstates=\n', eigenstates, '\n'
    print '\n\nhamiltonian_2d shape =', hamiltonian.shape, ' hamiltonian_2d size =', hamiltonian.size, \
        '\n\nhamiltonian_2d=\n', hamiltonian
    print '\n\neigenvalues shape =', eigenvalues.shape, ' eigenvalues size =', eigenvalues.size, '\n\neigenvalues=\n', \
        eigenvalues
    print "\n\neigenvectors shape =", eigenvectors.shape, ' eigenvectors size =', eigenvectors.size, \
        '\n\neigenvectors=\n', eigenvectors

    # Calculate sizes
    h_size = hu.naturalsize(hamiltonian.nbytes)
    evec_size = hu.naturalsize(eigenvectors.nbytes)
    eval_size = hu.naturalsize(eigenvalues.nbytes)

    print '\nsize of hamiltonian array =', h_size
    print 'size of eigenvector array =', evec_size
    print 'size of eigenvalue array =', eval_size

    s = Main.System()

    headers = ['', 'nop', 'nol', 'nos', 'h_time', 'e_time', 'd_time', 'h_size', 'evec_size', 'eval_size',
               'den_size']
    body = [time.strftime("%Y-%m-%d %T"), s.nop, s.nol, nos, h_time, e_time, h_size, evec_size,
            eval_size]
    table = tabulate.tabulate([body], headers=headers)

    with open('LOG.txt', 'a') as f:
        f.write(table)
        f.write('\n\n')

    # Output to .csv files (CAUTION when storing large(>100 million elements) files, NOTE file size before proceeding)
    np.savetxt('Eigenstates.csv', eigenstates, delimiter=',', fmt='%1d')
    np.savetxt('Hamiltonian_2d.csv', hamiltonian, delimiter=',', fmt='%1d')
    np.savetxt('Eigenvalues.csv', eigenvalues, delimiter=',', fmt='%0.4e')
    np.savetxt('Eigenvectors.csv', eigenvectors, delimiter=',', fmt='%0.4e')

    return


def plotting(x, y):
    plt.plot(x, y, 'bo', label='Entropy')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='0.50', linestyle='-')
    plt.minorticks_on()
    plt.legend(loc='best')
    plt.ylabel(r'Von-Neumann Entropy $\rightarrow$')
    plt.xlabel(r'Time $\rightarrow$')
    plt.title(r'Von-Neumann entropy vs time for a 2D sub lattice')
    plt.savefig('Entropy-Time.png', format='png', dpi=600)
    plt.show()

    return
