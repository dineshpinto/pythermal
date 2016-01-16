from __future__ import division, print_function

import itertools as it
import multiprocessing as mp

import numpy as np
import scipy.linalg as la
from tqdm import tqdm


# Returns eigenstates for the given lattice sites and particles
def eigenstates_lattice(lat, nop, lat_del_pos):
    if np.size(lat_del_pos) is not 0:
        lat_del = np.delete(lat, lat_del_pos - 1)
        eigenstates = np.array(list(it.combinations(lat_del, nop)), dtype=np.int32)
    else:
        eigenstates = np.array(list(it.combinations(lat, nop)), dtype=np.int32)
    return eigenstates, len(eigenstates)


# Generates hamiltonian from the system's eigenstates
def hamiltonian_2d(start, stop, nos, nsa, nop, eigenstates, queue, h):
    for j in tqdm(range(start, stop)):  # Start/Stop defined by distribute()
        for k in range(nos):  # k iterates over all possibilities
            c = np.intersect1d(eigenstates[j], eigenstates[k])
            c_sum = np.sum(c, dtype=np.int32)  # Sum of common elements
            c_size = np.size(c)  # No. of common elements
            j_sum = np.sum(eigenstates[j], dtype=np.int32)  # Sum of elements of m[j]
            k_sum = np.sum(eigenstates[k], dtype=np.int32)  # Sum of elements of m[k]

            if c_size == nop - 1:  # Only one element differs
                if abs(j_sum - k_sum) == nsa:  # Element differs by dimension
                    h[j, k] = float(1)
                elif (k_sum - j_sum) == 1 and not (j_sum - c_sum) % nsa == 0:  # Right/Left edge
                    h[j, k] = float(1)
                elif (j_sum - k_sum) == 1 and not (j_sum - c_sum) % nsa == 1:  # Right/Left edge
                    h[j, k] = float(1)
                else:
                    continue
            else:
                continue
    queue.put(h)
    return


# Distributes processes among processors
def distribute(n_items, n_processes, i):
    # Defines no. of items ([j] index in Hamiltonian) per process and starting point
    items_per_process = n_items // n_processes  # Integer division
    start = i * items_per_process

    # For last process, appends all remaining items to last core
    if i == n_processes - 1:
        stop = n_items
    else:
        stop = items_per_process * (i + 1)

    return start, stop


# Uses parallel processes to call function, automatically reads no. of cores
def parallel_call_hamiltonian(e_states, nos, nsa, nop):
    process_list = []
    queue = mp.Queue()  # Setting up a queue to store each processes' output
    h = np.zeros(shape=(nos, nos), dtype=np.float32)
    n_processes = mp.cpu_count()  # No. of processes to create for parallel processing

    for i in range(n_processes):
        start, stop = distribute(nos, n_processes, i)  # Start, stop points from distribute()
        args = (start, stop, nos, nsa, nop, e_states, queue, h)
        process = mp.Process(target=hamiltonian_2d, args=args)
        process_list.append(process)  # Create list of processes
        process.start()

    for i in range(n_processes):  # Retrieves output from queue
        h += queue.get()

    while not queue.empty():  # Clear queue
        h += queue.get()

    for jobs in process_list:  # Joins processes together
        jobs.join()

    return h


# Calculates eigenvectors and eigenvalues used Pade algorithm (link to OpenBLAS Fortran libraries)
def eigenvalvec(h):
    eigenvalues, eigenvectors = la.eig(h)

    # Sort eigenvalues and eigenvectors by ascending eigenvalue
    idx = eigenvalues.argsort()
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    
    return eigenvalues.real, eigenvectors
