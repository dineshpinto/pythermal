import multiprocessing as mp

import numpy as np
import scipy.linalg as la


# Hamiltonian called by parallel_call_hamiltonian, is based on the conjecture
def hamiltonian_2d(start, stop, nos, nsa, nop, eigenstates, queue, h):
    for j in xrange(start, stop):  # Start/Stop defined by distribute()
        for k in xrange(nos):  # k iterates over all possibilities
            c = np.intersect1d(eigenstates[j], eigenstates[k], assume_unique=True)
            c_sum = np.sum(c, dtype=np.int32)  # Sum of common elements
            c_size = np.size(c)  # No. of common elements
            j_sum = np.sum(eigenstates[j], dtype=np.int32)  # Sum of elements of m[j]
            k_sum = np.sum(eigenstates[k], dtype=np.int32)  # Sum of elements of m[k]

            if c_size == nop - 1:  # Only one element differs
                if abs(j_sum - k_sum) == nsa:  # Element differs by dimension
                    h[j][k] = float(1)
                elif (k_sum - j_sum) == 1 and not (j_sum - c_sum) % nsa == 0:  # Right/Left edge
                    h[j][k] = float(1)
                elif (j_sum - k_sum) == 1 and not (j_sum - c_sum) % nsa == 1:  # Right/Left edge
                    h[j][k] = float(1)
                else:
                    continue
            else:
                continue

    queue.put(h)
    return


# Distribution function used to distribute processes among processors
def distribute(n_items, n_processes, i):
    nitems_per_process = int(n_items / n_processes)  # Defines no. of [j] per process
    start = i * nitems_per_process  # Start adjusted depending on the process

    if i == n_processes - 1:  # For last process, appends remaining items to last core
        stop = n_items
    else:
        stop = nitems_per_process * (i + 1)

    return start, stop


# Uses parallel processes to call the Hamiltonian function, automatically reads no. of cores and extends itself
def parallel_call_hamiltonian(e_states, nos, nsa, nop):
    process_list = []
    queue = mp.Queue()  # Setting up queue to store each processes' output
    h = np.zeros(shape=(nos, nos), dtype=np.float32)
    n_processes = mp.cpu_count()  # No. of processes to create for parallel processing of Hamiltonian

    for i in xrange(n_processes):  # Iterate over the no. of processes
        start, stop = distribute(nos, n_processes, i)  # Start, stop points from distribution function
        args = (start, stop, nos, nsa, nop, e_states, queue, h)
        process = mp.Process(target=hamiltonian_2d, args=args)
        process_list.append(process)  # Create list of processes
        process.start()
        # print '(start, stop) = (', start, ',', stop, ') -- process ', i + 1, '-- PID', process.pid

    for i in xrange(n_processes):  # Retrieves output from queue
        h += queue.get()

    while not queue.empty():  # Clear queue(optional)
        h += queue.get()

    for jobs in process_list:  # Joins processes together
        jobs.join()

    return h


# Calculates eigenvectors and eigenvalues using the Node algorithm(..link..)
def eigenvalvec(h):
    e_val, e_vec = la.eig(h)
    return e_val.real, e_vec
