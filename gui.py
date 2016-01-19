# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import print_function

import os
import traceback

import numpy as np
try:
    import tkinter as Tkinter
    import tkinter.ttk as ttk
except ImportError:
    import Tkinter
    import ttk

import main

__author__ = 'D. Pinto'

# Set input text fields
fields = ['Total no. of particles', 'Shape of lattice', 'No. of sites in sub-lattice A', 'Start evolving at',
          'Stop evolving at', 'Time steps', 'Initial Eigenvector (Ground state = 0)', ]
fields2 = ['Show images during execution', 'Initialize with eigenvector of entire system (default is eigenvector of A)',
           'Manually define sub-lattices A and B (optional, in development)']

fields_func = ['Hamiltonian of whole system', 'Hamiltonian of sub-lattice A', 'Eigenvalues of whole system',
               'Eigenvectors of whole system', 'Eigenvalues of sub-lattice A', 'Eigenvectors of sub-lattice A',
               r'Time Evolution Psi(t)', 'Avg. particles in A', 'Avg. particles in B', 'Von-Neumann entropy of B',
               'Purity of B']


def fetch(values, lattice):
    """
    Prints data stored in entries for debugging purposes.
    :param values: List storing initial values
    :param lattice: List storing sub-lattices A and B

    """
    for idx, value in enumerate(values):
        if idx < len(fields):
            print('{}. {} = {}'.format(idx, fields[idx], value.get()))
        else:
            print('{}. {} = {}'.format(idx, fields2[idx - len(fields)], value.get()))

    print('Lattice A =', np.array(lattice[0].get()), '\nLattice B =', np.array(lattice[1].get()))
    print('\n')


def graphical_interface(base):
    """
    Base layout for text fields and their labels.
    :param base: Root
    :return: List of entries

    """
    # Variables for checkboxes
    var1, var2, var3 = Tkinter.IntVar(), Tkinter.IntVar(), Tkinter.IntVar()
    init_entries = []

    for field in fields:
        row = ttk.Frame(base)

        label = ttk.Label(row, width=40, text=field + ':', anchor='w')
        entry = ttk.Entry(row)

        entry.insert(0, '0')
        row.pack(padx=8, pady=8, expand=True)
        label.pack(side=Tkinter.LEFT, expand=True)
        entry.pack(side=Tkinter.RIGHT, expand=True)
        init_entries.append(entry)

    chk = ttk.Checkbutton(base, text=fields2[0], variable=var1)
    chk.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8, expand=True)
    init_entries.append(var1)

    chk2 = ttk.Checkbutton(base, text=fields2[1], variable=var2)
    chk2.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8, expand=True)
    init_entries.append(var2)

    chk = ttk.Checkbutton(base, text=fields2[2], variable=var3)
    chk.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8, expand=True)
    init_entries.append(var3)

    lattice_entries = []
    text_fields = ['A', 'B']

    for field in text_fields:
        row = ttk.Frame(base)

        label = ttk.Label(row, width=5, text=field + ':', anchor='w')
        entry = ttk.Entry(row, width=50)

        entry.insert(0, '[0]')
        row.pack(padx=8, pady=8, expand=50)
        label.pack(side=Tkinter.LEFT, expand=True)
        entry.pack(side=Tkinter.RIGHT, expand=True)
        lattice_entries.append(entry)

    return init_entries, lattice_entries


def execute(initial_entries, lattice_entries):
    """
    Calls main() from main.py.
    :param initial_entries: List of initial values
    :param lattice_entries: List of unctions to read from disk

    """
    fetch(initial_entries, lattice_entries)

    initial_values = [float(e.get()) for e in initial_entries]

    lat_a = np.array(lattice_entries[0].get())
    lat_b = np.array(lattice_entries[1].get())

    try:
        main.main(initial_values, lat_a, lat_b)
    except Exception as e:
        print(e, traceback.format_exc())
        pass


if __name__ == '__main__':
    root = Tkinter.Tk()
    root.title('PyThermal')

    # Setting icon for window
    try:
        if os.name is 'nt':
            root.iconbitmap('meta/icon.ico')
        else:
            root.iconbitmap('@meta/icon-0.xbm')
    except Tkinter.TclError:
        pass

    init_values, lat = graphical_interface(root)

    # Enter/Return key will execute
    root.bind('<Return>', lambda event: execute(init_values, lat))

    # Create buttons and assign tasks
    b1 = ttk.Button(root, text='Execute', command=lambda: execute(init_values, lat))
    b1.pack(side=Tkinter.RIGHT, padx=8, pady=8)

    b2 = ttk.Button(root, text='Close', command=root.quit)
    b2.pack(side=Tkinter.LEFT, padx=8, pady=8)

    b3 = ttk.Button(root, text='Print', command=lambda: fetch(init_values, lat))
    b3.pack(side=Tkinter.RIGHT, padx=8, pady=8)

    root.mainloop()
