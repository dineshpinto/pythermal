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
           'Manually define sub-lattices A and B (optional, create text \nfiles "a.txt" and "b.txt" in same folder '
           'with lattice sites as columns)']

fields_func = ['Hamiltonian of whole system', 'Hamiltonian of sub-lattice A', 'Eigenvalues of whole system',
               'Eigenvectors of whole system', 'Eigenvalues of sub-lattice A', 'Eigenvectors of sub-lattice A',
               r'Time Evolution Psi(t)', 'Avg. particles in A', 'Avg. particles in B', 'Von-Neumann entropy of B',
               'Purity of B']


def fetch(values, values2):
    """
    Prints data stored in entries for debugging purposes.
    :param values: List storing initial values
    :param values2: List storing sub-lattices A and B

    """
    for idx, value in enumerate(values):
        print('{}. {} = {}'.format(idx, fields[idx], value.get()))

    for idx, value2 in enumerate(values2):
        print('{}. {} = {}'.format(idx, fields2[idx], value2.get()))

    if values2[2].get():
        try:
            print('A = ', np.genfromtxt('a.txt', dtype=np.int32))
            print('B = ', np.genfromtxt('b.txt', dtype=np.int32))
        except IOError as e:
            print(e, traceback.format_exc())
            pass


def graphical_interface(base):
    """
    Base layout for text fields and their labels.
    :param base: Root
    :return: List of entries

    """
    # Variables for checkboxes
    var1, var2, var3 = Tkinter.IntVar(), Tkinter.IntVar(), Tkinter.IntVar()
    initial_values = []
    optional_values = []

    for field in fields:
        row = ttk.Frame(base)

        label = ttk.Label(row, width=40, text=field + ':', anchor='w')
        entry = ttk.Entry(row)

        entry.insert(0, '0')
        row.pack(padx=8, pady=8, expand=True)
        label.pack(side=Tkinter.LEFT, expand=True)
        entry.pack(side=Tkinter.RIGHT, expand=True)
        initial_values.append(entry)

    chk = ttk.Checkbutton(base, text=fields2[0], variable=var1)
    chk.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8, expand=True)
    optional_values.append(var1)

    chk2 = ttk.Checkbutton(base, text=fields2[1], variable=var2)
    chk2.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8, expand=True)
    optional_values.append(var2)

    chk = ttk.Checkbutton(base, text=fields2[2], variable=var3)
    chk.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8, expand=True)
    optional_values.append(var3)

    return initial_values, optional_values


def execute(initial_values, optional_values):
    """
    Calls main() from main.py.
    :param optional_values: List of optional values
    :param initial_values: List of initial values

    """
    fetch(initial_values, optional_values)
    initial_values = [float(e.get()) for e in initial_values]
    optional_values = [float(e.get()) for e in optional_values]

    if optional_values[2]:
        lat_a = np.genfromtxt('a.txt', dtype=np.int32)
        lat_b = np.genfromtxt('b.txt', dtype=np.int32)
    else:
        lat_a, lat_b = None, None

    try:
        main.main(initial_values, optional_values, lat_a, lat_b)
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

    init_values, opt_values = graphical_interface(root)

    # Enter/Return key will execute
    root.bind('<Return>', lambda event: execute(init_values, opt_values))

    # Create buttons and assign tasks
    b1 = ttk.Button(root, text='Execute', command=lambda: execute(init_values, opt_values))
    b1.pack(side=Tkinter.RIGHT, padx=8, pady=8)

    b2 = ttk.Button(root, text='Close', command=root.quit)
    b2.pack(side=Tkinter.LEFT, padx=8, pady=8)

    b3 = ttk.Button(root, text='Print', command=lambda: fetch(init_values, opt_values))
    b3.pack(side=Tkinter.RIGHT, padx=8, pady=8)

    root.mainloop()
