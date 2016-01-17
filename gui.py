"""
This file is a part of PyThermal. https://github.com/dkpinto/PyThermal

PyThermal - Time evolving fermions on a 2D crystal lattice
Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics

Project Mentor: Dr. A. Gupta
Project Students: A. Kumar, D. Pinto and M. Ghosh
"""

from __future__ import print_function

try:
    import tkinter as Tkinter
    import tkinter.ttk as ttk
except ImportError:
    import Tkinter
    import ttk

import main

__author__ = 'D. Pinto'

# Set input field text
fields = ['Total no. of particles', 'Shape of lattice', 'No. of sites in sub-lattice A', 'Start evolving at',
          'Stop evolving at', 'Time steps', 'Initial Eigenvector (Ground state = 0)']

fields_func = ['Hamiltonian of whole system', 'Hamiltonian of sub-lattice A', 'Eigenvalues of whole system',
               'Eigenvectors of whole system', 'Eigenvalues of sub-lattice A', 'Eigenvectors of sub-lattice A',
               r'Time Evolution Psi(t)', 'Avg. particles in A', 'Avg. particles in B', 'Von-Neumann entropy of B',
               'Purity of B']

text_data = \
    'OPTIONAL\nSelect variables currently saved on disk. Overrides automatic \nfilesystem check. If none selected, ' \
    'automatic checking is \nperformed.'


def fetch(entries):
    for idx, field in enumerate(fields):
        print('{} = {}'.format(field, entries[idx].get()))

    print('Show images during execution =', entries[idx].get())
    print('Initialize with first eigenvector of entire system =', entries[idx + 1].get())
    print('\n')


# Defines GUI Layout. Returns list of data input from GUI
def gui(base, text_fields):
    base.wm_title('PyThermal')

    # Variables for checkboxes
    var1, var2 = Tkinter.IntVar(), Tkinter.IntVar()
    entries = []

    for field in text_fields:
        row = ttk.Frame(base)

        label = ttk.Label(row, width=32, text=field + ':', anchor='w')
        entry = ttk.Entry(row)

        entry.insert(0, '0')
        row.pack(padx=8, pady=8)
        label.pack(side=Tkinter.LEFT, expand=True)
        entry.pack(side=Tkinter.RIGHT, expand=True)
        entries.append(entry)

    chk = ttk.Checkbutton(base, text='Show images during execution', variable=var1)
    chk.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8, expand=True)
    entries.append(var1)

    chk2 = ttk.Checkbutton(base, text='Initialize with eigenvector of entire system', variable=var2)
    chk2.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8, expand=True)
    entries.append(var2)

    return entries


def gui2(base, functions, text):
    entries = [Tkinter.IntVar()] * len(functions)

    row = ttk.Frame(base)

    label = ttk.Label(row, text=text, anchor='e')
    row.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8, expand=True)
    label.pack(side=Tkinter.LEFT, expand=True)

    for idx, field in enumerate(functions):
        row2 = ttk.Frame(base)
        chk = ttk.Checkbutton(base, text=field, variable=entries[idx])
        row2.pack(pady=3)
        chk.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, expand=True)

    return entries


# Call main() from Main and pass entries as arguments
def execute(entries, entries2):
    initial_values = [float(e.get()) for e in entries]
    function_values = [float(e.get()) for e in entries2]

    try:
        main.main(initial_values, function_values)
    except Exception as e:
        print(e)
        pass


if __name__ == '__main__':
    root = Tkinter.Tk()

    values = gui(root, fields)
    values2 = gui2(root, fields_func, text_data)

    # Enter/Return key will execute
    root.bind('<Return>', lambda event: execute(values, values2))

    # Create buttons and assign tasks
    b1 = ttk.Button(root, text='Execute', command=lambda: execute(values, values2))
    b1.pack(side=Tkinter.RIGHT, padx=8, pady=8)

    b2 = ttk.Button(root, text='Close', command=root.quit)
    b2.pack(side=Tkinter.LEFT, padx=8, pady=8)

    b3 = ttk.Button(root, text='Print', command=lambda: fetch(values))
    b3.pack(side=Tkinter.RIGHT, padx=8, pady=8)

    root.mainloop()
