from __future__ import print_function

# To maintain forward compatibility with Py3
try:
    import Tkinter
    import ttk
except ImportError:
    import tkinter as Tkinter
    import tkinter.ttk as ttk

import Main

__author__ = 'D. Pinto'


def fetch(entries):
    idx = 0
    for field in fields:
        print('{} = {}'.format(field, entries[idx].get()))
        idx += 1

    print('Show images during execution =', entries[idx].get())
    print('Initialize with first eigenvector of entire system =', entries[idx + 1].get())
    print('\n')


# Defines GUI Layout. Returns list of data input from GUI
def gui(base, text_fields):
    base.wm_title('PyThermal')
    entries = []
    # Variables for checkboxes
    var1, var2 = Tkinter.IntVar(), Tkinter.IntVar()

    for field in text_fields:
        row = ttk.Frame(base)
        label = ttk.Label(row, width=22, text=field + ':', anchor='w')
        entry = ttk.Entry(row)
        entry.insert(0, '0')
        row.pack(padx=8, pady=8)
        label.pack(side=Tkinter.LEFT, expand=True)
        entry.pack(side=Tkinter.RIGHT, expand=True)
        entries.append(entry)

    chk = ttk.Checkbutton(base, text='Show images during execution', variable=var1)
    chk.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8)
    entries.append(var1)
    chk2 = ttk.Checkbutton(base, text='Initialize with first eigenvector of entire system', variable=var2)
    chk2.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, padx=8, pady=8)
    entries.append(var2)

    return entries


# Call main() from Main and pass entries as arguments
def execute(entries):
    initial_values = [float(e.get()) for e in entries]
    try:
        Main.main(initial_values)
    except Exception as e:
        print(e)
        pass


if __name__ == '__main__':
    # Set input field text
    fields = (
    'No. of particles', 'Shape of lattice', 'No. of sites in lattice A', 'Start evolving at', 'Stop evolving at',
    'Time steps')
    root = Tkinter.Tk()
    values = gui(root, fields)
    root.bind('<Return>', lambda event: execute(values))
    # Create buttons and assign tasks
    b1 = ttk.Button(root, text='Execute', command=lambda: execute(values))
    b1.pack(side=Tkinter.RIGHT, padx=8, pady=8)
    b2 = ttk.Button(root, text='Close', command=root.quit)
    b2.pack(side=Tkinter.LEFT, padx=8, pady=8)
    b3 = ttk.Button(root, text='Print', command=lambda: fetch(values))
    b3.pack(side=Tkinter.RIGHT, padx=8, pady=8)

    root.mainloop()
