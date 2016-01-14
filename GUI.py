from __future__ import print_function

# To maintain forward compatibility
try:
    import Tkinter
    import ttk
except ImportError:
    import tkinter as Tkinter
    import tkinter.ttk as ttk

import Main


def gui(base, text_fields):
    base.wm_title('PyThermal')
    entries = []
    var = Tkinter.IntVar()
    var2 = Tkinter.IntVar()
    for field in text_fields:
        row = ttk.Frame(base)
        lab = ttk.Label(row, width=22, text=field + ":", anchor='w')
        ent = ttk.Entry(row)
        row.pack(padx=8, pady=8)
        lab.pack(side=Tkinter.LEFT, expand=True)
        ent.pack(side=Tkinter.RIGHT, expand=True)
        entries.append(ent)

    chk = ttk.Checkbutton(base, text='Show Output Images', variable=var)
    chk.pack(side=Tkinter.TOP, padx=5, pady=5)
    entries.append(var)
    chk2 = ttk.Checkbutton(base, text='Initialize with first eigenvector of entire system', variable=var2)
    chk2.pack(side=Tkinter.TOP, padx=5, pady=5)
    entries.append(var2)

    return entries


def execute(base, entries):
    initial_values = [float(e.get()) for e in entries]
    try:
        Main.main(initial_values)
    except Exception as e:
        print(e)
        pass


if __name__ == '__main__':
    fields = (
        'No. of particles', 'Shape of lattice', 'No. of sites in lattice A', 'Start evolving at', 'Stop evolving at',
        'Time steps')
    root = Tkinter.Tk()
    ents = gui(root, fields)
    root.bind('<Return>', (lambda event, e=ents: execute(root, e)))
    b1 = ttk.Button(root, text='Execute', command=(lambda e=ents: execute(root, e)))
    b1.pack(side=Tkinter.RIGHT, padx=8, pady=8)
    b2 = ttk.Button(root, text='Quit', command=root.quit)
    b2.pack(side=Tkinter.RIGHT, padx=8, pady=8)
    root.mainloop()
