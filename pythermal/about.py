"""
Command line output of information on PyTehermal and its dependencies. 
"""
from __future__ import print_function
import sys
import os
import platform
import numpy
import scipy
import multiprocessing
from main import __version__

__all__ = ['about_pythermal']


def about_pythermal():
    """
    Gives version numbers for PyThermal, 
    NumPy, SciPy, MatPlotLib and tqdm.
    
    """
    print("")
    print("PyThermal - Time evolving bosons on a 2D crystal lattice")
    print("Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics")
    print("")
    print("PyThermal version: {}".format(__version__))
    print("NumPy Version: {}".format(numpy.__version__))
    print("SciPy Version: {}".format(scipy.__version__))
    try:
        import matplotlib
        matplotlib_ver = matplotlib.__version__
    except:
        matplotlib_ver = 'None'
    print("Matplotlib Version: {}".format(matplotlib_ver))
    print("Python Version: {}{}{}".format(sys.version_info[0:3]))
    print("Number of CPUs: {}".format(multiprocessing.cpucount())
    print("Platform Info: {} {}".format(platform.system(), platform.machine()))
    print("")

if __name__ == "__main__":
    about_pythermal()