# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Time evolving hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics, New Delhi
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import print_function

import multiprocessing
import platform
import sys

from main import __version__

__all__ = ['about']


# noinspection PyUnresolvedReferences
def about():
    """
    Checks dependencies and return version numbers for PyThermal.
    """

    print("PyThermal - Time evolving bosons on a 2D crystal lattice")
    print("Thermalization and Quantum Entanglement Project Group, "
          "St. Stephen's Centre for Theoretical Physics")
    print("")
    print("PyThermal version: {}".format(__version__))
    try:
        import numpy
        print("NumPy Version: {}".format(numpy.__version__))
    except ImportError:
        sys.stderr.write("[REQUIRED] Unable to import the 'numpy' module")

    try:
        import scipy
        print("SciPy Version: {}".format(scipy.__version__))
    except ImportError:
        sys.stderr.write("[REQUIRED] Unable to import the 'scipy' module")

    try:
        import matplotlib
        print("Matplotlib Version: {}".format(matplotlib.__version__))
    except ImportError:
        sys.stderr.write("[REQUIRED] Unable to import the 'matplotlib' module")

    try:
        import tqdm
        print("tqdm Version: {}".format(tqdm.__version__))
    except ImportError:
        sys.stderr.write("[REQUIRED] Unable to import the 'tqdm' module")

    print("Python Version: {}".format(platform.python_version()))
    print("Number of CPUs: {}".format(multiprocessing.cpu_count()))
    print("Platform Info: {} {}".format(platform.system(), platform.machine()))
    print("")

if __name__ == "__main__":
    about()
