# This file is a part of PyThermal. https://github.com/dkpinto/PyThermal
#
# PyThermal - Thermal equilibrium of hard-core bosons on a 2D crystal lattice
# Thermalization and Quantum Entanglement Project Group
# St. Stephen's Centre for Theoretical Physics, New Delhi
#
# Project Mentor: Dr. A. Gupta
# Project Students: A. Kumar, D. Pinto and M. Ghosh

from __future__ import print_function, division, absolute_import

import multiprocessing
import os
import platform
import sys

from main import __version__

__all__ = ['about']


# noinspection PyUnresolvedReferences
def about(test=True):
    """
    Checks dependencies and return version numbers for PyThermal.
    Tests NumPy and SciPy. Checks OpenBLAS for Numpy.
    :param test: Run tests for NumPy and SciPy

    """
    print("PyThermal - Time evolving bosons on a 2D crystal lattice")
    print("Thermalization and Quantum Entanglement Project Group, "
          "St. Stephen's Centre for Theoretical Physics")
    print("")
    print("PyThermal Version: {}".format(__version__))
    print("Python Version: {}".format(platform.python_version()))
    print("Number of CPUs: {}".format(multiprocessing.cpu_count()))

    try:
        ram = check_ram()
    except Exception:
        pass
    else:
        print("Available RAM: {} GB".format(ram))

    print("Platform Info: {} {}".format(platform.system(), platform.machine()))

    try:
        import matplotlib
        print("Matplotlib Version: {}".format(matplotlib.__version__))
    except ImportError:
        sys.stderr.write("[REQUIRED] Unable to import the 'matplotlib' module")
        print("")

    try:
        import tqdm
        print("Tqdm Version: {}".format(tqdm.__version__))
    except ImportError:
        sys.stderr.write("[REQUIRED] Unable to import the 'tqdm' module")
        print("")

    try:
        import numpy
        print("NumPy Version: {}".format(numpy.__version__))
        if test:
            test_numpy()
    except ImportError:
        sys.stderr.write("[REQUIRED] Unable to import the 'numpy' module")
        print("")

    try:
        import scipy
        print("SciPy Version: {}".format(scipy.__version__))
        if test:
            test_scipy()
    except ImportError:
        sys.stderr.write("[REQUIRED] Unable to import the 'scipy' module")
        print("")
    print("Complete.")


def check_ram():
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    mem_gib = mem_bytes / (1024 ** 3)
    return round(mem_gib, 2)


def test_numpy():
    from numpy.distutils.system_info import get_info
    import timeit
    import warnings

    warnings.filterwarnings('ignore', category=UserWarning)

    print("Testing NumPy...")
    info = get_info('blas_opt')
    print('\tBLAS info:')
    for kk, vv in info.items():
        print('\t * ' + kk + ' ' + str(vv))

    setup = "import numpy; x = numpy.random.random((1000, 1000))"
    count = 10

    t = timeit.Timer("numpy.dot(x, x.T)", setup=setup)
    print("\tdot: {} sec".format(t.timeit(count) / count))


def test_scipy():
    import timeit

    print("Testing SciPy...")
    setup = "import numpy;\
            import scipy.linalg as linalg;\
            x = numpy.random.random((1000,1000));\
            z = numpy.dot(x, x.T)"
    count = 5

    t = timeit.Timer("linalg.cholesky(z, lower=True)", setup=setup)
    print("\tcholesky: {} sec".format(t.timeit(count) / count))

    t = timeit.Timer("linalg.svd(z)", setup=setup)
    print("\tsvd: {} sec".format(t.timeit(count) / count))


if __name__ == "__main__":
    about()
