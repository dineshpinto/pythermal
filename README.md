#### This branch is currently in beta. 

## PyThermal - Time evolving fermions on a 2D crystal lattice

**Thermalization and Quantum Entanglement Project Group, St. Stephen's Centre for Theoretical Physics**

*Project Mentor: Dr. A. Gupta*   
*Project Students: A. Kumar, D. Pinto and M. Ghosh*

Program to simulate n-particles on a 2D lattice, which is divided into sub-lattices A and B after
deletion of sites. The variation in Von-Neumann entropy of these sub-lattices is then studied.

## Cython

Cython implementation of [PyThermal](https://github.com/dkpinto/PyThermal). Cython is an optimising static compiler for both the Python programming language and the extended Cython programming language. 


Around **7%** faster as of latest testing.  

To  generate shared objects:

        $ python setup.py build_ext --inplace
        
To execute code:

        $ python __init__.py
        
