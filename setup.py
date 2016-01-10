from distutils.core import setup 
from distutils.extension import Extension
from Cython.Build import cythonize 


extensions = [Extension('Main', ['Main.pyx']), Extension('SubRoutine1', ['SubRoutine1.pyx']), Extension('SubRoutine2', ['SubRoutine2.pyx']), 
                Extension('SubRoutine3', ['SubRoutine3.pyx']), Extension('Output', ['Output.pyx'])]

setup(ext_modules = cythonize(extensions))

