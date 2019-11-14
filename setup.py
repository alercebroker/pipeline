from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

module_name = "paps"
pyx_path = f"{module_name}/functions.pyx"
lib_path = f"{module_name}/lib"

examples_extension = Extension(
    name=module_name,
    sources=[pyx_path],
    libraries=["functions"],
    library_dirs=[lib_path],
    include_dirs=[lib_path,numpy.get_include()]
)

setup(
    name=module_name,
    ext_modules=cythonize([examples_extension]),
    install_requires=["numpy==1.17.4"],
    build_requires=["Cython==0.29.12","numpy==1.17.4"]
)
