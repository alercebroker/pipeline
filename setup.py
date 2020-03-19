from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

module_name = "mhps"
pyx_path = [f"{module_name}/{module_name}_wrapper.pyx", f"{module_name}/lib/functions.c"]
headers_path = [f"{module_name}/lib/functions.h"]
lib_path = f"{module_name}/lib"

paps_extension = Extension(
    name=f"{module_name}.{module_name}_wrapper",
    sources=pyx_path,
    depends=headers_path,
    include_dirs=[lib_path,numpy.get_include(), lib_path]
)

setup(
    name=module_name,
    ext_modules=cythonize([paps_extension]),
    install_requires=["numpy==1.17.4","Cython==0.29.12"],
    build_requires=["Cython==0.29.12","numpy==1.17.4"],
    packages=[module_name],
    url='https://github.com/alercebroker/paps',
    description='Patricia Arevalo Power Spectra https://arxiv.org/abs/1207.5825',
    author='ALeRCE',
    version='0.0.1',
)
