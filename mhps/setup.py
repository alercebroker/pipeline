from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import os
import numpy as np

module_name = "mhps"
pyx_path = [f"{module_name}/{module_name}_wrapper.pyx", f"{module_name}/lib/functions.c"]
headers_path = [f"{module_name}/lib/functions.h"]
lib_path = f"{module_name}/lib"

library_dirs = []
if os.name == 'nt':  # Windows, assumming MSVC compiler
    libraries = []
    compiler_args = ['/Ox', '/fp:fast']
elif os.name == 'posix':  # UNIX, assumming GCC compiler
    libraries = ['m']
    compiler_args = ['-O3', '-ffast-math', '-march=x86-64-v3', '-flto']
else:
    raise Exception('Unsupported operating system')

mhps_extension = Extension(
    name=f"{module_name}.{module_name}_wrapper",
    sources=pyx_path,
    extra_compile_args=compiler_args,
    libraries=libraries,
    library_dirs=library_dirs,
    depends=headers_path,
    include_dirs=[lib_path, np.get_include()]
)

setup(
    name=module_name,
    ext_modules=cythonize([mhps_extension], force=True),
    install_requires=["numpy>=1.17.4", "Cython>=0.29.12"],
    build_requires=["Cython>=0.29.12", "numpy>=1.17.4"],
    packages=[module_name],
    url='https://github.com/alercebroker/mhps',
    description='Mexican Hat Power Spectra https://arxiv.org/abs/1207.5825',
    author='ALeRCE',
    version='0.1.1',
)
