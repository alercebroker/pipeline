from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding


# Get the long description from the relevant file
def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='turbofats',
    version='0.1',
    description='Library with compilation of features for time series',
    long_description=readme(),
    url='https://github.com/alercebroker/turbo-fats',
    download_url='https://github.com/alercebroker/turbo-fats',
    author='ALeRCE Team',
    author_email='contact@alerce.online',
    license='MIT licence',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'Programming Language :: Python :: 3.6',
    ],
    keywords='times series features, light curves',
    packages=['turbofats'],
    include_package_data=True,
    zip_safe=False, install_requires=['numpy', 'numba', 'scipy', 'statsmodels']
)
