#!/usr/bin/env python2
from setuptools import setup, find_packages

setup(name='StarVine',
      version='0.0.1',
      description='C- and D-Vine copula library',
      author='William Gurecky',
      packages=find_packages(),
      test_suite="tests",
      install_requires=['numpy>=1.8.0', 'scipy>=0.13',
                        'pandas>=0.14.0', 'h5py>=2.2.0',
                        'seaborn>=0.7.0', 'networkx>=1.8.1',
                        'emcee>=2.0.0'],
      package_data={'': ['*.txt']},
      license='BSD-3clause',
      author_email='william.gurecky@utexas.edu',
      entry_points={
          'console_scripts': [
              'starvine = StarVine:main'
            ]
        }
)
