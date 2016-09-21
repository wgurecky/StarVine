#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import
from numpy.distutils.core import setup


def configuration(parent_pacakge='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_pacakge, top_path)
    config.set_options(assume_default_configuration=True,
                       quiet=True)
    config.add_subpackage('starvine')
    config.add_data_files(('starvine', '*.txt'))
    return config

def setup_package():
    metadata = dict(name='StarVine',
          version='0.0.1',
          description='C- and D-Vine copula library',
          author='William Gurecky',
          test_suite="tests",
          platforms=["Linux", "Mac OS-X"],
          build_requires=['numpy>=1.8.0'],
          install_requires=['numpy>=1.8.0', 'scipy>=0.13',
                            'pandas>=0.14.0', 'h5py>=2.2.0',
                            'seaborn>=0.7.0', 'networkx>=1.8.1',
                            'emcee>=2.0.0'],
          package_data={'': ['*.txt']},
          license='BSD-3clause',
          author_email='william.gurecky@utexas.edu',
          )

    metadata['configuration'] = configuration
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
