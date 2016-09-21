#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import
from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('bvcopula', parent_package, top_path)
    config.add_subpackage('copula')
    # UNIT TESTING
    config.add_data_dir('tests')
    config.add_data_dir('tests/data')
    return config


if __name__ == "__main__":
    setup(**configuration(top_path='').todict())
