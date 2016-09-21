#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import
from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('mvar', parent_package, top_path)
    config.add_data_dir('tests')
    return config


if __name__ == "__main__":
    setup(**configuration(top_path='').todict())
