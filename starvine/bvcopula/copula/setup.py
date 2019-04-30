#!/usr/bin/env python3

from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('copula', parent_package, top_path)
    config.add_subpackage('mvtdstpack')
    return config


if __name__ == "__main__":
    setup(**configuration(top_path='').todict())
