#!/usr/bin/env python3

from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('uvar', parent_package, top_path)
    config.add_subpackage('uvmodels')
    config.add_data_dir('tests')
    return config


if __name__ == "__main__":
    setup(**configuration(top_path='').todict())
