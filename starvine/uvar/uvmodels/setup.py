#!/usr/bin/env python3

from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('uvmodels', parent_package, top_path)
    return config


if __name__ == "__main__":
    setup(**configuration(top_path='').todict())
