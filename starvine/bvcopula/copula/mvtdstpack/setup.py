#!/usr/bin/env python2

from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('mvtdstpack', parent_package, top_path)
    config.add_extension('mvtdstpack',
                         sources=['mvtdstpack_custom.pyf',
                                  'mvtdstpack.f'])
    return config


if __name__ == "__main__":
    setup(**configuration(top_path='').todict())
