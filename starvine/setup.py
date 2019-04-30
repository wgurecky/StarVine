
import sys


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('starvine', parent_package, top_path)
    config.add_subpackage('bvcopula')
    config.add_subpackage('mvar')
    config.add_subpackage('uvar')
    config.add_subpackage('vine')
    config.make_config_py()
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
