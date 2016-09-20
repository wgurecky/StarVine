##
# \breif Copula function rotation helpers
#
# These helpers must be implemented outside of
# copula_base since we need access to them in all
# our child copula classes as decorators.
#
# Rotate the data before fitting copula
#
# Always rotate data to original orientation after
# evaluation of copula functions

def rotatePDF(input_pdf):
    def rotatedFn(self, *args, **kwargs):
        if args[2] == 0:
            # 0 deg rotation (no action)
            return input_pdf(self, *args, **kwargs)
        if args[2] == 1:
            # 90 deg rotation (flip U)
            return input_pdf(self, *args, **kwargs)
        if args[2] == 2:
            # 180 deg rotation
            # TODO: Implement
            return input_pdf(self, *args, **kwargs)
        if args[2] == 3:
            # 180 deg rotation
            # TODO: Implement
            return input_pdf(self, *args, **kwargs)
    return rotatedFn

def rotateCDF(input_cdf):
    def rotatedFn(self, *args, **kwargs):
        if args[2] == 0:
            # 0 deg rotation (no action)
            return input_cdf(self, *args, **kwargs)
        if args[2] == 1:
            # 90 deg rotation (flip U)
            return input_cdf(self, *args, **kwargs)
    return rotatedFn

def rotateHfun(input_h):
    """!
    H fun provides U given v
    """
    def rotatedFn(self, *args, **kwargs):
        if args[2] == 0:
            # 0 deg rotation (no action)
            return input_h(self, *args, **kwargs)
        if args[2] == 1:
            # 90 deg rotation (flip U)
            return input_h(self, *args, **kwargs)
    return rotatedFn

def rotateVFun(input_v):
    """!
    V fun provides V given u
    """
    def rotatedFn(self, *args, **kwargs):
        if args[2] == 0:
            # 0 deg rotation (no action)
            return input_v(self, *args, **kwargs)
        if args[2] == 1:
            # 90 deg rotation (no action)
            return input_v(self, *args, **kwargs)
    return rotatedFn
